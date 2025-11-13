"""
Parallel Experiment Runner - Maximizes CPU Utilization

Uses multiprocessing to run experiments in parallel across all available CPU cores.
Master-worker architecture for efficient scheduling and data collection.
"""

import sys
sys.path.append('simulation')

import multiprocessing as mp
from multiprocessing import Pool, Manager, Lock
import numpy as np
import pybullet as p
import time
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple

from test_scenarios import TestScenarios
from baseline_controller import BaselineController
from embodied_sim import EmbodiedRobotSimulation


def run_single_trial_worker(args):
    """
    Worker function to run a single trial in isolation.
    Each worker gets its own PyBullet instance to avoid conflicts.
    
    Args:
        args: Tuple of (scenario_dict, method_name, trial_num, scenario_idx, worker_id)
    
    Returns:
        Dict with trial results
    """
    scenario_dict, method_name, trial_num, scenario_idx, worker_id = args
    
    # Each worker must initialize its own PyBullet connection
    physics_client = p.connect(p.DIRECT)  # Headless mode for speed
    p.setGravity(0, 0, -9.81, physicsClientId=physics_client)
    p.setTimeStep(1/240, physicsClientId=physics_client)
    
    try:
        # Create ground plane
        p.createCollisionShape(p.GEOM_PLANE, physicsClientId=physics_client)
        p.createMultiBody(0, 0, physicsClientId=physics_client)
        
        # Create obstacles
        obstacle_ids = []
        for obs in scenario_dict['obstacles']:
            pos = obs['position']
            radius = obs['radius']
            
            col_shape = p.createCollisionShape(
                p.GEOM_CYLINDER,
                radius=radius,
                height=1.0,
                physicsClientId=physics_client
            )
            vis_shape = p.createVisualShape(
                p.GEOM_CYLINDER,
                radius=radius,
                length=1.0,
                rgbaColor=[0.5, 0.5, 0.5, 1],
                physicsClientId=physics_client
            )
            
            obs_id = p.createMultiBody(
                baseMass=0,
                baseCollisionShapeIndex=col_shape,
                baseVisualShapeIndex=vis_shape,
                basePosition=[pos[0], pos[1], 0.5],
                physicsClientId=physics_client
            )
            obstacle_ids.append(obs_id)
        
        # Initialize controller based on method
        start_time = time.time()
        
        if method_name == "Baseline (PID only)":
            controller = BaselineController(
                robot_urdf="robot_with_sensors.urdf",
                start_position=scenario_dict['start'],
                physics_client=physics_client
            )
            max_steps = 10000
        else:  # Enhanced
            controller = EmbodiedRobotSimulation(
                robot_urdf="robot_with_sensors.urdf",
                start_position=scenario_dict['start'],
                physics_client=physics_client
            )
            max_steps = 15000
        
        # Run episode
        stats = controller.run_episode(
            goal_position=scenario_dict['goal'],
            max_steps=max_steps
        )
        
        computation_time = time.time() - start_time
        
        # Calculate metrics
        direct_distance = np.linalg.norm(
            np.array(scenario_dict['goal'][:2]) - np.array(scenario_dict['start'][:2])
        )
        
        path_efficiency = 0.0
        if stats['path_length'] > 0:
            path_efficiency = direct_distance / stats['path_length']
        
        # Compile result
        result = {
            'method': method_name,
            'scenario': scenario_dict['name'],
            'trial': trial_num,
            'scenario_idx': scenario_idx,
            'worker_id': worker_id,
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            
            # Performance metrics
            'success': stats['success'],
            'final_distance': stats['final_distance'],
            'steps': stats['steps'],
            'path_length': stats['path_length'],
            'computation_time': computation_time,
            
            # Efficiency metrics
            'direct_distance': float(direct_distance),
            'path_efficiency': float(path_efficiency),
            
            # Safety metrics
            'safety_violations': stats.get('safety_violations', 0),
            'min_obstacle_distance': stats.get('min_obstacle_distance', float('inf')),
            
            # Additional data for analysis
            'trajectory': stats.get('trajectory', []),
            'adaptive_gains': stats.get('adaptive_gains', {}),
            'sensor_data': stats.get('sensor_data', {}),
        }
        
        # Clean up PyBullet connection
        p.disconnect(physicsClientId=physics_client)
        
        return result
        
    except Exception as e:
        # Clean up on error
        try:
            p.disconnect(physicsClientId=physics_client)
        except:
            pass
        
        return {
            'error': str(e),
            'method': method_name,
            'scenario': scenario_dict['name'],
            'trial': trial_num,
            'worker_id': worker_id,
            'success': False
        }


def convert_numpy_types(obj):
    """Recursively convert numpy types to native Python types for JSON serialization."""
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: convert_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [convert_numpy_types(item) for item in obj]
    else:
        return obj


class ParallelExperimentMaster:
    """Master scheduler for parallel experiment execution."""
    
    def __init__(self, num_workers=None, output_dir="results/experiments"):
        """
        Initialize parallel experiment master.
        
        Args:
            num_workers: Number of parallel workers (default: CPU count - 1)
            output_dir: Directory to save results
        """
        if num_workers is None:
            # Use all cores except 1 for system responsiveness
            self.num_workers = max(1, mp.cpu_count() - 1)
        else:
            self.num_workers = num_workers
        
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"🚀 Parallel Experiment Master Initialized")
        print(f"   CPU Cores Available: {mp.cpu_count()}")
        print(f"   Workers to Use: {self.num_workers}")
        print(f"   Expected Speedup: ~{self.num_workers}x")
        print()
    
    def create_task_list(self, trials_per_scenario=20, existing_results=None):
        """
        Create complete list of tasks to execute.
        Skip tasks that are already completed in existing_results.
        
        Returns:
            List of (scenario_dict, method_name, trial_num, scenario_idx, worker_id) tuples
        """
        scenario_names = TestScenarios.get_all_scenarios()
        methods = ["Baseline (PID only)", "Enhanced (Sensors+LLM+Adaptive+Safety)"]
        
        tasks = []
        completed_set = set()
        
        # Track what's already done
        if existing_results:
            for result in existing_results:
                key = (result['scenario'], result['method'], result['trial'])
                completed_set.add(key)
        
        # Generate task list
        for scenario_idx, scenario_name in enumerate(scenario_names):
            # Get scenario dict
            scenario = TestScenarios.get_scenario(scenario_name)
            
            # Convert scenario to serializable dict
            scenario_dict = {
                'name': scenario['name'],
                'start': scenario['start'],
                'goal': scenario['goal'],
                'obstacles': scenario['obstacles']
            }
            
            for method in methods:
                for trial_num in range(1, trials_per_scenario + 1):
                    # Check if already completed
                    key = (scenario['name'], method, trial_num)
                    if key not in completed_set:
                        tasks.append((scenario_dict, method, trial_num, scenario_idx, 0))
        
        return tasks
    
    def run_parallel_experiments(self, trials_per_scenario=20, existing_results=None):
        """
        Run all experiments in parallel using worker pool.
        
        Args:
            trials_per_scenario: Number of trials per scenario
            existing_results: Previously completed results to continue from
            
        Returns:
            List of all results (existing + new)
        """
        # Create task list
        tasks = self.create_task_list(trials_per_scenario, existing_results)
        
        if not tasks:
            print("✅ All experiments already completed!")
            return existing_results or []
        
        print(f"📋 Total tasks to execute: {len(tasks)}")
        print(f"⏭️  Tasks already completed: {len(existing_results) if existing_results else 0}")
        print(f"🔄 Tasks remaining: {len(tasks)}")
        print()
        
        # Estimate time
        avg_time_per_trial = 25  # seconds (conservative estimate)
        total_time_sequential = len(tasks) * avg_time_per_trial
        total_time_parallel = total_time_sequential / self.num_workers
        
        print(f"⏱️  Estimated time:")
        print(f"   Sequential: {total_time_sequential/60:.1f} minutes")
        print(f"   Parallel ({self.num_workers} workers): {total_time_parallel/60:.1f} minutes")
        print(f"   Speedup: {self.num_workers:.1f}x faster!")
        print()
        
        # Start parallel execution
        print("=" * 80)
        print("🚀 STARTING PARALLEL EXECUTION")
        print("=" * 80)
        print()
        
        start_time = time.time()
        all_results = existing_results.copy() if existing_results else []
        
        # Use multiprocessing Pool for parallel execution
        with Pool(processes=self.num_workers) as pool:
            # Run tasks in parallel with progress tracking
            completed = 0
            save_interval = 10  # Save every 10 completed trials
            
            for result in pool.imap_unordered(run_single_trial_worker, tasks):
                completed += 1
                all_results.append(result)
                
                # Progress update
                progress = 100 * completed / len(tasks)
                elapsed = time.time() - start_time
                eta = (elapsed / completed) * (len(tasks) - completed) if completed > 0 else 0
                
                print(f"[PROGRESS] {completed}/{len(tasks)} ({progress:.1f}%)")
                print(f"[TIME] Elapsed: {elapsed/60:.1f}m | ETA: {eta/60:.1f}m")
                
                if result.get('error'):
                    print(f"[ERROR] Worker {result.get('worker_id', '?')}: {result['error']}")
                else:
                    status = "✅ SUCCESS" if result['success'] else "❌ FAILED"
                    print(f"[RESULT] {result['scenario']} - {result['method'][:20]}... Trial {result['trial']}: {status}")
                print()
                
                # Periodic saving
                if completed % save_interval == 0:
                    self.save_results(all_results, f"intermediate_results_trial{len(all_results)}.json")
                    print(f"💾 Intermediate results saved ({len(all_results)} trials)")
                    print()
        
        total_time = time.time() - start_time
        
        print()
        print("=" * 80)
        print("✅ PARALLEL EXECUTION COMPLETED!")
        print("=" * 80)
        print(f"Total time: {total_time/60:.1f} minutes")
        print(f"Average per trial: {total_time/len(tasks):.1f} seconds")
        print(f"Speedup achieved: {(len(tasks)*avg_time_per_trial)/total_time:.1f}x")
        print()
        
        # Save final results
        self.save_results(all_results, "final_experiment_results.json")
        
        return all_results
    
    def save_results(self, results, filename):
        """Save results to JSON file."""
        filepath = self.output_dir / filename
        
        # Convert numpy types for JSON serialization
        serializable_results = convert_numpy_types(results)
        
        with open(filepath, 'w') as f:
            json.dump(serializable_results, f, indent=2)
        
        print(f"💾 Saved: {filepath}")


def main():
    """Main entry point for parallel experiment execution."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Run parallel experiments')
    parser.add_argument('--trials', type=int, default=20,
                        help='Number of trials per scenario (default: 20)')
    parser.add_argument('--workers', type=int, default=None,
                        help='Number of parallel workers (default: CPU count - 1)')
    parser.add_argument('--resume', action='store_true',
                        help='Resume from latest intermediate backup')
    
    args = parser.parse_args()
    
    # Load existing results if resuming
    existing_results = None
    if args.resume:
        experiments_dir = Path("results/experiments")
        intermediate_files = sorted(experiments_dir.glob("intermediate_results_trial*.json"))
        
        if intermediate_files:
            latest_backup = intermediate_files[-1]
            print(f"📁 Resuming from: {latest_backup.name}")
            with open(latest_backup, 'r') as f:
                existing_results = json.load(f)
            print(f"✅ Loaded {len(existing_results)} completed trials")
            print()
    
    # Create master and run experiments
    master = ParallelExperimentMaster(num_workers=args.workers)
    results = master.run_parallel_experiments(
        trials_per_scenario=args.trials,
        existing_results=existing_results
    )
    
    # Summary
    print()
    print("=" * 80)
    print("EXPERIMENT SUMMARY")
    print("=" * 80)
    
    total_trials = len(results)
    successful = sum(1 for r in results if r.get('success', False))
    
    print(f"Total trials: {total_trials}")
    print(f"Successful: {successful} ({100*successful/total_trials:.1f}%)")
    print()
    
    # Success by method
    methods = {}
    for r in results:
        method = r.get('method', 'Unknown')
        if method not in methods:
            methods[method] = {'total': 0, 'success': 0}
        methods[method]['total'] += 1
        if r.get('success', False):
            methods[method]['success'] += 1
    
    print("Success rates by method:")
    for method, stats in methods.items():
        rate = 100 * stats['success'] / stats['total']
        print(f"  {method}: {stats['success']}/{stats['total']} ({rate:.1f}%)")
    
    print()
    print("✅ All results saved to: results/experiments/final_experiment_results.json")
    print()


if __name__ == "__main__":
    main()
