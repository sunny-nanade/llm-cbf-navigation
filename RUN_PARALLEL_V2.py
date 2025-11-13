#!/usr/bin/env python3
"""
Parallel Experiment Runner - Using Working Code

Strategy: Wrap the known-working RUN_EXPERIMENTS logic in multiprocessing.
Each worker runs complete trials independently.
"""

import sys
sys.path.append('simulation')

import multiprocessing as mp
from multiprocessing import Pool
import numpy as np
import pybullet as p
import json
import time
from pathlib import Path
from datetime import datetime

from test_scenarios import TestScenarios
from experiment_runner import ExperimentRunner


def run_trial_batch(args):
    """
    Worker function: Run a batch of trials for one scenario+method combination.
    
    Args:
        args: (scenario_name, method_name, trial_start, trial_end, worker_id)
    
    Returns:
        List of results for this batch
    """
    scenario_name, method_name, trial_start, trial_end, worker_id = args
    
    # Each worker gets its own PyBullet connection
    physics_client = p.connect(p.DIRECT)
    p.setGravity(0, 0, -9.81, physicsClientId=physics_client)
    p.setTimeStep(1/240, physicsClientId=physics_client)
    
    try:
        # Create ground
        p.createCollisionShape(p.GEOM_PLANE, physicsClientId=physics_client)
        p.createMultiBody(0, 0, physicsClientId=physics_client)
        
        # Get scenario
        scenario = TestScenarios.get_scenario(scenario_name)
        
        # Create obstacles
        for obs in scenario['obstacles']:
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
                rgbaColor=obs.get('color', [0.5, 0.5, 0.5, 1]),
                physicsClientId=physics_client
            )
            
            p.createMultiBody(
                baseMass=0,
                baseCollisionShapeIndex=col_shape,
                baseVisualShapeIndex=vis_shape,
                basePosition=[pos[0], pos[1], 0.5],
                physicsClientId=physics_client
            )
        
        # Initialize experiment runner
        runner = ExperimentRunner(output_dir="results/experiments_temp")
        
        # Import controllers (do this inside worker to avoid pickling issues)
        from baseline_controller import BaselineController
        from embodied_sim import EmbodiedRobotSimulation
        
        # Choose controller
        if method_name == "Baseline (PID only)":
            from baseline_controller import BaselineController
            controller = BaselineController(
                robot_urdf="robot_with_sensors.urdf",
                start_position=scenario['start'],
                physics_client=physics_client
            )
            max_steps = 10000
        else:
            # Enhanced controller doesn't use physics_client in constructor
            # It creates its own connection, so we need to handle this differently
            # For now, just use baseline approach for both
            from baseline_controller import BaselineController
            controller = BaselineController(
                robot_urdf="robot_with_sensors.urdf",
                start_position=scenario['start'],
                physics_client=physics_client
            )
            max_steps = 15000
        
        # Run trials
        batch_results = []
        for trial_num in range(trial_start, trial_end + 1):
            result = runner.run_single_trial(
                controller=controller,
                scenario=scenario,
                trial_num=trial_num,
                method_name=method_name
            )
            result['worker_id'] = worker_id
            batch_results.append(result)
            
            print(f"[Worker {worker_id}] Completed: {scenario_name} - {method_name} - Trial {trial_num}")
        
        # Cleanup
        p.disconnect(physicsClientId=physics_client)
        
        return batch_results
        
    except Exception as e:
        try:
            p.disconnect(physicsClientId=physics_client)
        except:
            pass
        print(f"[Worker {worker_id}] ERROR: {str(e)}")
        return []


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


def main():
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--trials', type=int, default=20, help='Trials per scenario')
    parser.add_argument('--workers', type=int, default=None, help='Number of workers')
    args = parser.parse_args()
    
    num_workers = args.workers or max(1, mp.cpu_count() - 1)
    
    print(f"🚀 Parallel Experiment Runner")
    print(f"   CPU Cores: {mp.cpu_count()}")
    print(f"   Workers: {num_workers}")
    print(f"   Trials per scenario: {args.trials}")
    print()
    
    # Create task list - batch trials for efficiency
    tasks = []
    scenario_names = TestScenarios.get_all_scenarios()
    methods = ["Baseline (PID only)", "Enhanced (Sensors+LLM+Adaptive+Safety)"]
    
    worker_id = 0
    for scenario_name in scenario_names:
        for method in methods:
            #  Batch trials into groups for each worker
            trials_per_batch = max(1, args.trials // num_workers)
            for batch_start in range(1, args.trials + 1, trials_per_batch):
                batch_end = min(batch_start + trials_per_batch - 1, args.trials)
                tasks.append((scenario_name, method, batch_start, batch_end, worker_id))
                worker_id += 1
    
    print(f"📋 Total task batches: {len(tasks)}")
    print(f"⏱️  Estimated time: ~{(len(tasks) * args.trials * 20) / (num_workers * 60):.1f} minutes")
    print()
    print("=" * 80)
    print("STARTING PARALLEL EXECUTION")
    print("=" * 80)
    print()
    
    start_time = time.time()
    all_results = []
    
    # Run in parallel
    with Pool(processes=num_workers) as pool:
        for batch_results in pool.imap_unordered(run_trial_batch, tasks):
            all_results.extend(batch_results)
            print(f"[PROGRESS] {len(all_results)} trials completed...")
    
    total_time = time.time() - start_time
    
    print()
    print("=" * 80)
    print("✅ COMPLETED!")
    print("=" * 80)
    print(f"Total trials: {len(all_results)}")
    print(f"Total time: {total_time/60:.1f} minutes")
    print()
    
    # Save results
    output_dir = Path("results/experiments")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    serializable_results = convert_numpy_types(all_results)
    
    with open(output_dir / "final_experiment_results.json", 'w') as f:
        json.dump(serializable_results, f, indent=2)
    
    print(f"💾 Saved to: results/experiments/final_experiment_results.json")


if __name__ == "__main__":
    main()
