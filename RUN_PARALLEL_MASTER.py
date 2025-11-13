#!/usr/bin/env python3
"""
MASTER-WORKER EXPERIMENT RUNNER
Efficiently uses all CPU cores to run experiments in parallel.
Automatically resumes from crashes. Saves incrementally.
"""

import sys
sys.path.append('simulation')

import multiprocessing as mp
from multiprocessing import Queue, Process, Manager
import numpy as np
import pybullet as p
import json
import time
from pathlib import Path
from datetime import datetime
from queue import Empty

from test_scenarios import TestScenarios


def convert_numpy(obj):
    """Convert numpy types for JSON."""
    if isinstance(obj, np.bool_):
        return bool(obj)
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {k: convert_numpy(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [convert_numpy(i) for i in obj]
    return obj


def worker_process(worker_id, task_queue, result_queue, progress_dict):
    """
    Worker process: Takes tasks from queue, runs them, returns results.
    Each trial gets a fresh PyBullet instance to avoid state corruption.
    """
    # Import here to avoid pickling issues
    from baseline_controller import BaselineController
    from embodied_sim import EmbodiedRobotSimulation
    
    print(f"[Worker {worker_id}] Started", flush=True)
    
    # Suppress warnings
    import warnings
    warnings.filterwarnings('ignore')
    
    trials_done = 0
    
    try:
        while True:
            try:
                # Get task (timeout so we can check if done)
                task = task_queue.get(timeout=1)
                
                if task is None:  # Poison pill - shutdown signal
                    print(f"[Worker {worker_id}] Shutting down ({trials_done} trials completed)")
                    break
                
                scenario_name, method_name, trial_num = task
                
                # Get scenario
                scenario = TestScenarios.get_scenario(scenario_name)
                
                # Initialize controller based on method
                if method_name == "Baseline (PID only)":
                    # Create fresh PyBullet instance for this Baseline trial
                    physics_client = p.connect(p.DIRECT)
                    p.setAdditionalSearchPath("simulation/assets")
                    p.setGravity(0, 0, -9.81, physicsClientId=physics_client)
                    p.setTimeStep(1/240, physicsClientId=physics_client)
                    
                    # Create ground
                    p.createCollisionShape(p.GEOM_PLANE, physicsClientId=physics_client)
                    p.createMultiBody(0, 0, physicsClientId=physics_client)
                    
                    # Create obstacles
                    for obs in scenario['obstacles']:
                        col_shape = p.createCollisionShape(
                            p.GEOM_CYLINDER,
                            radius=obs['radius'],
                            height=1.0,
                            physicsClientId=physics_client
                        )
                        vis_shape = p.createVisualShape(
                            p.GEOM_CYLINDER,
                            radius=obs['radius'],
                            length=1.0,
                            rgbaColor=obs.get('color', [0.5, 0.5, 0.5, 1]),
                            physicsClientId=physics_client
                        )
                        obs_id = p.createMultiBody(
                            baseMass=0,
                            baseCollisionShapeIndex=col_shape,
                            baseVisualShapeIndex=vis_shape,
                            basePosition=[obs['position'][0], obs['position'][1], 0.5],
                            physicsClientId=physics_client
                        )
                    
                    # Load robot
                    start_ori = p.getQuaternionFromEuler([0, 0, 0], physicsClientId=physics_client)
                    robot_id = p.loadURDF(
                        "robot_with_sensors.urdf",
                        scenario['start'],
                        start_ori,
                        physicsClientId=physics_client
                    )
                    
                    controller = BaselineController(
                        robot_id=robot_id,
                        timestep=1/240,
                        physics_client=physics_client,
                        worker_id=worker_id
                    )
                    max_steps = 10000
                else:  # Enhanced - creates its own PyBullet internally
                    controller = EmbodiedRobotSimulation(
                        headless=True,
                        use_real_llm=False,
                        enable_sensors=True,
                        enable_adaptive_control=True,
                        worker_id=worker_id
                    )
                    max_steps = 15000
                
                # Run trial
                trial_start = time.time()
                stats = controller.run_episode(
                    goal_position=scenario['goal'],
                    max_steps=max_steps
                )
                trial_time = time.time() - trial_start
                
                # Clean up PyBullet for this trial
                if method_name == "Baseline (PID only)":
                    try:
                        p.disconnect(physicsClientId=physics_client)
                    except Exception:
                        pass  # Already disconnected or invalid - safe to ignore
                
                # Calculate metrics
                direct_dist = np.linalg.norm(
                    np.array(scenario['goal'][:2]) - np.array(scenario['start'][:2])
                )
                
                result = {
                    'method': method_name,
                    'scenario': scenario['name'],
                    'trial': trial_num,
                    'worker_id': worker_id,
                    'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    'success': stats['success'],
                    'final_distance': stats['final_distance'],
                    'steps': stats['steps'],
                    'path_length': stats['path_length'],
                    'computation_time': trial_time,
                    'direct_distance': float(direct_dist),
                    'path_efficiency': float(direct_dist / stats['path_length']) if stats['path_length'] > 0 else 0.0,
                    'safety_violations': stats.get('safety_violations', 0),
                    'min_obstacle_distance': stats.get('min_obstacle_distance', float('inf')),
                    'adaptive_gains_history': stats.get('adaptive_gains_history', []),
                    'sensor_data_samples': stats.get('sensor_data_samples', []),
                    'llm_interventions': stats.get('llm_interventions', 0),
                    'llm_confidence': stats.get('llm_confidence', 0.0),
                    'trajectory': stats.get('trajectory', [])
                }
                
                # Send result back
                result_queue.put(result)
                trials_done += 1
                
                # Update progress (with error handling for Windows manager cleanup)
                try:
                    progress_dict[worker_id] = trials_done
                except Exception:
                    pass  # Ignore manager connection errors during shutdown
                
            except Empty:
                continue  # No task available, try again
            except Exception as e:
                print(f"[Worker {worker_id}] ERROR: {e}", flush=True)
                import traceback
                traceback.print_exc()
                # Put error result
                result_queue.put({
                    'error': str(e),
                    'scenario': scenario_name if 'scenario_name' in locals() else 'unknown',
                    'method': method_name if 'method_name' in locals() else 'unknown',
                    'trial': trial_num if 'trial_num' in locals() else 0,
                    'worker_id': worker_id
                })
    
    finally:
        # No persistent PyBullet connection to clean up anymore
        pass


def master_process(num_workers=19, trials_per_scenario=20):
    """
    Master process: Distributes work, collects results, saves data.
    """
    print("=" * 80)
    print("PARALLEL EXPERIMENT MASTER")
    print("=" * 80)
    print(f"CPU Cores available: {mp.cpu_count()}")
    print(f"Workers to use: {num_workers}")
    print(f"Trials per scenario: {trials_per_scenario}")
    print()
    
    # Create queues
    task_queue = Queue()
    result_queue = Queue()
    
    # Shared progress tracking
    manager = Manager()
    progress_dict = manager.dict()
    
    # Load existing results if any
    experiments_dir = Path("results/experiments")
    experiments_dir.mkdir(parents=True, exist_ok=True)
    
    intermediate_files = sorted(experiments_dir.glob("intermediate_results_trial*.json"))
    
    existing_results = []
    completed_set = set()
    
    if intermediate_files:
        latest = intermediate_files[-1]
        print(f"[*] Found existing backup: {latest.name}")
        with open(latest, 'r') as f:
            existing_results = json.load(f)
        
        for r in existing_results:
            completed_set.add((r['scenario'], r['method'], r['trial']))
        
        print(f"[OK] Loaded {len(existing_results)} completed trials")
    else:
        print("[*] No existing backup - starting fresh")
    
    print()
    
    # Build task list (only incomplete trials)
    scenario_names = TestScenarios.get_all_scenarios()  # ['easy', 'medium', 'hard', 'dynamic', 'cluttered']
    methods = ["Baseline (PID only)", "Enhanced (Sensors+LLM+Adaptive+Safety)"]
    
    tasks = []
    for scenario_name in scenario_names:
        scenario = TestScenarios.get_scenario(scenario_name)
        for method in methods:
            for trial_num in range(1, trials_per_scenario + 1):
                key = (scenario['name'], method, trial_num)
                if key not in completed_set:
                    # Store short name for worker, not full name
                    tasks.append((scenario_name, method, trial_num))
    
    total_needed = len(scenario_names) * len(methods) * trials_per_scenario
    total_remaining = len(tasks)
    
    print(f"Total trials needed: {total_needed}")
    print(f"Already completed: {len(existing_results)}")
    print(f"Remaining: {total_remaining}")
    print()
    
    if total_remaining == 0:
        print("[OK] All experiments already completed!")
        return existing_results
    
    # Add tasks to queue
    print(f"[#] Adding {total_remaining} tasks to queue...")
    for task in tasks:
        task_queue.put(task)
    
    # Add poison pills (shutdown signals)
    for _ in range(num_workers):
        task_queue.put(None)
    
    print(f"[OK] Task queue ready")
    print()
    
    # Start workers
    workers = []
    for i in range(num_workers):
        w = Process(target=worker_process, args=(i, task_queue, result_queue, progress_dict))
        w.start()
        workers.append(w)
        progress_dict[i] = 0
    
    print(f"[>>] Started {num_workers} workers")
    print()
    print("=" * 80)
    print("RUNNING EXPERIMENTS")
    print("=" * 80)
    print()
    
    # Collect results
    start_time = time.time()
    save_interval = 10
    last_save_count = len(existing_results)
    
    all_results = existing_results.copy()
    completed_count = len(existing_results)
    
    while completed_count < total_needed:
        try:
            result = result_queue.get(timeout=2)
            
            if 'error' in result:
                print(f"[X] [Worker {result['worker_id']}] Error in {result['scenario']} - {result['method']} Trial {result['trial']}")
            else:
                all_results.append(result)
                completed_count += 1
                
                # Progress update
                progress_pct = 100 * completed_count / total_needed
                elapsed = time.time() - start_time
                
                if completed_count > len(existing_results):
                    new_trials = completed_count - len(existing_results)
                    avg_time = elapsed / new_trials
                    eta = avg_time * (total_needed - completed_count)
                    
                    status = "SUCCESS" if result['success'] else "FAILED"
                    worker_id = result.get('worker_id', '?')
                    print(f"{completed_count} of 200 completed by Worker {worker_id} | Status: {status} | ETA: {eta/60:.1f} min", flush=True)
                
                # Periodic save
                if completed_count - last_save_count >= save_interval:
                    save_path = experiments_dir / f"intermediate_results_trial{completed_count}.json"
                    with open(save_path, 'w') as f:
                        json.dump(convert_numpy(all_results), f, indent=2)
                    print(f"[SAVE] Saved intermediate: {save_path.name}", flush=True)
                    last_save_count = completed_count
                    
        except Empty:
            # Check if all workers are done
            if all(not w.is_alive() for w in workers):
                break
    
    # Wait for all workers to finish
    for w in workers:
        w.join(timeout=5)
    
    # Final save
    final_path = experiments_dir / "final_experiment_results.json"
    with open(final_path, 'w') as f:
        json.dump(convert_numpy(all_results), f, indent=2)
    
    total_time = time.time() - start_time
    
    print()
    print("=" * 80)
    print("[OK] EXPERIMENTS COMPLETED!")
    print("=" * 80)
    print(f"Total trials: {len(all_results)}")
    print(f"New trials: {len(all_results) - len(existing_results)}")
    print(f"Total time: {total_time/60:.1f} minutes")
    print(f"Average per trial: {total_time/max(1, len(all_results) - len(existing_results)):.1f} seconds")
    print(f"[SAVE] Saved to: {final_path}")
    print()
    
    # Summary
    successful = sum(1 for r in all_results if r.get('success', False))
    print(f"Overall success rate: {successful}/{len(all_results)} ({100*successful/len(all_results):.1f}%)")
    
    # By method
    for method in methods:
        method_results = [r for r in all_results if r['method'] == method]
        method_success = sum(1 for r in method_results if r.get('success', False))
        print(f"  {method}: {method_success}/{len(method_results)} ({100*method_success/len(method_results):.1f}%)")
    
    return all_results


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Run parallel experiments')
    parser.add_argument('--trials', type=int, default=20, help='Trials per scenario')
    parser.add_argument('--workers', type=int, default=19, help='Number of worker processes')
    
    args = parser.parse_args()
    
    # Run experiments
    results = master_process(
        num_workers=args.workers,
        trials_per_scenario=args.trials
    )
    
    print()
    print("[DONE] ALL DONE! Ready for analysis.")

