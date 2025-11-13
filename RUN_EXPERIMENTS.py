"""
Main Experiment Runner for Comparative Study

Runs 200 trials comparing baseline vs enhanced controller across 5 scenarios.
This is the main script for generating experimental data for the paper.
"""

import sys
sys.path.append('simulation')

import numpy as np
import pybullet as p
import time
from pathlib import Path

from test_scenarios import TestScenarios
from experiment_runner import ExperimentRunner
from baseline_controller import BaselineController
from embodied_sim import EmbodiedRobotSimulation

def setup_simulation(headless=True):
    """Initialize PyBullet simulation."""
    if headless:
        p.connect(p.DIRECT)  # No GUI for faster execution
    else:
        p.connect(p.GUI)
        # Set camera for better view
        p.resetDebugVisualizerCamera(
            cameraDistance=12.0,
            cameraYaw=45,
            cameraPitch=-35,
            cameraTargetPosition=[2.5, 2.5, 0]
        )
    
    p.setGravity(0, 0, -9.81)
    p.setTimeStep(1/240)
    
    # Create ground plane
    p.createCollisionShape(p.GEOM_PLANE)
    p.createMultiBody(0, 0)
    
    print("[OK] PyBullet simulation initialized")


def create_obstacles(scenario):
    """Create obstacles for a scenario."""
    obstacle_ids = []
    
    for obs in scenario['obstacles']:
        pos = obs['position']
        radius = obs['radius']
        color = obs['color']
        
        # Create visual and collision shape
        visual_shape = p.createVisualShape(
            p.GEOM_CYLINDER,
            radius=radius,
            length=0.5,
            rgbaColor=color
        )
        collision_shape = p.createCollisionShape(
            p.GEOM_CYLINDER,
            radius=radius,
            height=0.5
        )
        
        # Create obstacle body
        obs_id = p.createMultiBody(
            baseMass=0,  # Static obstacle
            baseCollisionShapeIndex=collision_shape,
            baseVisualShapeIndex=visual_shape,
            basePosition=pos
        )
        
        obstacle_ids.append({'id': obs_id, 'position': pos, 'radius': radius})
    
    return obstacle_ids


def reset_robot(robot_id, start_position):
    """Reset robot to starting position."""
    start_orientation = p.getQuaternionFromEuler([0, 0, 0])
    p.resetBasePositionAndOrientation(robot_id, start_position, start_orientation)
    p.resetBaseVelocity(robot_id, [0, 0, 0], [0, 0, 0])
    
    # Step simulation a few times to settle
    for _ in range(10):
        p.stepSimulation()


def run_baseline_trial(robot_id, scenario, trial_num, headless=True):
    """Run a single baseline trial."""
    print(f"\n{'='*70}")
    print(f"BASELINE | Scenario: {scenario['name']} | Trial: {trial_num}/20")
    print(f"{'='*70}")
    
    # Reset robot
    reset_robot(robot_id, scenario['start'])
    
    # Create baseline controller
    controller = BaselineController(robot_id, timestep=1/240)
    
    # Run episode
    start_time = time.time()
    stats = controller.run_episode(
        goal_position=scenario['goal'],
        max_steps=10000  # Increased to allow complex navigation in cluttered scenarios
    )
    computation_time = time.time() - start_time
    
    # Add extra metrics (convert to native Python types for JSON)
    stats['computation_time'] = float(computation_time)
    stats['method'] = 'Baseline (PID only)'
    stats['scenario'] = scenario['name']
    stats['trial'] = int(trial_num)
    
    # Compute direct distance
    direct_distance = float(np.linalg.norm(
        np.array(scenario['goal'][:2]) - np.array(scenario['start'][:2])
    ))
    stats['direct_distance'] = float(direct_distance)
    stats['path_efficiency'] = float(direct_distance / stats['path_length']) if stats.get('path_length', 0) > 0 else 0.0
    
    # Convert any remaining numpy types to Python types
    for key, value in stats.items():
        if isinstance(value, np.bool_):
            stats[key] = bool(value)
        elif isinstance(value, (np.integer, np.floating)):
            stats[key] = float(value)
    
    return stats


def run_enhanced_trial(scenario, trial_num, obstacle_ids, headless=True):
    """Run a single enhanced trial."""
    print(f"\n{'='*70}")
    print(f"ENHANCED | Scenario: {scenario['name']} | Trial: {trial_num}/20")
    print(f"{'='*70}")
    
    # Create enhanced simulation controller (creates its own robot)
    sim = EmbodiedRobotSimulation(
        enable_sensors=True,
        enable_adaptive_control=True,
        headless=headless
    )
    
    # Reset robot to start position
    reset_robot(sim.robot_id, scenario['start'])
    
    # Set obstacles for enhanced system
    sim.obstacle_ids = obstacle_ids
    
    # Run episode
    start_time = time.time()
    stats = sim.run_episode(
        goal_position=scenario['goal'],
        max_steps=15000,  # Increased for cluttered scenarios - more time to navigate dense obstacles
        use_safety_filter=True,
        sensor_noise=True,
        record_video=False  # Skip video recording during experiments for speed
    )
    computation_time = time.time() - start_time
    
    # Add extra metrics
    stats['computation_time'] = float(computation_time)
    stats['method'] = 'Enhanced (Sensors+LLM+Adaptive+Safety)'
    stats['scenario'] = scenario['name']
    stats['trial'] = int(trial_num)
    
    # Compute direct distance (convert all to native Python types)
    direct_distance = float(np.linalg.norm(
        np.array(scenario['goal'][:2]) - np.array(scenario['start'][:2])
    ))
    stats['direct_distance'] = float(direct_distance)
    stats['path_efficiency'] = float(direct_distance / stats['path_length']) if stats.get('path_length', 0) > 0 else 0.0
    
    # Convert any remaining numpy types to Python types
    for key, value in stats.items():
        if isinstance(value, np.bool_):
            stats[key] = bool(value)
        elif isinstance(value, (np.integer, np.floating)):
            stats[key] = float(value)
    
    return stats


def run_all_experiments(num_trials_per_scenario=20, headless=True):
    """
    Run complete experimental study.
    
    Args:
        num_trials_per_scenario: Number of trials to run per scenario per method (default: 20)
        headless: Run without GUI (faster)
    """
    print(f"\n{'#'*70}")
    print("# COMPREHENSIVE EXPERIMENTAL STUDY")
    print(f"# Baseline vs Enhanced Controller Comparison")
    print(f"# {num_trials_per_scenario} trials × 5 scenarios × 2 methods = {num_trials_per_scenario*5*2} total trials")
    print(f"{'#'*70}\n")
    
    # Initialize simulation
    setup_simulation(headless=headless)
    
    # Create experiment runner
    runner = ExperimentRunner(output_dir="results/experiments")
    
    # Get all scenarios
    scenario_names = TestScenarios.get_all_scenarios()
    
    # Create one robot for baseline trials (will reset position each trial)
    # Enhanced trials create new robots each time (needed for sensor initialization)
    import pybullet_data
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    urdf_path = "simulation/robot_with_sensors.urdf"  # Same robot model
    start_pos = [0, 0, 0.08]
    start_orientation = p.getQuaternionFromEuler([0, 0, 0])
    robot_id = p.loadURDF(urdf_path, start_pos, start_orientation)
    
    total_trials = 0
    total_to_run = num_trials_per_scenario * len(scenario_names) * 2
    
    # Track timing for ETA
    import time
    start_time = time.time()
    
    # Run experiments for each scenario
    for scenario_name in scenario_names:
        scenario = TestScenarios.get_scenario(scenario_name)
        TestScenarios.print_scenario_info(scenario)
        
        # Create obstacles for this scenario
        obstacle_ids = create_obstacles(scenario)
        
        print(f"\n{'*'*70}")
        print(f"* SCENARIO: {scenario['name']}")
        print(f"* {num_trials_per_scenario} baseline + {num_trials_per_scenario} enhanced = {num_trials_per_scenario*2} trials")
        print(f"{'*'*70}\n")
        
        # Run baseline trials
        print(f"\n--- BASELINE TRIALS ---")
        for trial in range(1, num_trials_per_scenario + 1):
            try:
                result = run_baseline_trial(robot_id, scenario, trial, headless)
                runner.results.append(result)
                total_trials += 1
                
                # Calculate progress and ETA
                progress = (total_trials / total_to_run) * 100
                elapsed = time.time() - start_time
                avg_time_per_trial = elapsed / total_trials
                remaining_trials = total_to_run - total_trials
                eta_seconds = avg_time_per_trial * remaining_trials
                eta_minutes = int(eta_seconds / 60)
                eta_seconds_remaining = int(eta_seconds % 60)
                
                print(f"\n{'='*70}")
                print(f"[PROGRESS] {total_trials}/{total_to_run} trials ({progress:.1f}%)")
                print(f"[TIME] Elapsed: {int(elapsed/60)}m {int(elapsed%60)}s | ETA: {eta_minutes}m {eta_seconds_remaining}s")
                print(f"{'='*70}")
                
                # Save intermediate results every 10 trials
                if total_trials % 10 == 0:
                    runner.save_results(f"intermediate_results_trial{total_trials}.json")
            except Exception as e:
                print(f"[ERROR] Trial failed: {e}")
                continue
        
        # Run enhanced trials
        print(f"\n--- ENHANCED TRIALS ---")
        for trial in range(1, num_trials_per_scenario + 1):
            try:
                result = run_enhanced_trial(scenario, trial, obstacle_ids, headless)
                runner.results.append(result)
                total_trials += 1
                
                # Calculate progress and ETA
                progress = (total_trials / total_to_run) * 100
                elapsed = time.time() - start_time
                avg_time_per_trial = elapsed / total_trials
                remaining_trials = total_to_run - total_trials
                eta_seconds = avg_time_per_trial * remaining_trials
                eta_minutes = int(eta_seconds / 60)
                eta_seconds_remaining = int(eta_seconds % 60)
                
                print(f"\n{'='*70}")
                print(f"[PROGRESS] {total_trials}/{total_to_run} trials ({progress:.1f}%)")
                print(f"[TIME] Elapsed: {int(elapsed/60)}m {int(elapsed%60)}s | ETA: {eta_minutes}m {eta_seconds_remaining}s")
                print(f"{'='*70}")
                
                # Save intermediate results every 10 trials
                if total_trials % 10 == 0:
                    runner.save_results(f"intermediate_results_trial{total_trials}.json")
            except Exception as e:
                print(f"[ERROR] Trial failed: {e}")
                continue
        
        # Clean up obstacles
        for obs in obstacle_ids:
            p.removeBody(obs['id'])
    
    # Save final results
    final_results_file = runner.save_results("final_experiment_results.json")
    
    # Print summary
    runner.print_summary()
    
    # Generate comparison table
    table_data = runner.generate_comparison_table()
    
    print(f"\n{'='*70}")
    print("EXPERIMENT COMPLETE!")
    print(f"Total trials completed: {total_trials}/{total_to_run}")
    print(f"Results saved to: {final_results_file}")
    print(f"{'='*70}\n")
    
    # Disconnect simulation
    p.disconnect()
    
    return runner


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Run comparative experiments')
    parser.add_argument('--trials', type=int, default=20, help='Trials per scenario (default: 20)')
    parser.add_argument('--gui', action='store_true', help='Show GUI (slower)')
    parser.add_argument('--quick', action='store_true', help='Quick test (2 trials per scenario)')
    
    args = parser.parse_args()
    
    if args.quick:
        print("[QUICK TEST MODE] Running 2 trials per scenario for testing")
        num_trials = 2
    else:
        num_trials = args.trials
    
    headless = not args.gui
    
    print(f"\nStarting experiments:")
    print(f"  Trials per scenario: {num_trials}")
    print(f"  Headless mode: {headless}")
    print(f"  Total trials: {num_trials * 5 * 2}\n")
    
    # Run experiments
    runner = run_all_experiments(num_trials_per_scenario=num_trials, headless=headless)
