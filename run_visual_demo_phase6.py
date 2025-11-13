"""
Visual Demo - Showcase Publication-Quality Visualization
Demonstrates all Phase 6 visual feedback elements in action.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'simulation'))

from embodied_sim import EmbodiedRobotSimulation
from test_scenarios import TestScenarios
import time


def run_visual_demo():
    """Run visual demonstration with all feedback elements enabled."""
    
    print("\n" + "="*70)
    print(" VISUAL FEEDBACK DEMONSTRATION")
    print(" Phase 6: Publication-Quality Visualization")
    print("="*70)
    print("\nThis demo showcases:")
    print("  ✓ LiDAR ray visualization (green=clear, red=obstacle)")
    print("  ✓ Safety filter status indicator (green/yellow/red sphere)")
    print("  ✓ LLM waypoint decision overlay (cyan markers + text)")
    print("  ✓ Control force arrows (green arrows)")
    print("  ✓ Adaptive gains evolution tracking")
    print("\nPress Ctrl+C to exit early\n")
    
    # Create simulation with visualization enabled
    print("[1/5] Initializing simulation with visual feedback...")
    sim = EmbodiedRobotSimulation(
        headless=False,  # MUST have GUI for visualization
        use_real_llm=False,  # Mock LLM for reproducibility
        enable_sensors=True,  # Enable multimodal sensors
        enable_adaptive_control=True,  # Enable adaptive gains
        enable_visualization=True  # *** ENABLE VISUAL FEEDBACK ***
    )
    print("      ✓ Visualization manager initialized")
    
    # Load test scenarios
    print("\n[2/5] Loading test scenarios...")
    scenarios = TestScenarios()
    available_scenarios = [
        ('easy', "No obstacles"),
        ('medium', "1 obstacle blocking path"),
        ('hard', "4 obstacles with narrow passages"),
        ('dynamic', "3 strategically placed obstacles"),
        ('cluttered', "8 obstacles in dense field")
    ]
    
    for idx, (name, desc) in enumerate(available_scenarios):
        print(f"      {idx+1}. {name.capitalize()}: {desc}")
    
    # Select scenario
    print("\n[3/5] Select scenario to visualize:")
    print("      (Using MEDIUM scenario - best for visualization demo)")
    scenario_name = 'medium'
    scenario_config = scenarios.get_scenario(scenario_name)
    print(f"      ✓ Selected: {scenario_name.capitalize()}")
    
    # Setup scenario
    print("\n[4/5] Setting up scenario environment...")
    for obs in scenario_config['obstacles']:
        sim.add_obstacle(
            position=obs['position'],
            radius=obs['radius']
        )
    print(f"      ✓ Added {len(scenario_config['obstacles'])} obstacles")
    print(f"      ✓ Start: {scenario_config['start']}")
    print(f"      ✓ Goal: {scenario_config['goal']}")
    
    # Add goal marker for visualization
    sim.add_goal_marker(scenario_config['goal'])
    
    # Run episode with visualization
    print("\n[5/5] Running episode with full visual feedback...")
    print("\n" + "-"*70)
    print("WATCH FOR:")
    print("  • Green LiDAR rays = clear space")
    print("  • Red LiDAR rays = obstacle detected")
    print("  • Sphere above robot = safety status (green/yellow/red)")
    print("  • Cyan cylinder = current waypoint target")
    print("  • Green arrows = control force direction")
    print("  • Text overlay = waypoint info, confidence, planning mode")
    print("-"*70 + "\n")
    
    time.sleep(2)  # Pause to read instructions
    
    try:
        stats = sim.run_episode(
            goal_position=scenario_config['goal'],
            max_steps=2000,
            use_safety_filter=True,  # Enable CBF safety filter
            sensor_noise=True,  # Add realistic noise
            workspace_bounds=(-5, 10, -5, 10),
            record_video=True,  # Record video for later
            video_filename=f"results/visual_demo_{scenario_name}.mp4"
        )
        
        # Print results
        print("\n" + "="*70)
        print(" DEMONSTRATION COMPLETE")
        print("="*70)
        print(f"\nResults:")
        print(f"  Success: {stats['success']}")
        print(f"  Steps: {stats['steps']}")
        print(f"  Safety Violations: {stats['safety_violations']}")
        print(f"  Path Length: {stats['path_length']:.2f}m")
        print(f"  Min Obstacle Distance: {stats['min_obstacle_distance']:.2f}m")
        print(f"  LLM Confidence: {stats['llm_confidence']:.2f}")
        
        if stats['adaptive_gains_history']:
            print(f"\nAdaptive Gains Evolution:")
            print(f"  Initial: Kp={stats['adaptive_gains_history'][0]['kp']:.1f}, "
                  f"Ki={stats['adaptive_gains_history'][0]['ki']:.1f}, "
                  f"Kd={stats['adaptive_gains_history'][0]['kd']:.1f}")
            print(f"  Final:   Kp={stats['adaptive_gains_history'][-1]['kp']:.1f}, "
                  f"Ki={stats['adaptive_gains_history'][-1]['ki']:.1f}, "
                  f"Kd={stats['adaptive_gains_history'][-1]['kd']:.1f}")
        
        print(f"\nVideo saved to: results/visual_demo_{scenario_name}.mp4")
        print("\nVisualization elements demonstrated successfully! ✓")
        
    except KeyboardInterrupt:
        print("\n\nDemo interrupted by user")
    except Exception as e:
        print(f"\n\nError during demo: {e}")
        import traceback
        traceback.print_exc()
    finally:
        print("\nCleaning up...")
        sim.close()
        print("Done!\n")


if __name__ == "__main__":
    # Ensure results directory exists
    os.makedirs("results", exist_ok=True)
    
    run_visual_demo()
