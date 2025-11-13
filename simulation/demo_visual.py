"""
Visual demonstration of the full embodied intelligence system.
Shows the robot navigating with multimodal sensors, LLM planning, and safety filtering.
"""

from embodied_sim import EmbodiedRobotSimulation
import time


def run_visual_demo():
    """
    Run a visual demonstration showing all system components.
    """
    print("\n" + "="*80)
    print(" VISUAL DEMONSTRATION")
    print(" Watch the robot navigate with full sensor suite and safety filtering")
    print("="*80)
    
    # Create simulation with GUI
    sim = EmbodiedRobotSimulation(
        headless=False,  # Show GUI
        use_real_llm=False,  # Mock LLM for reproducibility
        enable_sensors=True,
        enable_adaptive_control=True
    )
    
    try:
        # Setup environment
        obstacle_pos = [2, 0, 0.5]
        obstacle_radius = 0.5
        sim.add_obstacle(position=obstacle_pos, radius=obstacle_radius)
        
        goal_pos = [4, 0, 0.5]
        sim.add_goal_marker(goal_pos)
        
        print("\n" + "="*80)
        print(" DEMO 1: WITHOUT Safety Filter")
        print("="*80)
        print("\nWatch the robot attempt to reach the goal WITHOUT safety filtering...")
        print("(It may collide with the obstacle)\n")
        
        input("Press Enter to start Demo 1...")
        
        stats_no_safety = sim.run_episode(
            goal_position=goal_pos,
            max_steps=1000,
            use_safety_filter=False
        )
        
        print(f"\nResults WITHOUT safety:")
        print(f"  Success: {stats_no_safety['success']}")
        print(f"  Safety violations: {stats_no_safety['safety_violations']}")
        print(f"  Final distance: {stats_no_safety['final_distance']:.3f}m")
        
        time.sleep(3)
        
        # Reset robot
        import pybullet as p
        p.resetBasePositionAndOrientation(
            sim.robot_id,
            [0, 0, 0.5],
            p.getQuaternionFromEuler([0, 0, 0])
        )
        p.resetBaseVelocity(sim.robot_id, [0, 0, 0], [0, 0, 0])
        
        print("\n" + "="*80)
        print(" DEMO 2: WITH Safety Filter + Sensors + Adaptive Control")
        print("="*80)
        print("\nNow watch with the full system enabled:")
        print("  ✓ Multimodal sensor fusion (LiDAR, Proprioception)")
        print("  ✓ LLM-based high-level planning")
        print("  ✓ Adaptive control gain scheduling")
        print("  ✓ CBF safety filter")
        print("\n(The robot should avoid the obstacle safely)\n")
        
        input("Press Enter to start Demo 2...")
        
        stats_with_safety = sim.run_episode(
            goal_position=goal_pos,
            max_steps=1000,
            use_safety_filter=True
        )
        
        print(f"\nResults WITH full system:")
        print(f"  Success: {stats_with_safety['success']}")
        print(f"  Safety violations: {stats_with_safety['safety_violations']}")
        print(f"  Final distance: {stats_with_safety['final_distance']:.3f}m")
        print(f"  LLM confidence: {stats_with_safety['llm_confidence']:.2f}")
        print(f"  Waypoints planned: {len(stats_with_safety['waypoints'])}")
        
        print("\n" + "="*80)
        print(" COMPARISON")
        print("="*80)
        print(f"  Safety improvement: {stats_no_safety['safety_violations'] - stats_with_safety['safety_violations']} fewer violations")
        print(f"  Distance improvement: {stats_no_safety['final_distance'] - stats_with_safety['final_distance']:.3f}m closer to goal")
        
        print("\n✓ Demonstration complete!")
        print("\nThe full experimental suite will run many trials like this")
        print("(but in headless mode for speed)")
        
        input("\nPress Enter to close...")
        
    finally:
        sim.close()


if __name__ == "__main__":
    print("\n" + "="*80)
    print(" EMBODIED INTELLIGENCE SYSTEM - VISUAL DEMO")
    print(" Multimodal Sensors + LLM Planning + Adaptive Control + CBF Safety")
    print("="*80)
    print("\nThis demo will show you:")
    print("  1. Robot navigation WITHOUT safety filter (may collide)")
    print("  2. Robot navigation WITH full system (avoids obstacle)")
    print("\nYou'll see the PyBullet GUI with the robot (blue sphere),")
    print("obstacle (red sphere), and goal (green sphere).")
    print("\n" + "="*80 + "\n")
    
    run_visual_demo()
