import os
import json
import numpy as np
import pybullet as p
import time
from main import Simulation
from control import PIDController, SafetyFilter, system_dynamics, obstacle_barrier_function, mock_llm_planner

def run_scenario(headless=True, use_safety_filter=False, duration=20):
    """
    Runs a single simulation scenario.

    Args:
        headless (bool): Whether to run in headless mode.
        use_safety_filter (bool): Whether to use the CBF safety filter.
        duration (int): Duration of the simulation in seconds.

    Returns:
        dict: A dictionary containing the simulation results.
    """
    sim = Simulation(headless=headless)
    
    # --- Experiment Setup ---
    start_pos = np.array([0.0, 0.0])
    goal_pos = np.array([5.0, 5.0])
    obstacle_pos = np.array([2.5, 2.5])
    obstacle_radius = 0.5
    
    # Get waypoints from the mock LLM
    waypoints = mock_llm_planner(start_pos, goal_pos, obstacle_pos, obstacle_radius)
    current_waypoint_index = 1
    current_goal = np.array(waypoints[current_waypoint_index])

    # Add a visual marker for the obstacle and goal
    vis_shape_id = p.createVisualShape(p.GEOM_SPHERE, radius=obstacle_radius, rgbaColor=[1, 0, 0, 0.7])
    p.createMultiBody(baseVisualShapeIndex=vis_shape_id, basePosition=np.hstack([obstacle_pos, 0.25]))
    goal_vis_shape_id = p.createVisualShape(p.GEOM_SPHERE, radius=0.1, rgbaColor=[0, 1, 0, 0.7])
    p.createMultiBody(baseVisualShapeIndex=goal_vis_shape_id, basePosition=np.hstack([goal_pos, 0.25]))


    # Controllers
    pid_x = PIDController(kp=1.0, ki=0.1, kd=0.5, setpoint=current_goal[0])
    pid_y = PIDController(kp=1.0, ki=0.1, kd=0.5, setpoint=current_goal[1])
    
    # Safety Filter
    safety_filter = SafetyFilter(dim=2, alpha=0.8)

    # Data logging
    log = {
        "time": [], "pos": [], "vel": [], "u_ref": [], "u_safe": [],
        "h": [], "safety_violations": 0, "goal_reached": False, "waypoints": waypoints
    }

    try:
        for t in np.arange(0, duration, sim.timestep):
            pos, _, vel, _ = sim.get_robot_state()
            robot_pos = pos[:2]
            robot_vel = vel[:2]

            # --- Waypoint Navigation ---
            dist_to_waypoint = np.linalg.norm(robot_pos - current_goal)
            if dist_to_waypoint < 0.3 and current_waypoint_index < len(waypoints) - 1:
                current_waypoint_index += 1
                current_goal = np.array(waypoints[current_waypoint_index])
                pid_x.setpoint = current_goal[0]
                pid_y.setpoint = current_goal[1]
            
            # --- Performance Control (PID) ---
            u_ref_x = pid_x.compute(robot_pos[0], sim.timestep)
            u_ref_y = pid_y.compute(robot_pos[1], sim.timestep)
            u_ref = np.array([u_ref_x, u_ref_y])
            
            # --- Safety Filtering (CBF-QP) ---
            u_final = u_ref
            h_val = -1 # Default safe value
            if use_safety_filter:
                x_state = np.hstack([robot_pos, robot_vel])
                h_val, lfh, lgh = obstacle_barrier_function(x_state, obstacle_pos, obstacle_radius)
                
                u_final = safety_filter.solve(u_ref, h_val, lfh, lgh)
                
                if h_val < 0:
                    log["safety_violations"] += 1
            
            # Apply control to robot
            p.applyExternalForce(sim.robot_id, -1, forceObj=np.hstack([u_final, 0]), posObj=pos, flags=p.WORLD_FRAME)

            # Step simulation
            p.stepSimulation()
            
            # Log data
            log["time"].append(t)
            log["pos"].append(robot_pos.tolist())
            log["vel"].append(robot_vel.tolist())
            log["u_ref"].append(u_ref.tolist())
            log["u_safe"].append(u_final.tolist())
            log["h"].append(float(h_val))

            if not headless:
                time.sleep(sim.timestep)

        # Check if goal was reached
        final_dist_to_goal = np.linalg.norm(robot_pos - goal_pos)
        if final_dist_to_goal < 0.3:
            log["goal_reached"] = True
            
    finally:
        sim.close()
        
    return log

class NumpyArrayEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)

def main():
    """
    Main function to run experiments and save results.
    """
    output_dir = "simulation_results"
    os.makedirs(output_dir, exist_ok=True)

    print("Running baseline scenario (without safety filter)...")
    baseline_results = run_scenario(headless=True, use_safety_filter=False)
    with open(os.path.join(output_dir, "baseline.json"), "w") as f:
        json.dump(baseline_results, f, indent=4, cls=NumpyArrayEncoder)
    print("Baseline results saved.")

    print("\nRunning safety-filtered scenario...")
    safe_results = run_scenario(headless=True, use_safety_filter=True)
    with open(os.path.join(output_dir, "safe.json"), "w") as f:
        json.dump(safe_results, f, indent=4, cls=NumpyArrayEncoder)
    print("Safety-filtered results saved.")

if __name__ == "__main__":
    main()
