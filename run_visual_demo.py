import time
import numpy as np
import pybullet as p
from simulation.main import Simulation
from simulation.control import PIDController, SafetyFilter, system_dynamics, obstacle_barrier_function

def run_visual_demo():
    """
    Runs a single, visible simulation instance to demonstrate the behavior
    of the safety-filtered controller.
    """
    print("Starting visual demonstration...")
    sim = Simulation(headless=False)
    
    # --- Experiment Setup ---
    goal_pos = np.array([5.0, 5.0])
    obstacle_pos = np.array([2.5, 2.5])
    obstacle_radius = 0.5
    
    # Add visual markers for the obstacle and goal
    vis_obstacle_id = p.createVisualShape(p.GEOM_SPHERE, radius=obstacle_radius, rgbaColor=[1, 0, 0, 0.7])
    p.createMultiBody(baseVisualShapeIndex=vis_obstacle_id, basePosition=np.hstack([obstacle_pos, 0.25]))
    
    vis_goal_id = p.createVisualShape(p.GEOM_SPHERE, radius=0.1, rgbaColor=[0, 1, 0, 0.7])
    p.createMultiBody(baseVisualShapeIndex=vis_goal_id, basePosition=np.hstack([goal_pos, 0.1]))

    # Controllers
    pid_x = PIDController(kp=1.0, ki=0.1, kd=0.5, setpoint=goal_pos[0])
    pid_y = PIDController(kp=1.0, ki=0.1, kd=0.5, setpoint=goal_pos[1])
    
    # Safety Filter
    safety_filter = SafetyFilter(dim=2, alpha=0.8)

    try:
        for _ in range(int(20 / sim.timestep)): # Run for 20 seconds
            pos, _, vel, _ = sim.get_robot_state()
            robot_pos = pos[:2]
            robot_vel = vel[:2]
            
            # Performance Control (PID)
            u_ref_x = pid_x.compute(robot_pos[0], sim.timestep)
            u_ref_y = pid_y.compute(robot_pos[1], sim.timestep)
            u_ref = np.array([u_ref_x, u_ref_y])
            
            # Safety Filtering (CBF-QP)
            x_state = np.hstack([robot_pos, robot_vel])
            h_val, lfh, lgh = obstacle_barrier_function(x_state, obstacle_pos, obstacle_radius)
            u_final = safety_filter.solve(u_ref, h_val, lfh, lgh)
            
            # Apply control
            p.applyExternalForce(sim.robot_id, -1, forceObj=np.hstack([u_final, 0]), posObj=pos, flags=p.WORLD_FRAME)

            p.stepSimulation()
            time.sleep(sim.timestep)
            
    finally:
        print("Demonstration finished. Closing simulation.")
        sim.close()

if __name__ == "__main__":
    run_visual_demo()
