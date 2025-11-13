"""
Baseline PID controller for comparison experiments.

This is a simple controller WITHOUT:
- Multimodal sensors
- LLM planning
- Adaptive control gains
- Safety filters

Used as baseline to demonstrate the benefits of the enhanced system.
"""
import numpy as np
import pybullet as p
import time


class PIDController:
    """Simple PID controller."""
    def __init__(self, kp, ki, kd, setpoint=0):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.setpoint = setpoint
        self.integral = 0
        self.prev_error = 0
        
    def compute(self, measurement, dt):
        """Compute control output."""
        error = self.setpoint - measurement
        self.integral += error * dt
        derivative = (error - self.prev_error) / dt if dt > 0 else 0
        self.prev_error = error
        
        return self.kp * error + self.ki * self.integral + self.kd * derivative


class BaselineController:
    """
    Baseline robot controller with only basic PID control.
    No sensors, no LLM planning, no adaptive gains, no safety filter.
    """
    
    def __init__(self, robot_id, timestep=1/240, physics_client=None, worker_id=None):
        self.robot_id = robot_id
        self.timestep = timestep
        self.physics_client = physics_client  # Store for multi-client support
        self.worker_id = worker_id  # For logging identification
        self.trajectory = []
        
    def get_robot_position(self):
        """Get current robot position (only basic proprioception)."""
        pos, _ = p.getBasePositionAndOrientation(self.robot_id, physicsClientId=self.physics_client)
        return np.array(pos[:2])  # Only X, Y
    
    def apply_control(self, force):
        """Apply 2D control force to robot."""
        # Same velocity-based control as enhanced system for fair comparison
        max_velocity = 5.0
        desired_vel = np.array([force[0], force[1]]) * 2.0
        
        # Clamp to maximum velocity
        vel_magnitude = np.linalg.norm(desired_vel)
        if vel_magnitude > max_velocity:
            desired_vel = desired_vel * (max_velocity / vel_magnitude)
        
        # Get current velocity
        current_vel, _ = p.getBaseVelocity(self.robot_id, physicsClientId=self.physics_client)
        current_vel_2d = np.array([current_vel[0], current_vel[1]])
        
        # Calculate velocity error
        vel_error = desired_vel - current_vel_2d
        
        # Apply force
        force_gain = 300.0
        force_3d = [vel_error[0] * force_gain, vel_error[1] * force_gain, 0]
        
        # Orient toward movement direction
        if np.linalg.norm(desired_vel) > 0.1:
            desired_angle = np.arctan2(desired_vel[1], desired_vel[0])
            pos, current_quat = p.getBasePositionAndOrientation(self.robot_id, physicsClientId=self.physics_client)
            new_quat = p.getQuaternionFromEuler([0, 0, desired_angle], physicsClientId=self.physics_client)
            
            # Smooth rotation
            t = 0.1
            interp_quat = [(1-t) * current_quat[i] + t * new_quat[i] for i in range(4)]
            norm = np.sqrt(sum(q*q for q in interp_quat))
            interp_quat = [q/norm for q in interp_quat]
            
            p.resetBasePositionAndOrientation(self.robot_id, pos, interp_quat, physicsClientId=self.physics_client)
        
        p.applyExternalForce(self.robot_id, -1, force_3d, [0, 0, 0], p.WORLD_FRAME, physicsClientId=self.physics_client)
    
    def run_episode(self, goal_position, max_steps=5000):
        """
        Run navigation episode with BASELINE controller.
        No sensors, no LLM planning, no adaptive gains, no safety filter.
        Just simple PID control directly toward goal.
        """
        print(f"\n{'='*60}")
        print(f"BASELINE Controller: Goal at {goal_position}")
        print(f"No sensors, No LLM, No adaptive gains, No safety filter")
        print(f"{'='*60}\n")
        
        self.trajectory = []
        safety_violations = 0
        min_obstacle_distance = float('inf')
        
        # Get obstacle info from PyBullet for safety tracking
        obstacle_info = []
        num_bodies = p.getNumBodies(physicsClientId=self.physics_client)
        for i in range(num_bodies):
            if i != self.robot_id and i != 0:  # Skip robot and ground plane
                pos, _ = p.getBasePositionAndOrientation(i, physicsClientId=self.physics_client)
                # Assume obstacles are boxes with size 1.0
                obstacle_info.append({'position': np.array(pos[:2]), 'radius': 0.7})  # Approximate radius
        
        # Fixed PID gains (no adaptation)
        kp, ki, kd = 200.0, 5.0, 20.0
        pid_x = PIDController(kp, ki, kd, setpoint=goal_position[0])
        pid_y = PIDController(kp, ki, kd, setpoint=goal_position[1])
        
        goal_threshold = 0.5
        
        for step in range(max_steps):
            # Get position (basic proprioception only)
            pos = self.get_robot_position()
            self.trajectory.append(pos.copy())
            
            # Compute distance to goal
            dist_to_goal = np.linalg.norm(pos - np.array(goal_position[:2]))
            
            # Logging
            if step % 200 == 0:
                worker_prefix = f"[Worker {self.worker_id}] " if self.worker_id is not None else ""
                print(f"{worker_prefix}  Step {step:4d}: Robot=[{pos[0]:6.2f}, {pos[1]:6.2f}]  Goal=[{goal_position[0]:5.1f}, {goal_position[1]:5.1f}]  Dist={dist_to_goal:5.2f}m")
            
            # Check goal reached
            if dist_to_goal < goal_threshold:
                print(f"[SUCCESS] Goal reached in {step} steps!")
                break
            
            # Simple PID control (no LLM waypoints, just direct to goal)
            u_x = pid_x.compute(pos[0], self.timestep)
            u_y = pid_y.compute(pos[1], self.timestep)
            u_nom = np.array([u_x, u_y])
            
            # Track safety violations and minimum distance for fair comparison
            for obs in obstacle_info:
                obs_dist = np.linalg.norm(pos - obs['position']) - obs['radius']
                min_obstacle_distance = min(min_obstacle_distance, obs_dist)
                if obs_dist < 0.3:  # Collision or very close approach
                    safety_violations += 1
            
            # Apply control (no safety filter)
            self.apply_control(u_nom)
            
            # Step simulation
            p.stepSimulation(physicsClientId=self.physics_client)
            time.sleep(self.timestep)
        
        # Compute final statistics
        final_pos = self.get_robot_position()
        final_dist = np.linalg.norm(final_pos - np.array(goal_position[:2]))
        
        stats = {
            'success': final_dist < goal_threshold,
            'final_distance': final_dist,
            'steps': len(self.trajectory),
            'trajectory': np.array(self.trajectory),
            'path_length': self.compute_path_length(),
            'safety_violations': safety_violations,
            'min_obstacle_distance': min_obstacle_distance
        }
        
        print(f"\nBaseline Episode Complete:")
        print(f"  Success: {stats['success']}")
        print(f"  Final distance: {final_dist:.3f}m")
        print(f"  Steps: {stats['steps']}")
        print(f"  Path length: {stats['path_length']:.2f}m")
        
        return stats
    
    def compute_path_length(self):
        """Compute total path length traveled."""
        if len(self.trajectory) < 2:
            return 0.0
        
        total_length = 0.0
        for i in range(1, len(self.trajectory)):
            total_length += np.linalg.norm(self.trajectory[i] - self.trajectory[i-1])
        
        return total_length
