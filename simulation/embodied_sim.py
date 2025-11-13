"""
Enhanced PyBullet Simulation with Multimodal Sensors and LLM Integration
Implements complete embodied intelligence architecture.
"""

import pybullet as p
import pybullet_data
import time
import numpy as np
from datetime import datetime
from sensors import SensorSuite, StateEstimator
from llm_planner import LLMPlanner, AdaptiveControlGains
from control import PIDController, SafetyFilter, system_dynamics, obstacle_barrier_function
from visualization import VisualizationManager


class EmbodiedRobotSimulation:
    """
    Complete simulation environment for LLM-driven multimodal robot navigation.
    """
    
    def __init__(self, 
                 headless=False,
                 use_real_llm=False,
                 enable_sensors=True,
                 enable_adaptive_control=True,
                 enable_visualization=True,
                 worker_id=None):
        """
        Initialize enhanced simulation.
        
        Args:
            headless: Run without GUI
            use_real_llm: Use real LLM API (requires OPENAI_API_KEY)
            enable_sensors: Enable multimodal sensor suite
            enable_adaptive_control: Enable adaptive control gain scheduling
            enable_visualization: Enable visual feedback (LiDAR rays, status indicators)
            worker_id: Worker ID for logging (optional)
        """
        self.headless = headless
        self.enable_sensors = enable_sensors
        self.enable_adaptive_control = enable_adaptive_control
        self.enable_visualization = enable_visualization and not headless
        self.worker_id = worker_id  # For logging identification
        
        # Connect to PyBullet
        if self.headless:
            self.client = p.connect(p.DIRECT)
        else:
            self.client = p.connect(p.GUI)
            p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
            p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 1)
            # Disable shadows for faster rendering
            p.configureDebugVisualizer(p.COV_ENABLE_SHADOWS, 0)
        
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.81)
        
        # Enable parallel physics (uses multiple CPU cores)
        p.setPhysicsEngineParameter(numSolverIterations=5, physicsClientId=self.client)
        
        # Set better camera view for full workspace visibility
        if not headless:
            p.resetDebugVisualizerCamera(
                cameraDistance=12.0,      # Zoom out to see entire 10x10 workspace
                cameraYaw=45,             # Diagonal view
                cameraPitch=-35,          # Look down at angle
                cameraTargetPosition=[2.5, 2.5, 0]  # Center of workspace
            )
        
        # Simulation timestep
        self.timestep = 1.0 / 240.0
        p.setTimeStep(self.timestep)
        
        # Load environment - use default plane (revert solid color experiment)
        self.plane_id = p.loadURDF("plane.urdf")
        self.robot_id = self._create_robot()
        self.obstacle_ids = []
        
        # Initialize components
        if self.enable_sensors:
            self.sensor_suite = SensorSuite(self.robot_id, self.client, lightweight_mode=True)
            # Initialize state estimator with 6D state: [x, y, z, vx, vy, vz]
            initial_state = np.zeros(6)
            initial_cov = np.eye(6) * 0.1
            self.state_estimator = StateEstimator(initial_state, initial_cov)
            print("[OK] Multimodal sensor suite initialized (lightweight mode)")
        else:
            self.sensor_suite = None
            self.state_estimator = None
        
        # Initialize LLM planner
        self.llm_planner = LLMPlanner(use_mock=not use_real_llm)
        
        # Initialize adaptive control
        if self.enable_adaptive_control:
            self.adaptive_gains = AdaptiveControlGains(
                initial_kp=200.0,  # Start with same gains as baseline for fair speed comparison
                initial_ki=5.0,
                initial_kd=20.0
            )
            print("[OK] Adaptive control initialized")
        else:
            self.adaptive_gains = None
        
        # Safety filter
        self.safety_filter = SafetyFilter(dim=2, alpha=0.5)
        
        # Initialize visualization manager
        if self.enable_visualization:
            try:
                self.viz_manager = VisualizationManager(self.robot_id, self.client)
                print("[OK] Visualization manager initialized")
            except Exception as e:
                print(f"[WARNING] Failed to initialize visualization: {e}")
                self.viz_manager = None
                self.enable_visualization = False
        else:
            self.viz_manager = None
        
        # Logging
        self.trajectory = []
        self.safety_violations = 0
        self.llm_interventions = 0
        self.sensor_data_log = []
        self.simulation_time = 0.0
        
    def _create_robot(self):
        """Create robot with sensors attached - looks like a mobile robot platform."""
        start_pos = [0, 0, 0.08]  # Start at wheel radius height (wheels touch ground)
        start_orientation = p.getQuaternionFromEuler([0, 0, 0])
        
        # Create realistic robot URDF with base platform, sensor tower, and wheels
        urdf_path = "robot_with_sensors.urdf"
        with open(urdf_path, "w") as f:
            f.write("""
<robot name="multimodal_robot">
  <!-- Main chassis/base platform -->
  <link name="base_link">
    <visual>
      <origin xyz="0 0 0.05" rpy="0 0 0"/>
      <geometry>
        <box size="0.4 0.3 0.1"/>
      </geometry>
      <material name="robot_blue">
        <color rgba="0.1 0.4 0.9 1.0"/>
      </material>
    </visual>
    <collision>
      <origin xyz="0 0 0.05" rpy="0 0 0"/>
      <geometry>
        <box size="0.4 0.3 0.1"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="5.0"/>
      <inertia ixx="0.1" ixy="0.0" ixz="0.0" iyy="0.15" iyz="0.0" izz="0.2"/>
    </inertial>
  </link>
  
  <!-- Sensor tower on top -->
  <link name="sensor_tower">
    <visual>
      <origin xyz="0 0 0.15" rpy="0 0 0"/>
      <geometry>
        <cylinder radius="0.08" length="0.2"/>
      </geometry>
      <material name="sensor_gray">
        <color rgba="0.3 0.3 0.3 1.0"/>
      </material>
    </visual>
  </link>
  
  <joint name="tower_joint" type="fixed">
    <parent link="base_link"/>
    <child link="sensor_tower"/>
    <origin xyz="0 0 0.1" rpy="0 0 0"/>
  </joint>
  
  <!-- LiDAR/Camera head -->
  <link name="lidar_head">
    <visual>
      <origin xyz="0 0 0" rpy="0 0 0"/>
      <geometry>
        <cylinder radius="0.1" length="0.05"/>
      </geometry>
      <material name="lidar_black">
        <color rgba="0.1 0.1 0.1 1.0"/>
      </material>
    </visual>
  </link>
  
  <joint name="lidar_joint" type="fixed">
    <parent link="sensor_tower"/>
    <child link="lidar_head"/>
    <origin xyz="0 0 0.2" rpy="0 0 0"/>
  </joint>
  
  <!-- Front left wheel -->
  <link name="wheel_fl">
    <visual>
      <origin xyz="0 0 0" rpy="1.5708 0 0"/>
      <geometry>
        <cylinder radius="0.08" length="0.05"/>
      </geometry>
      <material name="wheel_black">
        <color rgba="0.1 0.1 0.1 1.0"/>
      </material>
    </visual>
  </link>
  
  <joint name="wheel_fl_joint" type="fixed">
    <parent link="base_link"/>
    <child link="wheel_fl"/>
    <origin xyz="0.15 0.18 0.08" rpy="0 0 0"/>
  </joint>
  
  <!-- Front right wheel -->
  <link name="wheel_fr">
    <visual>
      <origin xyz="0 0 0" rpy="1.5708 0 0"/>
      <geometry>
        <cylinder radius="0.08" length="0.05"/>
      </geometry>
      <material name="wheel_black">
        <color rgba="0.1 0.1 0.1 1.0"/>
      </material>
    </visual>
  </link>
  
  <joint name="wheel_fr_joint" type="fixed">
    <parent link="base_link"/>
    <child link="wheel_fr"/>
    <origin xyz="0.15 -0.18 0.08" rpy="0 0 0"/>
  </joint>
  
  <!-- Rear left wheel -->
  <link name="wheel_rl">
    <visual>
      <origin xyz="0 0 0" rpy="1.5708 0 0"/>
      <geometry>
        <cylinder radius="0.08" length="0.05"/>
      </geometry>
      <material name="wheel_black">
        <color rgba="0.1 0.1 0.1 1.0"/>
      </material>
    </visual>
  </link>
  
  <joint name="wheel_rl_joint" type="fixed">
    <parent link="base_link"/>
    <child link="wheel_rl"/>
    <origin xyz="-0.15 0.18 0.08" rpy="0 0 0"/>
  </joint>
  
  <!-- Rear right wheel -->
  <link name="wheel_rr">
    <visual>
      <origin xyz="0 0 0" rpy="1.5708 0 0"/>
      <geometry>
        <cylinder radius="0.08" length="0.05"/>
      </geometry>
      <material name="wheel_black">
        <color rgba="0.1 0.1 0.1 1.0"/>
      </material>
    </visual>
  </link>
  
  <joint name="wheel_rr_joint" type="fixed">
    <parent link="base_link"/>
    <child link="wheel_rr"/>
    <origin xyz="-0.15 -0.18 0.08" rpy="0 0 0"/>
  </joint>
</robot>
            """)
        
        robot_id = p.loadURDF(urdf_path, start_pos, start_orientation)
        
        return robot_id
    
    def add_obstacle(self, position, radius=0.5):
        """Add spherical obstacle to environment."""
        visual_shape = p.createVisualShape(
            shapeType=p.GEOM_SPHERE,
            radius=radius,
            rgbaColor=[0.8, 0.1, 0.1, 1.0]
        )
        collision_shape = p.createCollisionShape(
            shapeType=p.GEOM_SPHERE,
            radius=radius
        )
        
        obstacle_id = p.createMultiBody(
            baseMass=0,  # Static obstacle
            baseCollisionShapeIndex=collision_shape,
            baseVisualShapeIndex=visual_shape,
            basePosition=position
        )
        
        self.obstacle_ids.append({
            'id': obstacle_id,
            'position': position,
            'radius': radius
        })
        
        return obstacle_id
    
    def add_goal_marker(self, position, radius=0.3):
        """Add visual goal marker."""
        visual_shape = p.createVisualShape(
            shapeType=p.GEOM_SPHERE,
            radius=radius,
            rgbaColor=[0.1, 0.8, 0.1, 0.5]
        )
        
        goal_id = p.createMultiBody(
            baseMass=0,
            baseVisualShapeIndex=visual_shape,
            basePosition=position
        )
        
        return goal_id
    
    def get_robot_state(self, use_sensors=None):
        """
        Get robot state - either from sensors or ground truth.
        
        Args:
            use_sensors: Override to force sensor/ground truth
        
        Returns:
            state dict with position, velocity, etc.
        """
        if use_sensors is None:
            use_sensors = self.enable_sensors
        
        if use_sensors and self.sensor_suite is not None:
            # Get measurements from all sensors
            measurements = self.sensor_suite.get_measurements(
                add_noise=True,
                dropout_prob=0.05
            )
            
            # Update state estimator
            self.state_estimator.predict(self.timestep)
            self.state_estimator.update(measurements)
            
            # Get fused state estimate
            state_vec = self.state_estimator.get_state()
            
            # Log sensor data
            self.sensor_data_log.append({
                'time': p.getDebugVisualizerCamera()[3],
                'measurements': {
                    k: v is not None for k, v in measurements.items() if k != 'timestamp'
                },
                'state_estimate': state_vec.copy()
            })
            
            return {
                'position': state_vec[:3],
                'velocity': state_vec[3:6],
                'measurements': measurements
            }
        else:
            # Ground truth from PyBullet
            pos, orn = p.getBasePositionAndOrientation(self.robot_id)
            lin_vel, ang_vel = p.getBaseVelocity(self.robot_id)
            
            return {
                'position': np.array(pos),
                'orientation': np.array(orn),
                'velocity': np.array(lin_vel),
                'angular_velocity': np.array(ang_vel)
            }
    
    def apply_control(self, force):
        """Apply 2D control force to robot and orient it toward movement direction."""
        # Use velocity control instead of pure force to prevent jerky motion
        # PID output represents desired velocity (m/s), not force
        max_velocity = 5.0  # Maximum velocity in m/s (fast movement)
        desired_vel = np.array([force[0], force[1]]) * 2.0  # Much higher scaling: PID output to velocity
        
        # Clamp to maximum velocity
        vel_magnitude = np.linalg.norm(desired_vel)
        if vel_magnitude > max_velocity:
            desired_vel = desired_vel * (max_velocity / vel_magnitude)
        
        # Get current velocity
        current_vel, _ = p.getBaseVelocity(self.robot_id)
        current_vel_2d = np.array([current_vel[0], current_vel[1]])
        
        # Calculate velocity error
        vel_error = desired_vel - current_vel_2d
        
        # Apply proportional force based on velocity error (acts like damping)
        force_gain = 300.0  # Very high N/(m/s) - force per unit velocity error for fast response
        force_3d = [vel_error[0] * force_gain, vel_error[1] * force_gain, 0]
        
        # Calculate desired heading angle from desired velocity direction
        if np.linalg.norm(desired_vel) > 0.1:  # Only rotate if moving significantly
            desired_angle = np.arctan2(desired_vel[1], desired_vel[0])
            
            # Directly set orientation (holonomic robot - can face any direction while moving)
            pos, _ = p.getBasePositionAndOrientation(self.robot_id)
            new_quat = p.getQuaternionFromEuler([0, 0, desired_angle])
            
            # Smoothly interpolate rotation
            _, current_quat = p.getBasePositionAndOrientation(self.robot_id)
            vel, ang_vel = p.getBaseVelocity(self.robot_id)
            
            # Use slerp for smooth rotation (0.1 = interpolation factor)
            t = 0.1
            interp_quat = [
                (1-t) * current_quat[i] + t * new_quat[i] 
                for i in range(4)
            ]
            # Normalize quaternion
            norm = np.sqrt(sum(q*q for q in interp_quat))
            interp_quat = [q/norm for q in interp_quat]
            
            p.resetBasePositionAndOrientation(self.robot_id, pos, interp_quat)
        
        # Apply translational force
        p.applyExternalForce(
            self.robot_id,
            -1,  # Base link
            force_3d,
            [0, 0, 0],
            p.WORLD_FRAME
        )
    
    def check_safety_violation(self):
        """Check if robot is in collision with obstacles."""
        for obs in self.obstacle_ids:
            obs_id = obs['id']
            contacts = p.getContactPoints(bodyA=self.robot_id, bodyB=obs_id)
            if len(contacts) > 0:
                return True
        return False
    
    def run_episode(self,
                   goal_position,
                   max_steps=1000,
                   use_safety_filter=True,
                   sensor_noise=True,
                   workspace_bounds=(-5, 5, -5, 5),
                   record_video=False,
                   video_filename=None):
        """
        Run complete navigation episode with LLM planning.
        
        Args:
            goal_position: Target position [x, y, z]
            max_steps: Maximum simulation steps
            use_safety_filter: Enable CBF safety filter
            sensor_noise: Add sensor noise
            workspace_bounds: (x_min, x_max, y_min, y_max)
            record_video: Whether to record video of simulation
            video_filename: Output video filename (auto-generated with timestamp if None)
        
        Returns:
            dict: Episode statistics and trajectory
        """
        # Auto-generate filename with timestamp if not provided
        if record_video and video_filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            video_filename = f"results/simulation_{timestamp}.mp4"
        
        # Initialize log buffer for metadata
        log_buffer = []
        
        def log_and_print(msg):
            """Print to console and store in buffer"""
            print(msg)
            log_buffer.append(msg)
        
        log_and_print(f"\n{'='*60}")
        log_and_print(f"Starting Episode: Goal at {goal_position}")
        log_and_print(f"Safety Filter: {'ENABLED' if use_safety_filter else 'DISABLED'}")
        log_and_print(f"Sensors: {'ENABLED' if self.enable_sensors else 'DISABLED'}")
        log_and_print(f"Adaptive Control: {'ENABLED' if self.enable_adaptive_control else 'DISABLED'}")
        if record_video:
            log_and_print(f"Recording: {video_filename}")
        log_and_print(f"{'='*60}\n")
        
        # Start video recording if requested
        if record_video and not self.headless:
            log_id = p.startStateLogging(p.STATE_LOGGING_VIDEO_MP4, video_filename)
            log_and_print(f"[RECORDING] Started video logging to {video_filename}")
        else:
            log_id = None
        
        # Reset tracking
        self.trajectory = []
        self.safety_violations = 0
        self.llm_interventions = 0
        
        # Track adaptive gains evolution and sensor data for paper analysis
        self.adaptive_gains_history = []
        self.sensor_data_samples = []
        self.min_obstacle_distance = float('inf')
        
        # Get initial state
        current_state = self.get_robot_state()
        
        # Plan with LLM
        waypoints, confidence = self.llm_planner.plan(
            current_state=current_state,
            goal_state={'position': goal_position},
            obstacles=self.obstacle_ids,
            workspace_bounds=workspace_bounds
        )
        
        # Apply confidence gating
        waypoints, llm_accepted = self.llm_planner.apply_confidence_gate(
            waypoints, confidence, current_state, {'position': goal_position}
        )
        
        if not llm_accepted:
            self.llm_interventions += 1
        
        # Initialize controllers for each waypoint
        current_waypoint_idx = 0
        waypoint_threshold = 1.0  # Increased: less precise following allows faster navigation
        waypoint_timeout = 2000  # Reduced: skip unreachable waypoints faster for rerouting
        steps_at_current_waypoint = 0
        last_waypoint_distance = float('inf')
        
        # Create PID controller (higher gains for heavier robot model)
        if self.enable_adaptive_control and self.adaptive_gains is not None:
            kp, ki, kd = self.adaptive_gains.get_gains()
        else:
            kp, ki, kd = 200.0, 5.0, 20.0  # Much higher gains for faster movement
        
        pid_x = PIDController(kp, ki, kd, setpoint=waypoints[current_waypoint_idx][0])
        pid_y = PIDController(kp, ki, kd, setpoint=waypoints[current_waypoint_idx][1])
        
        # Run episode
        debug_text_id = None
        goal_reached = False  # Track if goal was ever reached
        for step in range(max_steps):
            # Get current state
            state = self.get_robot_state()
            pos = state['position'][:2]  # 2D position
            vel = state['velocity'][:2]
            
            # Log trajectory
            self.trajectory.append(pos.copy())
            
            # Compute distance to goal (needed for logging and termination)
            dist_to_goal = np.linalg.norm(pos - np.array(goal_position[:2]))
            
            # Debug visualization
            if not self.headless and step % 50 == 0:
                if debug_text_id is not None:
                    p.removeUserDebugItem(debug_text_id)
                debug_text = f"Robot: [{pos[0]:.2f}, {pos[1]:.2f}]\nGoal: [{goal_position[0]:.1f}, {goal_position[1]:.1f}]\nWaypoint {current_waypoint_idx+1}/{len(waypoints)}"
                debug_text_id = p.addUserDebugText(debug_text, [pos[0], pos[1], 1.5], textColorRGB=[0,0,0], textSize=1.2)
            
            # Terminal logging every 200 steps
            if step % 200 == 0:
                worker_prefix = f"[Worker {self.worker_id}] " if self.worker_id is not None else ""
                log_and_print(f"{worker_prefix}  Step {step:4d}: Robot=[{pos[0]:6.2f}, {pos[1]:6.2f}]  Goal=[{goal_position[0]:5.1f}, {goal_position[1]:5.1f}]  Dist={dist_to_goal:5.2f}m  WP={current_waypoint_idx+1}/{len(waypoints)}")
            
            # Check if current waypoint reached
            if current_waypoint_idx < len(waypoints):
                target = waypoints[current_waypoint_idx]
                dist_to_waypoint = np.linalg.norm(pos - target)
                
                # Track progress toward waypoint
                steps_at_current_waypoint += 1
                
                # Check if waypoint reached
                if dist_to_waypoint < waypoint_threshold:
                    log_and_print(f"  Waypoint {current_waypoint_idx + 1}/{len(waypoints)} reached")
                    current_waypoint_idx += 1
                    steps_at_current_waypoint = 0  # Reset timeout counter
                    
                    # Update PID setpoints if more waypoints remain
                    if current_waypoint_idx < len(waypoints):
                        pid_x.setpoint = waypoints[current_waypoint_idx][0]
                        pid_y.setpoint = waypoints[current_waypoint_idx][1]
                        log_and_print(f"  --> Targeting next waypoint: {waypoints[current_waypoint_idx]}")
                
                # Check if stuck at waypoint (timeout mechanism)
                elif steps_at_current_waypoint >= waypoint_timeout:
                    # Check if making progress (distance decreasing)
                    if dist_to_waypoint < last_waypoint_distance - 0.2:  # Increased threshold for progress detection
                        # Still making progress, reset timeout
                        last_waypoint_distance = dist_to_waypoint
                        steps_at_current_waypoint = 0
                    else:
                        # Stuck! Skip to next waypoint or goal
                        log_and_print(f"  [TIMEOUT] Waypoint {current_waypoint_idx + 1}/{len(waypoints)} unreachable after {waypoint_timeout} steps")
                        log_and_print(f"  --> Skipping to next waypoint (distance was {dist_to_waypoint:.2f}m)")
                        current_waypoint_idx += 1
                        steps_at_current_waypoint = 0
                        
                        # Update PID setpoints
                        if current_waypoint_idx < len(waypoints):
                            pid_x.setpoint = waypoints[current_waypoint_idx][0]
                            pid_y.setpoint = waypoints[current_waypoint_idx][1]
                            log_and_print(f"  --> Now targeting waypoint {current_waypoint_idx + 1}/{len(waypoints)}: {waypoints[current_waypoint_idx]}")
                        else:
                            log_and_print(f"  --> All waypoints exhausted, will timeout at max_steps")
                
                # Update last distance for progress tracking
                last_waypoint_distance = min(last_waypoint_distance, dist_to_waypoint)
            
            # Check goal reached (dist_to_goal already computed earlier)
            if dist_to_goal < 0.5:  # 0.5m tolerance for goal reaching
                log_and_print(f"[SUCCESS] Goal reached in {step} steps!")
                goal_reached = True
                break
            
            # Compute nominal control
            u_x = pid_x.compute(pos[0], self.timestep)
            u_y = pid_y.compute(pos[1], self.timestep)
            u_nom = np.array([u_x, u_y])
            
            # Adapt control gains if enabled
            if self.enable_adaptive_control and self.adaptive_gains is not None:
                error = dist_to_goal
                error_derivative = -np.dot(vel, (goal_position[:2] - pos)) / (dist_to_goal + 1e-6)
                self.adaptive_gains.adapt(error, error_derivative, self.timestep)
                kp, ki, kd = self.adaptive_gains.get_gains()
                pid_x.kp, pid_y.kp = kp, kp
                pid_x.kd, pid_y.kd = kd, kd
            
            # Apply safety filter if enabled
            if use_safety_filter:
                # Compute barrier function for each obstacle
                u_safe = u_nom
                for obs in self.obstacle_ids:
                    # Full state: [x, y, z, vx, vy, vz]
                    full_state = np.hstack([state['position'], state['velocity']])
                    
                    h, Lfh, Lgh = obstacle_barrier_function(
                        full_state,
                        obs['position'][:2],
                        obs['radius'] + 0.3  # Safety margin
                    )
                    
                    # Only apply filter if close to obstacle
                    if h < 1.0:
                        u_safe = self.safety_filter.solve(u_safe, h, Lfh, Lgh)
                
                u_final = u_safe
            else:
                u_final = u_nom
            
            # Apply control
            self.apply_control(u_final)
            
            # Update visualizations if enabled
            if self.enable_visualization and self.viz_manager is not None and step % 10 == 0:  # Update every 10 steps instead of 5
                # Get sensor data for visualization
                if self.enable_sensors and self.sensor_suite is not None:
                    try:
                        sensor_reading = self.sensor_suite.get_measurements(add_noise=sensor_noise)
                        
                        # Draw LiDAR rays
                        if sensor_reading.get('lidar') is not None:
                            self.viz_manager.draw_lidar_rays(
                                sensor_reading['lidar'],
                                obstacle_threshold=3.0,
                                subsample=8  # Draw every 8th ray instead of 6 to reduce clutter
                            )
                    except Exception as e:
                        pass  # Silently skip visualization errors
                
                # Draw safety status indicator
                safety_active = not np.allclose(u_final, u_nom, atol=0.1)
                self.viz_manager.draw_safety_status(
                    safety_active=safety_active,
                    min_obstacle_distance=self.min_obstacle_distance,
                    position=pos.tolist() + [state['position'][2]]
                )
                
                # Draw LLM decision overlay
                if current_waypoint_idx < len(waypoints):
                    current_waypoint = waypoints[current_waypoint_idx]
                    self.viz_manager.draw_llm_decision(
                        waypoint=current_waypoint,
                        confidence=confidence,
                        planning_mode="WAYPOINT_FOLLOWING",
                        position=pos.tolist() + [state['position'][2]]
                    )
                
                # Update gains history for graphing
                if self.enable_adaptive_control and self.adaptive_gains is not None:
                    kp_vis, ki_vis, kd_vis = self.adaptive_gains.get_gains()
                    self.viz_manager.update_gains_history(
                        kp_vis, ki_vis, kd_vis, 
                        step * self.timestep
                    )
            
            # Collect data for paper analysis (every 100 steps to avoid huge files)
            if step % 100 == 0:
                # Track adaptive gains evolution
                if self.enable_adaptive_control and self.adaptive_gains is not None:
                    kp, ki, kd = self.adaptive_gains.get_gains()
                    self.adaptive_gains_history.append({
                        'step': step,
                        'kp': float(kp),
                        'ki': float(ki),
                        'kd': float(kd)
                    })
                
                # Sample sensor data
                if self.enable_sensors and self.sensor_suite is not None:
                    try:
                        sensor_reading = self.sensor_suite.get_measurements(add_noise=False)
                        # Store lightweight summary (not full arrays)
                        if sensor_reading.get('lidar') is not None:
                            lidar_ranges = sensor_reading['lidar']['ranges']
                            self.sensor_data_samples.append({
                                'step': step,
                                'lidar_min': float(np.min(lidar_ranges)) if len(lidar_ranges) > 0 else float('inf'),
                                'lidar_mean': float(np.mean(lidar_ranges)) if len(lidar_ranges) > 0 else float('inf')
                            })
                    except Exception as e:
                        # Skip sensor data collection if error
                        pass
            
            # Track minimum obstacle distance for safety analysis
            for obs in self.obstacle_ids:
                obs_dist = np.linalg.norm(pos - obs['position'][:2]) - obs['radius']
                self.min_obstacle_distance = min(self.min_obstacle_distance, obs_dist)
            
            # Visualize control force
            if not self.headless and step % 10 == 0:
                # Draw arrow showing control direction
                arrow_start = [pos[0], pos[1], 0.5]
                arrow_scale = 0.5
                arrow_end = [pos[0] + u_final[0] * arrow_scale, 
                            pos[1] + u_final[1] * arrow_scale, 
                            0.5]
                p.addUserDebugLine(arrow_start, arrow_end, [0, 1, 0], lineWidth=3, lifeTime=0.5)
            
            # Step simulation
            p.stepSimulation()
            
            # Check safety violation
            if self.check_safety_violation():
                self.safety_violations += 1
            
            # Visualize
            if not self.headless:
                time.sleep(self.timestep)
        
        # Compute statistics
        final_state = self.get_robot_state()
        final_dist = np.linalg.norm(final_state['position'][:2] - np.array(goal_position[:2]))
        
        # Stop video recording if active
        if log_id is not None:
            p.stopStateLogging(log_id)
            log_and_print(f"[RECORDING] Stopped video logging")
            
            # Save metadata log for this recording
            if video_filename:
                metadata_file = video_filename.replace('.mp4', '_metadata.txt')
                with open(metadata_file, 'w') as f:
                    f.write(f"Simulation Recording - Complete Log\n")
                    f.write(f"="*60 + "\n")
                    f.write(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                    f.write(f"Video File: {video_filename}\n\n")
                    f.write(f"Configuration:\n")
                    f.write(f"  Goal Position: {goal_position}\n")
                    f.write(f"  Max Steps: {max_steps}\n")
                    f.write(f"  Safety Filter: {'ENABLED' if use_safety_filter else 'DISABLED'}\n")
                    f.write(f"  Sensor Noise: {'ENABLED' if sensor_noise else 'DISABLED'}\n")
                    f.write(f"  Sensors: {'ENABLED' if self.enable_sensors else 'DISABLED'}\n")
                    f.write(f"  Adaptive Control: {'ENABLED' if self.enable_adaptive_control else 'DISABLED'}\n")
                    f.write(f"  Workspace Bounds: {workspace_bounds}\n\n")
                    f.write(f"Results:\n")
                    f.write(f"  Success: {final_dist < 0.5}\n")
                    f.write(f"  Final Distance: {final_dist:.3f}m\n")
                    f.write(f"  Steps Taken: {len(self.trajectory)}\n")
                    f.write(f"  Safety Violations: {self.safety_violations}\n")
                    f.write(f"  LLM Confidence: {confidence:.2f}\n")
                    f.write(f"  Waypoints: {waypoints}\n\n")
                    f.write(f"="*60 + "\n")
                    f.write(f"COMPLETE EXECUTION LOG:\n")
                    f.write(f"="*60 + "\n")
                    for log_line in log_buffer:
                        f.write(log_line + "\n")
                log_and_print(f"[RECORDING] Saved complete log to {metadata_file}")
        
        # Compute path length
        path_length = 0.0
        if len(self.trajectory) >= 2:
            for i in range(1, len(self.trajectory)):
                path_length += np.linalg.norm(
                    np.array(self.trajectory[i]) - np.array(self.trajectory[i-1])
                )
        
        stats = {
            'success': goal_reached,  # Use tracked goal_reached instead of final distance
            'final_distance': final_dist,
            'steps': len(self.trajectory),
            'path_length': path_length,
            'safety_violations': self.safety_violations,
            'min_obstacle_distance': self.min_obstacle_distance,
            'trajectory': np.array(self.trajectory),
            'llm_interventions': self.llm_interventions,
            'waypoints': waypoints,
            'llm_confidence': confidence,
            'adaptive_gains_history': self.adaptive_gains_history,
            'sensor_data_samples': self.sensor_data_samples
        }
        
        log_and_print(f"\nEpisode Complete:")
        log_and_print(f"  Success: {stats['success']}")
        log_and_print(f"  Final distance: {final_dist:.3f}m")
        log_and_print(f"  Steps: {stats['steps']}")
        log_and_print(f"  Safety violations: {stats['safety_violations']}")
        log_and_print(f"  LLM confidence: {confidence:.2f}")
        
        # Clear visualization elements
        if self.enable_visualization and self.viz_manager is not None:
            self.viz_manager.clear_all()
        
        return stats
    
    def close(self):
        """Clean up simulation."""
        p.disconnect()


if __name__ == "__main__":
    # Example usage
    sim = EmbodiedRobotSimulation(
        headless=False,
        use_real_llm=False,  # Set True if you have OPENAI_API_KEY
        enable_sensors=True,
        enable_adaptive_control=True
    )
    
    try:
        # Add obstacle
        sim.add_obstacle(position=[2, 0, 0.5], radius=0.5)
        
        # Add goal marker
        goal_pos = [4, 0, 0.5]
        sim.add_goal_marker(goal_pos)
        
        # Run episode with safety filter
        stats = sim.run_episode(
            goal_position=goal_pos,
            max_steps=2000,
            use_safety_filter=True
        )
        
        print(f"\n{'='*60}")
        print("Episode Statistics:")
        for key, value in stats.items():
            if key not in ['trajectory', 'waypoints']:
                print(f"  {key}: {value}")
        print(f"{'='*60}")
        
    finally:
        time.sleep(2)
        sim.close()
