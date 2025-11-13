"""
Multimodal Sensor Suite for Embodied Robotics
Implements RGB-D, LiDAR, tactile, and proprioceptive sensors with realistic noise models.
"""

import numpy as np
import pybullet as p
from scipy.stats import multivariate_normal


class SensorSuite:
    """Manages all sensor modalities for the robot."""
    
    def __init__(self, robot_id, sim_client, lightweight_mode=True):
        self.robot_id = robot_id
        self.sim_client = sim_client
        self.lightweight_mode = lightweight_mode
        
        # Initialize individual sensors
        if not lightweight_mode:
            self.camera = RGBDCamera(robot_id, sim_client)
        else:
            self.camera = None  # Skip expensive camera rendering in lightweight mode
        self.lidar = LiDARSensor(robot_id, sim_client, num_rays=72 if lightweight_mode else 360)
        self.tactile = TactileSensor(robot_id, sim_client)
        self.proprioception = ProprioceptiveSensor(robot_id, sim_client)
        
    def get_measurements(self, add_noise=True, dropout_prob=0.0):
        """
        Collect measurements from all sensor modalities.
        
        Args:
            add_noise: Whether to add realistic sensor noise
            dropout_prob: Probability of sensor dropout (0.0 to 1.0)
            
        Returns:
            dict: Measurements from all sensors with timestamps
        """
        import time
        measurements = {
            'timestamp': time.time(),
            'camera': None,
            'lidar': None,
            'tactile': None,
            'proprioception': None
        }
        
        # RGB-D Camera (skip in lightweight mode)
        if self.camera is not None and np.random.random() > dropout_prob:
            measurements['camera'] = self.camera.capture(add_noise=add_noise)
        
        # LiDAR
        if np.random.random() > dropout_prob:
            measurements['lidar'] = self.lidar.scan(add_noise=add_noise)
        
        # Tactile sensors
        if np.random.random() > dropout_prob:
            measurements['tactile'] = self.tactile.read(add_noise=add_noise)
        
        # Proprioception (motor encoders, IMU)
        if np.random.random() > dropout_prob:
            measurements['proprioception'] = self.proprioception.read(add_noise=add_noise)
        
        return measurements


class RGBDCamera:
    """Simulated RGB-D camera sensor."""
    
    def __init__(self, robot_id, sim_client, width=64, height=64):  # Reduced resolution for speed
        self.robot_id = robot_id
        self.sim_client = sim_client
        self.width = width
        self.height = height
        
        # Camera intrinsics
        self.fov = 60
        self.near = 0.1
        self.far = 10.0
        
        # Noise parameters
        self.rgb_noise_std = 5.0  # RGB noise in [0, 255]
        self.depth_noise_std = 0.01  # Depth noise in meters
        
        # Cache for performance
        self._last_capture_time = 0
        self._capture_cache = None
        self._cache_duration = 0.1  # Cache for 100ms
        
    def capture(self, add_noise=True):
        """
        Capture RGB-D image from robot's perspective.
        
        Returns:
            dict: RGB image, depth map, and camera pose
        """
        # Get robot pose
        pos, orn = p.getBasePositionAndOrientation(self.robot_id)
        rot_matrix = p.getMatrixFromQuaternion(orn)
        rot_matrix = np.array(rot_matrix).reshape(3, 3)
        
        # Camera points forward in robot frame
        camera_forward = rot_matrix @ np.array([1, 0, 0])
        camera_up = rot_matrix @ np.array([0, 0, 1])
        target = np.array(pos) + camera_forward * 2.0
        
        # Compute view and projection matrices
        view_matrix = p.computeViewMatrix(
            cameraEyePosition=pos,
            cameraTargetPosition=target.tolist(),
            cameraUpVector=camera_up.tolist()
        )
        
        proj_matrix = p.computeProjectionMatrixFOV(
            fov=self.fov,
            aspect=self.width / self.height,
            nearVal=self.near,
            farVal=self.far
        )
        
        # Render image
        _, _, rgb, depth, _ = p.getCameraImage(
            width=self.width,
            height=self.height,
            viewMatrix=view_matrix,
            projectionMatrix=proj_matrix,
            renderer=p.ER_BULLET_HARDWARE_OPENGL
        )
        
        # Convert to numpy arrays
        rgb_array = np.array(rgb, dtype=np.uint8).reshape(self.height, self.width, 4)[:, :, :3]
        depth_array = np.array(depth, dtype=np.float32).reshape(self.height, self.width)
        
        # Convert depth buffer to actual depth in meters
        depth_array = self.far * self.near / (self.far - (self.far - self.near) * depth_array)
        
        # Add noise
        if add_noise:
            rgb_noise = np.random.normal(0, self.rgb_noise_std, rgb_array.shape)
            rgb_array = np.clip(rgb_array + rgb_noise, 0, 255).astype(np.uint8)
            
            depth_noise = np.random.normal(0, self.depth_noise_std, depth_array.shape)
            depth_array = np.clip(depth_array + depth_noise, self.near, self.far)
        
        return {
            'rgb': rgb_array,
            'depth': depth_array,
            'pose': {'position': pos, 'orientation': orn},
            'intrinsics': {
                'fov': self.fov,
                'width': self.width,
                'height': self.height
            }
        }


class LiDARSensor:
    """Simulated 2D LiDAR sensor."""
    
    def __init__(self, robot_id, sim_client, num_rays=360, max_range=10.0):
        self.robot_id = robot_id
        self.sim_client = sim_client
        self.num_rays = num_rays
        self.max_range = max_range
        
        # Noise parameters
        self.range_noise_std = 0.02  # 2cm range noise
        self.dropout_per_ray = 0.01  # 1% ray dropout
        
    def scan(self, add_noise=True):
        """
        Perform 360-degree LiDAR scan.
        
        Returns:
            dict: Range measurements and angles
        """
        # Get robot pose
        pos, orn = p.getBasePositionAndOrientation(self.robot_id)
        rot_matrix = p.getMatrixFromQuaternion(orn)
        rot_matrix = np.array(rot_matrix).reshape(3, 3)
        
        # LiDAR is at robot height
        lidar_height = pos[2]
        
        ranges = []
        angles = np.linspace(0, 2 * np.pi, self.num_rays, endpoint=False)
        
        for angle in angles:
            # Ray direction in world frame
            local_dir = np.array([np.cos(angle), np.sin(angle), 0])
            world_dir = rot_matrix @ local_dir
            
            ray_from = pos
            ray_to = np.array(pos) + world_dir * self.max_range
            
            # Cast ray
            result = p.rayTest(ray_from, ray_to.tolist())
            
            if result[0][0] == -1:  # No hit
                measured_range = self.max_range
            else:
                hit_fraction = result[0][2]
                measured_range = hit_fraction * self.max_range
            
            # Add noise and dropout
            if add_noise:
                if np.random.random() < self.dropout_per_ray:
                    measured_range = self.max_range  # Invalid reading
                else:
                    noise = np.random.normal(0, self.range_noise_std)
                    measured_range = np.clip(measured_range + noise, 0, self.max_range)
            
            ranges.append(measured_range)
        
        return {
            'ranges': np.array(ranges),
            'angles': angles,
            'max_range': self.max_range,
            'pose': {'position': pos, 'orientation': orn}
        }


class TactileSensor:
    """Simulated tactile/contact sensors."""
    
    def __init__(self, robot_id, sim_client):
        self.robot_id = robot_id
        self.sim_client = sim_client
        
        # Noise parameters
        self.force_noise_std = 0.1  # Force noise in Newtons
        
    def read(self, add_noise=True):
        """
        Read contact forces on robot.
        
        Returns:
            dict: Contact information
        """
        # Get contact points
        contact_points = p.getContactPoints(bodyA=self.robot_id)
        
        total_force = np.zeros(3)
        contacts = []
        
        for contact in contact_points:
            normal_force = contact[9]  # Normal force magnitude
            contact_normal = np.array(contact[7])  # Contact normal
            
            force_vector = normal_force * contact_normal
            
            if add_noise:
                noise = np.random.normal(0, self.force_noise_std, 3)
                force_vector += noise
            
            total_force += force_vector
            
            contacts.append({
                'position': contact[6],  # Position on robot
                'normal': contact_normal,
                'force': force_vector
            })
        
        return {
            'total_force': total_force,
            'num_contacts': len(contacts),
            'contacts': contacts,
            'in_contact': len(contacts) > 0
        }


class ProprioceptiveSensor:
    """Simulated proprioceptive sensors (encoders, IMU, etc.)."""
    
    def __init__(self, robot_id, sim_client):
        self.robot_id = robot_id
        self.sim_client = sim_client
        
        # Noise parameters
        self.position_noise_std = 0.001  # 1mm position noise
        self.velocity_noise_std = 0.01   # Velocity noise
        self.orientation_noise_std = 0.001  # Orientation noise (radians)
        
    def read(self, add_noise=True):
        """
        Read robot's internal state (position, velocity, orientation).
        
        Returns:
            dict: Proprioceptive measurements
        """
        # Get base state
        pos, orn = p.getBasePositionAndOrientation(self.robot_id)
        lin_vel, ang_vel = p.getBaseVelocity(self.robot_id)
        
        # Convert to numpy
        pos = np.array(pos)
        orn = np.array(orn)
        lin_vel = np.array(lin_vel)
        ang_vel = np.array(ang_vel)
        
        # Add noise
        if add_noise:
            pos += np.random.normal(0, self.position_noise_std, 3)
            lin_vel += np.random.normal(0, self.velocity_noise_std, 3)
            orn += np.random.normal(0, self.orientation_noise_std, 4)
            orn /= np.linalg.norm(orn)  # Renormalize quaternion
            ang_vel += np.random.normal(0, self.velocity_noise_std, 3)
        
        # Convert to list if numpy array
        orn_list = orn.tolist() if isinstance(orn, np.ndarray) else list(orn)
        
        return {
            'position': pos,
            'orientation': orn,
            'linear_velocity': lin_vel,
            'angular_velocity': ang_vel,
            'euler': p.getEulerFromQuaternion(orn_list)
        }


class StateEstimator:
    """
    Extended Kalman Filter for multimodal sensor fusion.
    Fuses RGB-D, LiDAR, tactile, and proprioceptive measurements.
    """
    
    def __init__(self, initial_state, initial_covariance):
        """
        Initialize EKF for state estimation.
        
        Args:
            initial_state: Initial state estimate [x, y, z, vx, vy, vz, ...]
            initial_covariance: Initial state covariance matrix
        """
        self.state = np.array(initial_state)
        self.covariance = np.array(initial_covariance)
        
        self.n = len(initial_state)
        
        # Process noise
        self.Q = np.eye(self.n) * 0.01
        
        # Measurement noise for different sensors
        self.R_camera = np.eye(3) * 0.05  # Position from depth camera
        self.R_lidar = np.eye(2) * 0.02   # 2D position from LiDAR
        self.R_proprioception = np.diag([0.001, 0.001, 0.001, 0.01, 0.01, 0.01])
        
    def predict(self, dt, control_input=None):
        """
        EKF prediction step using motion model.
        
        Args:
            dt: Time step
            control_input: Control input applied (optional)
        """
        # Simple constant velocity model
        F = np.eye(self.n)
        F[:3, 3:6] = np.eye(3) * dt  # Position updated by velocity
        
        # Predict state
        self.state = F @ self.state
        
        # Predict covariance
        self.covariance = F @ self.covariance @ F.T + self.Q
        
    def update(self, measurements):
        """
        EKF update step with multimodal measurements.
        
        Args:
            measurements: Dict of sensor measurements
        """
        # Update from proprioception (most reliable)
        if measurements['proprioception'] is not None:
            self._update_proprioception(measurements['proprioception'])
        
        # Update from LiDAR
        if measurements['lidar'] is not None:
            self._update_lidar(measurements['lidar'])
        
        # Update from camera depth
        if measurements['camera'] is not None:
            self._update_camera(measurements['camera'])
        
        # Tactile sensors provide contact info but not direct state measurements
        # Could be used for contact-based localization or obstacle detection
        
    def _update_proprioception(self, proprioception):
        """Update with proprioceptive measurements."""
        # Measurement: [x, y, z, vx, vy, vz]
        z = np.hstack([
            proprioception['position'],
            proprioception['linear_velocity']
        ])
        
        # Measurement matrix (direct observation of position and velocity)
        H = np.zeros((6, self.n))
        H[:6, :6] = np.eye(6)
        
        # Kalman update
        y = z - H @ self.state  # Innovation
        S = H @ self.covariance @ H.T + self.R_proprioception
        K = self.covariance @ H.T @ np.linalg.inv(S)
        
        self.state = self.state + K @ y
        self.covariance = (np.eye(self.n) - K @ H) @ self.covariance
        
    def _update_lidar(self, lidar_data):
        """Update with LiDAR scan data (simplified to position estimate)."""
        # In a real implementation, this would use scan matching
        # For now, we extract position from the LiDAR pose
        pos = lidar_data['pose']['position']
        z = np.array([pos[0], pos[1]])
        
        H = np.zeros((2, self.n))
        H[0, 0] = 1
        H[1, 1] = 1
        
        y = z - H @ self.state
        S = H @ self.covariance @ H.T + self.R_lidar
        K = self.covariance @ H.T @ np.linalg.inv(S)
        
        self.state = self.state + K @ y
        self.covariance = (np.eye(self.n) - K @ H) @ self.covariance
        
    def _update_camera(self, camera_data):
        """Update with RGB-D camera depth measurements."""
        # Extract 3D position from camera
        pos = camera_data['pose']['position']
        z = np.array(pos)
        
        H = np.zeros((3, self.n))
        H[:3, :3] = np.eye(3)
        
        y = z - H @ self.state
        S = H @ self.covariance @ H.T + self.R_camera
        K = self.covariance @ H.T @ np.linalg.inv(S)
        
        self.state = self.state + K @ y
        self.covariance = (np.eye(self.n) - K @ H) @ self.covariance
    
    def get_state(self):
        """Return current state estimate."""
        return self.state.copy()
    
    def get_covariance(self):
        """Return current covariance estimate."""
        return self.covariance.copy()
