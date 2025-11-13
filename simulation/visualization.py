"""
Visual Feedback System for Publication-Quality Simulation
Implements LiDAR rays, sensor panels, status indicators, and real-time graphs.
"""

import pybullet as p
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg
from PIL import Image
import io


class VisualizationManager:
    """Manages all visual feedback elements for the simulation."""
    
    def __init__(self, robot_id, sim_client):
        """
        Initialize visualization manager.
        
        Args:
            robot_id: PyBullet robot ID
            sim_client: PyBullet client ID
        """
        self.robot_id = robot_id
        self.sim_client = sim_client
        self.lidar_line_ids = []
        self.text_ids = []
        
        # Color scheme
        self.COLOR_LIDAR_CLEAR = [0, 1, 0]  # Green for clear rays
        self.COLOR_LIDAR_DETECT = [1, 0, 0]  # Red for obstacle detection
        self.COLOR_SAFETY_OK = [0, 1, 0]  # Green for safe
        self.COLOR_SAFETY_FILTER = [1, 1, 0]  # Yellow for filter active
        self.COLOR_SAFETY_WARNING = [1, 0, 0]  # Red for near collision
        
        # History for graphs
        self.gains_history = {'kp': [], 'ki': [], 'kd': [], 'time': []}
        self.safety_history = {'status': [], 'min_distance': [], 'time': []}
        
    def draw_lidar_rays(self, lidar_data, obstacle_threshold=5.0, subsample=4):
        """
        Draw LiDAR rays in 3D space.
        
        Args:
            lidar_data: LiDAR scan data from sensor
            obstacle_threshold: Distance threshold for obstacle detection (meters)
            subsample: Draw every Nth ray to reduce clutter
        """
        # Clear previous rays
        for line_id in self.lidar_line_ids:
            p.removeUserDebugItem(line_id, physicsClientId=self.sim_client)
        self.lidar_line_ids.clear()
        
        if lidar_data is None:
            return
        
        ranges = lidar_data['ranges']
        angles = lidar_data['angles']
        pose = lidar_data['pose']
        pos = pose['position']
        orn = pose['orientation']
        
        # Get rotation matrix
        rot_matrix = p.getMatrixFromQuaternion(orn, physicsClientId=self.sim_client)
        rot_matrix = np.array(rot_matrix).reshape(3, 3)
        
        # Draw subsampled rays
        for i in range(0, len(ranges), subsample):
            angle = angles[i]
            measured_range = ranges[i]
            
            # Ray direction in world frame
            local_dir = np.array([np.cos(angle), np.sin(angle), 0])
            world_dir = rot_matrix @ local_dir
            
            # Ray endpoint
            ray_end = np.array(pos) + world_dir * measured_range
            
            # Color based on detection
            if measured_range < obstacle_threshold:
                color = self.COLOR_LIDAR_DETECT
                line_width = 2.0
            else:
                color = self.COLOR_LIDAR_CLEAR
                line_width = 1.0
            
            # Draw line
            line_id = p.addUserDebugLine(
                lineFromXYZ=pos,
                lineToXYZ=ray_end.tolist(),
                lineColorRGB=color,
                lineWidth=line_width,
                lifeTime=0.1,  # Persist for next few frames
                physicsClientId=self.sim_client
            )
            self.lidar_line_ids.append(line_id)
    
    def draw_safety_status(self, safety_active, min_obstacle_distance, position):
        """
        Draw safety filter status indicator.
        
        Args:
            safety_active: Whether safety filter is currently active
            min_obstacle_distance: Minimum distance to any obstacle
            position: Robot position for placing indicator
        """
        # Determine status color
        if min_obstacle_distance < 0.5:
            color = self.COLOR_SAFETY_WARNING
            status_text = "SAFETY: CRITICAL"
        elif safety_active:
            color = self.COLOR_SAFETY_FILTER
            status_text = "SAFETY: FILTER ACTIVE"
        else:
            color = self.COLOR_SAFETY_OK
            status_text = "SAFETY: OK"
        
        # Draw status sphere above robot
        indicator_pos = [position[0], position[1], position[2] + 0.5]
        
        # Remove old indicator
        if hasattr(self, 'safety_sphere_id'):
            p.removeBody(self.safety_sphere_id, physicsClientId=self.sim_client)
        
        # Create visual sphere (collision disabled)
        visual_shape = p.createVisualShape(
            shapeType=p.GEOM_SPHERE,
            radius=0.1,
            rgbaColor=color + [0.8],
            physicsClientId=self.sim_client
        )
        
        self.safety_sphere_id = p.createMultiBody(
            baseMass=0,
            baseVisualShapeIndex=visual_shape,
            basePosition=indicator_pos,
            physicsClientId=self.sim_client
        )
        
        # Add text label
        self._add_text(status_text, 
                      [position[0], position[1], position[2] + 0.7],
                      color=color,
                      text_size=1.2)
    
    def draw_llm_decision(self, waypoint, confidence, planning_mode, position):
        """
        Draw LLM planner decision overlay.
        
        Args:
            waypoint: Current target waypoint (x, y)
            confidence: Planner confidence (0-1)
            planning_mode: String describing current mode
            position: Robot position for text placement
        """
        # Draw waypoint marker
        if waypoint is not None:
            waypoint_3d = [waypoint[0], waypoint[1], 0.1]
            
            # Remove old waypoint marker
            if hasattr(self, 'waypoint_marker_id'):
                p.removeBody(self.waypoint_marker_id, physicsClientId=self.sim_client)
            
            # Create waypoint visual
            visual_shape = p.createVisualShape(
                shapeType=p.GEOM_CYLINDER,
                radius=0.15,
                length=0.05,
                rgbaColor=[0, 0.8, 1, 0.6],
                physicsClientId=self.sim_client
            )
            
            self.waypoint_marker_id = p.createMultiBody(
                baseMass=0,
                baseVisualShapeIndex=visual_shape,
                basePosition=waypoint_3d,
                physicsClientId=self.sim_client
            )
            
            # Draw line from robot to waypoint
            p.addUserDebugLine(
                lineFromXYZ=[position[0], position[1], 0.1],
                lineToXYZ=waypoint_3d,
                lineColorRGB=[0, 0.8, 1],
                lineWidth=2.0,
                lifeTime=0.1,
                physicsClientId=self.sim_client
            )
        
        # Add text overlay
        text = f"Waypoint: ({waypoint[0]:.2f}, {waypoint[1]:.2f})\n"
        text += f"Confidence: {confidence:.0%}\n"
        text += f"Mode: {planning_mode}"
        
        self._add_text(text,
                      [position[0] - 1.0, position[1], position[2] + 1.0],
                      color=[0, 0.8, 1],
                      text_size=1.0)
    
    def update_gains_history(self, kp, ki, kd, current_time):
        """
        Update adaptive gains history for graphing.
        
        Args:
            kp, ki, kd: Current PID gains
            current_time: Simulation time
        """
        self.gains_history['kp'].append(kp)
        self.gains_history['ki'].append(ki)
        self.gains_history['kd'].append(kd)
        self.gains_history['time'].append(current_time)
    
    def draw_gains_graph(self, position):
        """
        Draw real-time adaptive gains graph as overlay.
        
        Args:
            position: Robot position for graph placement
        """
        if len(self.gains_history['time']) < 2:
            return
        
        # Create matplotlib figure
        fig, ax = plt.subplots(figsize=(4, 3), facecolor='white')
        
        times = self.gains_history['time']
        ax.plot(times, self.gains_history['kp'], 'r-', label='Kp', linewidth=2)
        ax.plot(times, self.gains_history['ki'], 'g-', label='Ki', linewidth=2)
        ax.plot(times, self.gains_history['kd'], 'b-', label='Kd', linewidth=2)
        
        ax.set_xlabel('Time (s)', fontsize=10)
        ax.set_ylabel('Gain Value', fontsize=10)
        ax.set_title('Adaptive Gains Evolution', fontsize=12, fontweight='bold')
        ax.legend(loc='best', fontsize=9)
        ax.grid(True, alpha=0.3)
        
        # Convert to image (for future overlay implementation)
        # This is a placeholder - full overlay requires OpenGL texture mapping
        plt.close(fig)
    
    def _add_text(self, text, position, color=[1, 1, 1], text_size=1.0):
        """
        Add debug text to simulation.
        
        Args:
            text: Text string to display
            position: 3D position [x, y, z]
            color: RGB color
            text_size: Text size multiplier
        """
        # Remove old text
        for text_id in self.text_ids:
            p.removeUserDebugItem(text_id, physicsClientId=self.sim_client)
        self.text_ids.clear()
        
        # Add new text
        text_id = p.addUserDebugText(
            text=text,
            textPosition=position,
            textColorRGB=color,
            textSize=text_size,
            lifeTime=0.1,
            physicsClientId=self.sim_client
        )
        self.text_ids.append(text_id)
    
    def clear_all(self):
        """Clear all visualization elements."""
        # Clear LiDAR rays
        for line_id in self.lidar_line_ids:
            p.removeUserDebugItem(line_id, physicsClientId=self.sim_client)
        self.lidar_line_ids.clear()
        
        # Clear text
        for text_id in self.text_ids:
            p.removeUserDebugItem(text_id, physicsClientId=self.sim_client)
        self.text_ids.clear()
        
        # Clear markers
        if hasattr(self, 'safety_sphere_id'):
            try:
                p.removeBody(self.safety_sphere_id, physicsClientId=self.sim_client)
            except:
                pass
        
        if hasattr(self, 'waypoint_marker_id'):
            try:
                p.removeBody(self.waypoint_marker_id, physicsClientId=self.sim_client)
            except:
                pass
    
    def reset(self):
        """Reset all history and visualization state."""
        self.clear_all()
        self.gains_history = {'kp': [], 'ki': [], 'kd': [], 'time': []}
        self.safety_history = {'status': [], 'min_distance': [], 'time': []}


class SensorDashboard:
    """
    Create sensor data dashboard overlay for camera view.
    """
    
    def __init__(self, figure_size=(10, 8)):
        """
        Initialize dashboard.
        
        Args:
            figure_size: Size of dashboard figure
        """
        self.fig = plt.figure(figsize=figure_size, facecolor='black')
        self.axes = {}
        
        # Create subplot layout
        gs = self.fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)
        
        # RGB-D Camera view
        self.axes['camera'] = self.fig.add_subplot(gs[0, 0])
        self.axes['camera'].set_title('RGB-D Camera', color='white', fontsize=10)
        self.axes['camera'].axis('off')
        
        # Depth view
        self.axes['depth'] = self.fig.add_subplot(gs[0, 1])
        self.axes['depth'].set_title('Depth Map', color='white', fontsize=10)
        self.axes['depth'].axis('off')
        
        # LiDAR polar plot
        self.axes['lidar'] = self.fig.add_subplot(gs[1, :], projection='polar')
        self.axes['lidar'].set_title('LiDAR Scan (360°)', color='white', fontsize=10)
        self.axes['lidar'].set_facecolor('#1a1a1a')
        
        # State estimate
        self.axes['state'] = self.fig.add_subplot(gs[2, 0])
        self.axes['state'].set_title('State Estimate', color='white', fontsize=10)
        self.axes['state'].set_facecolor('#1a1a1a')
        
        # Adaptive gains
        self.axes['gains'] = self.fig.add_subplot(gs[2, 1])
        self.axes['gains'].set_title('Adaptive Gains', color='white', fontsize=10)
        self.axes['gains'].set_facecolor('#1a1a1a')
        
    def update(self, sensor_data, state_estimate, adaptive_gains, gains_history):
        """
        Update dashboard with latest data.
        
        Args:
            sensor_data: Dictionary of sensor measurements
            state_estimate: Current state estimate
            adaptive_gains: Current adaptive gains object
            gains_history: History of gains for plotting
        """
        # Update camera view
        if sensor_data.get('rgbd') is not None:
            rgb_image = sensor_data['rgbd']['rgb']
            depth_image = sensor_data['rgbd']['depth']
            
            self.axes['camera'].clear()
            self.axes['camera'].imshow(rgb_image)
            self.axes['camera'].set_title('RGB-D Camera', color='white')
            self.axes['camera'].axis('off')
            
            self.axes['depth'].clear()
            self.axes['depth'].imshow(depth_image, cmap='viridis')
            self.axes['depth'].set_title('Depth Map', color='white')
            self.axes['depth'].axis('off')
        
        # Update LiDAR
        if sensor_data.get('lidar') is not None:
            lidar = sensor_data['lidar']
            angles = lidar['angles']
            ranges = lidar['ranges']
            
            self.axes['lidar'].clear()
            self.axes['lidar'].plot(angles, ranges, 'g-', linewidth=1.5)
            self.axes['lidar'].fill(angles, ranges, 'g', alpha=0.3)
            self.axes['lidar'].set_ylim(0, lidar['max_range'])
            self.axes['lidar'].set_title('LiDAR Scan (360°)', color='white')
            self.axes['lidar'].grid(True, alpha=0.3)
        
        # Update state estimate
        self.axes['state'].clear()
        if state_estimate is not None:
            state_text = f"Position: ({state_estimate[0]:.2f}, {state_estimate[1]:.2f})\n"
            state_text += f"Velocity: ({state_estimate[3]:.2f}, {state_estimate[4]:.2f})"
            self.axes['state'].text(0.5, 0.5, state_text, 
                                   ha='center', va='center',
                                   fontsize=12, color='white',
                                   transform=self.axes['state'].transAxes)
        self.axes['state'].axis('off')
        
        # Update gains history
        self.axes['gains'].clear()
        if len(gains_history['time']) > 0:
            times = gains_history['time']
            self.axes['gains'].plot(times, gains_history['kp'], 'r-', label='Kp', linewidth=2)
            self.axes['gains'].plot(times, gains_history['ki'], 'g-', label='Ki', linewidth=2)
            self.axes['gains'].plot(times, gains_history['kd'], 'b-', label='Kd', linewidth=2)
            self.axes['gains'].set_xlabel('Time (s)', color='white')
            self.axes['gains'].set_ylabel('Gain', color='white')
            self.axes['gains'].legend(loc='best')
            self.axes['gains'].grid(True, alpha=0.3)
            self.axes['gains'].tick_params(colors='white')
        
        plt.tight_layout()
    
    def save_frame(self, filepath):
        """Save current dashboard as image."""
        self.fig.savefig(filepath, facecolor='black', dpi=100)
    
    def close(self):
        """Close dashboard figure."""
        plt.close(self.fig)
