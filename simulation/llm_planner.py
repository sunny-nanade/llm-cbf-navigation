"""
LLM-based planning with adaptive control gains.
"""
import numpy as np
from typing import Dict, List, Tuple, Optional


class AdaptiveControlGains:
    """Adaptive PID gains that adjust based on tracking error."""
    
    def __init__(self, initial_kp=200.0, initial_ki=5.0, initial_kd=20.0, adaptation_rate=0.01):
        self.kp = initial_kp
        self.ki = initial_ki
        self.kd = initial_kd
        self.adaptation_rate = adaptation_rate
        self.error_history = []
        
    def adapt(self, tracking_error: float, error_derivative: float = 0.0, dt: float = 0.01):
        """Adapt gains based on tracking error magnitude."""
        self.error_history.append(tracking_error)
        if len(self.error_history) > 50:
            self.error_history.pop(0)
        
        avg_error = np.mean(np.abs(self.error_history))
        
        if avg_error > 1.0:
            self.kp += self.adaptation_rate
            self.kd += self.adaptation_rate * 0.5
        elif avg_error < 0.3:
            self.kp = max(100.0, self.kp - self.adaptation_rate)
            self.kd = max(10.0, self.kd - self.adaptation_rate * 0.5)
    
    def get_gains(self) -> Tuple[float, float, float]:
        """Return current PID gains."""
        return self.kp, self.ki, self.kd


class LLMPlanner:
    """LLM-based planner with mock mode for testing."""
    
    def __init__(self, use_mock=True, api_key=None):
        self.use_mock = use_mock
        self.api_key = api_key
        self.confidence_threshold = 0.7
        
    def plan(self, current_state: Dict, goal_state: Dict, obstacles: List, workspace_bounds: List[float]) -> Tuple[List, float]:
        """Generate waypoints based on sensor data."""
        if self.use_mock:
            return self._mock_plan(current_state, goal_state, obstacles, workspace_bounds)
        else:
            return self._plan_with_llm(current_state, goal_state, obstacles, workspace_bounds)
    
    def _mock_plan(self, current_state: Dict, goal_state: Dict, obstacles: List, workspace_bounds: List[float]) -> Tuple[List, float]:
        """Mock planner that generates simple waypoints avoiding obstacles."""
        robot_pos = current_state.get('position', np.array([0.0, 0.0, 0.5]))[:2]
        goal_pos_raw = goal_state.get('position', np.array([5.0, 5.0, 0.5]))
        
        # Convert to numpy array if it's a list, then get 2D position
        if isinstance(goal_pos_raw, list):
            goal_pos = np.array(goal_pos_raw)[:2]
        else:
            goal_pos = goal_pos_raw[:2]
        
        # Smart waypoint planning: find minimal detour around obstacles
        start_pos = current_state['position'][:2]
        
        # If no obstacles or simple scenario, go direct
        if not obstacles or len(obstacles) == 0:
            waypoints = [goal_pos.tolist() if isinstance(goal_pos, np.ndarray) else list(goal_pos)]
            confidence = 0.95
            return waypoints, confidence
        
        # Calculate direct path vector
        direct_vec = np.array(goal_pos) - np.array(start_pos)
        direct_dist = np.linalg.norm(direct_vec)
        
        # Check if any obstacles block direct path
        obstacles_in_way = []
        for obs in obstacles:
            obs_pos = np.array(obs['position'][:2])
            obs_radius = obs['radius']
            
            # Project obstacle onto direct path
            if direct_dist > 0:
                t = np.dot(obs_pos - start_pos, direct_vec) / (direct_dist ** 2)
                t = np.clip(t, 0, 1)  # Clamp to path segment
                closest_point = start_pos + t * direct_vec
                dist_to_path = np.linalg.norm(obs_pos - closest_point)
                
                # If obstacle is close to direct path, need to avoid it
                if dist_to_path < (obs_radius + 0.8):  # 0.8m safety margin
                    obstacles_in_way.append({
                        'pos': obs_pos, 
                        'radius': obs_radius,
                        't': t  # Position along path (0=start, 1=goal)
                    })
        
        # If path is clear, go direct
        if len(obstacles_in_way) == 0:
            waypoints = [goal_pos.tolist() if isinstance(goal_pos, np.ndarray) else list(goal_pos)]
            confidence = 0.95
        elif len(obstacles_in_way) <= 3:
            # Simple/Medium scenario: 1-3 obstacles, use single waypoint detour
            # Find the obstacle that's most in the way (closest to midpoint of path)
            mid_point = start_pos + 0.5 * direct_vec
            most_blocking = min(obstacles_in_way, 
                              key=lambda o: np.linalg.norm(o['pos'] - mid_point))
            obs_pos = most_blocking['pos']
            obs_radius = most_blocking['radius']
            
            # Perpendicular direction to direct path
            if direct_dist > 0:
                perp_dir = np.array([-direct_vec[1], direct_vec[0]]) / direct_dist
            else:
                perp_dir = np.array([0, 1])
            
            # Create two waypoint options
            margin = obs_radius + 1.5  # Larger margin for safer navigation
            option1 = obs_pos + perp_dir * margin
            option2 = obs_pos - perp_dir * margin
            
            # Score each option by: distance to workspace center + obstacles nearby
            center = np.array([5.0, 5.0])
            score1 = np.linalg.norm(option1 - center) * 0.5  # Workspace center score
            score2 = np.linalg.norm(option2 - center) * 0.5
            
            # Count obstacles within 2.5m of each option
            for obs in obstacles:
                obs_p = np.array(obs['position'][:2])
                if np.linalg.norm(obs_p - option1) < 2.5:
                    score1 += 3.0  # Penalty for nearby obstacles
                if np.linalg.norm(obs_p - option2) < 2.5:
                    score2 += 3.0
            
            # Pick option with better score (lower is better)
            waypoint = option2 if score2 < score1 else option1
            
            # Clamp waypoint to workspace bounds
            waypoint = np.clip(waypoint, 0.5, 9.5)
            
            waypoints = [
                waypoint.tolist(),
                goal_pos.tolist() if isinstance(goal_pos, np.ndarray) else list(goal_pos)
            ]
            confidence = 0.88
        else:
            # Dense obstacles (4+): Multi-waypoint path
            # Sort obstacles by position along direct path
            obstacles_in_way.sort(key=lambda o: o['t'])
            
            waypoints = []
            current_pos = start_pos
            
            # Create waypoints around each major obstacle cluster
            for i, obs in enumerate(obstacles_in_way[:3]):  # Handle up to 3 major obstacles
                obs_pos = obs['pos']
                obs_radius = obs['radius']
                
                # Direction from current position to this obstacle
                to_obs = obs_pos - current_pos
                dist_to_obs = np.linalg.norm(to_obs)
                
                if dist_to_obs > 0.1:
                    # Perpendicular to approach direction
                    perp = np.array([-to_obs[1], to_obs[0]]) / dist_to_obs
                    
                    # Try both sides, pick the one with fewer obstacles nearby
                    option1 = obs_pos + perp * (obs_radius + 1.5)
                    option2 = obs_pos - perp * (obs_radius + 1.5)
                    
                    # Count obstacles near each option
                    count1 = sum(1 for o in obstacles if np.linalg.norm(np.array(o['position'][:2]) - option1) < 2.0)
                    count2 = sum(1 for o in obstacles if np.linalg.norm(np.array(o['position'][:2]) - option2) < 2.0)
                    
                    waypoint = option2 if count2 < count1 else option1
                    
                    # Clamp to workspace
                    waypoint = np.clip(waypoint, 0.5, 9.5)
                    waypoints.append(waypoint.tolist())
                    current_pos = waypoint
            
            # Final waypoint is the goal
            waypoints.append(goal_pos.tolist() if isinstance(goal_pos, np.ndarray) else list(goal_pos))
            confidence = 0.85  # Lower confidence for complex multi-waypoint paths
        
        return waypoints, confidence
    
    def _plan_with_llm(self, current_state: Dict, goal_state: Dict, obstacles: List, workspace_bounds: List[float]) -> Tuple[List, float]:
        """Call OpenAI API for planning (not implemented yet)."""
        print("[LLM] API mode not implemented, falling back to mock")
        return self._mock_plan(current_state, goal_state, obstacles, workspace_bounds)
    
    def apply_confidence_gate(self, waypoints: List, confidence: float, current_state: Dict, goal_state: Dict) -> Tuple[List, bool]:
        """Check if plan confidence exceeds threshold."""
        accepted = confidence >= self.confidence_threshold
        return waypoints, accepted
