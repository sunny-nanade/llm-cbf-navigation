import numpy as np
import cvxpy as cp

class PIDController:
    """A simple PID controller."""
    def __init__(self, kp, ki, kd, setpoint):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.setpoint = setpoint
        self.integral = 0
        self.previous_error = 0

    def compute(self, current_value, dt):
        error = self.setpoint - current_value
        self.integral += error * dt
        derivative = (error - self.previous_error) / dt
        
        output = self.kp * error + self.ki * self.integral + self.kd * derivative
        
        self.previous_error = error
        return output

class SafetyFilter:
    """
    A Control Barrier Function (CBF) based Quadratic Program (QP) safety filter.
    Uses slack variables to ensure feasibility even with strict constraints.
    """
    def __init__(self, dim, alpha=0.5, slack_penalty=1000.0):
        self.dim = dim
        self.alpha = alpha
        self.slack_penalty = slack_penalty
        
        # CVXPY variables for the QP with slack
        self.u = cp.Variable(dim)
        self.slack = cp.Variable(nonneg=True)  # Non-negative slack variable
        self.u_ref = cp.Parameter(dim)
        self.Lfh = cp.Parameter()
        self.Lgh = cp.Parameter((1, dim))
        self.h = cp.Parameter()
        self.u_max = cp.Parameter(dim)
        self.u_min = cp.Parameter(dim)

        # QP formulation with slack and input bounds
        objective = cp.Minimize(
            cp.sum_squares(self.u - self.u_ref) + self.slack_penalty * self.slack
        )
        constraints = [
            self.Lgh @ self.u + self.Lfh + self.alpha * self.h >= -self.slack,
            self.u <= self.u_max,
            self.u >= self.u_min
        ]
        self.problem = cp.Problem(objective, constraints)
        
        # Track statistics
        self.num_infeasible = 0
        self.num_slack_used = 0

    def solve(self, u_ref, h, Lfh, Lgh, u_max=None, u_min=None):
        """
        Solves the safety QP to find a safe control input.
        
        Args:
            u_ref: The reference control input from the performance controller.
            h: The value of the barrier function h(x).
            Lfh: The Lie derivative of h along f, L_f h(x).
            Lgh: The Lie derivative of h along g, L_g h(x).
            u_max: Maximum control input (default: [10, 10])
            u_min: Minimum control input (default: [-10, -10])

        Returns:
            The safe control input u_safe.
        """
        if u_max is None:
            u_max = np.ones(self.dim) * 10.0
        if u_min is None:
            u_min = -np.ones(self.dim) * 10.0
            
        self.u_ref.value = u_ref
        self.h.value = h
        self.Lfh.value = Lfh
        self.Lgh.value = Lgh
        self.u_max.value = u_max
        self.u_min.value = u_min

        try:
            self.problem.solve(solver=cp.OSQP, verbose=False)
            
            if self.problem.status in [cp.OPTIMAL, cp.OPTIMAL_INACCURATE]:
                if self.slack.value > 1e-4:
                    self.num_slack_used += 1
                return self.u.value
            else:
                # QP is infeasible even with slack - return minimal intervention
                self.num_infeasible += 1
                if self.num_infeasible % 50 == 1:  # Print only occasionally
                    print(f"Warning: Safety QP infeasible (count: {self.num_infeasible})")
                # Return safe fallback: stop or minimal control
                return np.zeros(self.dim)
        except Exception as e:
            print(f"Error solving QP: {e}")
            return np.zeros(self.dim)
    
    def reset_statistics(self):
        """Reset tracking statistics."""
        self.num_infeasible = 0
        self.num_slack_used = 0

# --- System Dynamics and Barrier Functions ---

def system_dynamics(x):
    """
    Defines the system dynamics f(x) and g(x) for a simple integrator model.
    x = [pos_x, pos_y, vel_x, vel_y]
    u = [acc_x, acc_y]
    
    Returns:
        f(x): The drift dynamics.
        g(x): The control-affine dynamics.
    """
    f = np.array([x[2], x[3], 0, 0])
    g = np.array([
        [0, 0],
        [0, 0],
        [1, 0],
        [0, 1]
    ])
    return f, g

def obstacle_barrier_function(x, obstacle_pos, obstacle_radius):
    """
    Defines a barrier function to avoid a circular obstacle.
    h(x) = ||x_pos - obs_pos||^2 - r^2
    
    Args:
        x: System state [pos_x, pos_y, vel_x, vel_y]
        obstacle_pos: [x, y] position of the obstacle.
        obstacle_radius: Radius of the obstacle.
        
    Returns:
        h: Value of the barrier function.
        Lfh: Lie derivative of h along f.
        Lgh: Lie derivative of h along g.
    """
    x_pos = x[:2]
    x_vel = x[2:]
    
    dist_vec = x_pos - obstacle_pos
    
    h = np.dot(dist_vec, dist_vec) - obstacle_radius**2
    
    # Lie derivatives
    f, g = system_dynamics(x)
    grad_h = np.hstack([2 * dist_vec, np.zeros(2)]) # Gradient of h w.r.t x
    
    Lfh = np.dot(grad_h, f)
    Lgh = np.dot(grad_h, g).reshape(1, -1) # Ensure shape is (1, dim_u)
    
    return h, Lfh, Lgh

def mock_llm_planner(start_pos, goal_pos, obstacle_pos, obstacle_radius):
    """
    Simulates an LLM providing high-level waypoints.
    For this simple case, it generates one intermediate waypoint to go "around" the obstacle.
    """
    waypoints = [start_pos]
    
    # A simple strategy: if the obstacle is between start and goal, go to the side.
    direct_path_vec = goal_pos - start_pos
    obstacle_path_vec = obstacle_pos - start_pos
    
    # Project obstacle onto the direct path
    projection = np.dot(obstacle_path_vec, direct_path_vec) / np.dot(direct_path_vec, direct_path_vec)
    
    if 0 < projection < 1: # Obstacle is between start and goal
        # Go to a point perpendicular to the obstacle
        perp_vec = np.array([-direct_path_vec[1], direct_path_vec[0]])
        perp_vec /= np.linalg.norm(perp_vec)
        
        intermediate_point = obstacle_pos + perp_vec * (obstacle_radius * 1.5)
        waypoints.append(intermediate_point)

    waypoints.append(goal_pos)
    return waypoints
