import os
import json
import numpy as np
import matplotlib.pyplot as plt

def load_results(file_path):
    """Loads simulation results from a JSON file."""
    with open(file_path, 'r') as f:
        return json.load(f)

def generate_plots(baseline_data, safe_data, output_dir):
    """Generates and saves plots comparing baseline and safe controller."""
    
    # --- Trajectory Plot ---
    plt.figure(figsize=(8, 8))
    
    # Plot obstacle
    obstacle_pos = np.array([2.5, 2.5])
    obstacle_radius = 0.5
    circle = plt.Circle(obstacle_pos, obstacle_radius, color='r', alpha=0.5, label="Obstacle")
    plt.gca().add_artist(circle)

    # Plot trajectories
    baseline_pos = np.array(baseline_data['pos'])
    safe_pos = np.array(safe_data['pos'])
    plt.plot(baseline_pos[:, 0], baseline_pos[:, 1], 'b--', label="Baseline")
    plt.plot(safe_pos[:, 0], safe_pos[:, 1], 'g-', label="Safety-Filtered")
    
    # Plot start and goal
    plt.plot(baseline_pos[0, 0], baseline_pos[0, 1], 'ko', markersize=10, label="Start")
    plt.plot(5.0, 5.0, 'k*', markersize=15, label="Goal")

    plt.xlabel("X Position (m)")
    plt.ylabel("Y Position (m)")
    plt.title("Robot Trajectory")
    plt.legend()
    plt.grid(True)
    plt.axis('equal')
    plt.savefig(os.path.join(output_dir, "trajectory.png"))
    plt.close()

    # --- Safety Barrier Function Plot ---
    plt.figure(figsize=(10, 5))
    plt.plot(safe_data['time'], safe_data['h'], 'g-', label="h(x)")
    plt.axhline(0, color='r', linestyle='--', label="h(x) = 0 (Safety Boundary)")
    plt.xlabel("Time (s)")
    plt.ylabel("Barrier Function h(x)")
    plt.title("Safety Barrier Function Value Over Time")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, "safety.png"))
    plt.close()

def generate_tables(baseline_data, safe_data, output_dir):
    """Generates and saves result tables in LaTeX format."""
    
    # --- Metrics Summary Table ---
    metrics = {
        "Metric": ["Safety Violations", "Goal Reached", "Final Distance to Goal (m)"],
        "Baseline": [
            baseline_data['safety_violations'],
            "Yes" if baseline_data['goal_reached'] else "No",
            f"{np.linalg.norm(np.array(baseline_data['pos'][-1]) - np.array([5.0, 5.0])):.2f}"
        ],
        "Safety-Filtered": [
            safe_data['safety_violations'],
            "Yes" if safe_data['goal_reached'] else "No",
            f"{np.linalg.norm(np.array(safe_data['pos'][-1]) - np.array([5.0, 5.0])):.2f}"
        ]
    }

    # Create LaTeX table
    latex_table = """\\begin{table}[H]
\\centering
\\caption{Summary of Performance Metrics}
\\label{tab:metrics_summary_real}
\\begin{tabular}{lcc}
\\toprule
\\textbf{Metric} & \\textbf{Baseline} & \\textbf{Safety-Filtered} \\\\
\\midrule
"""
    for i in range(len(metrics["Metric"])):
        latex_table += f"{metrics['Metric'][i]} & {metrics['Baseline'][i]} & {metrics['Safety-Filtered'][i]} \\\\\n"
    
    latex_table += """\\bottomrule
\\end{tabular}
\\end{table}
"""
    with open(os.path.join(output_dir, "metrics_summary_real.tex"), "w") as f:
        f.write(latex_table)


def main():
    """
    Main function to analyze results and generate outputs.
    """
    results_dir = "simulation_results"
    output_dir = "generated_figures"
    os.makedirs(output_dir, exist_ok=True)

    print("Loading simulation results...")
    try:
        baseline_data = load_results(os.path.join(results_dir, "baseline.json"))
        safe_data = load_results(os.path.join(results_dir, "safe.json"))
    except FileNotFoundError:
        print("Error: Result files not found. Please run the simulation first.")
        return

    print("Generating plots...")
    generate_plots(baseline_data, safe_data, output_dir)
    print(f"Plots saved to {output_dir}")

    print("Generating LaTeX tables...")
    generate_tables(baseline_data, safe_data, output_dir)
    print(f"Tables saved to {output_dir}")

if __name__ == "__main__":
    main()
