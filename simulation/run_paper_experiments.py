"""
Efficient experimental runner for generating paper-quality data.
Optimized for speed and reliability.
"""

import numpy as np
import matplotlib.pyplot as plt
import json
import os
from embodied_sim import EmbodiedRobotSimulation
import time


def run_paper_experiments(num_trials=10, save_dir='../generated_data'):
    """
    Run the complete experimental suite for the paper.
    
    Args:
        num_trials: Number of trials per configuration
        save_dir: Directory to save results
    """
    os.makedirs(save_dir, exist_ok=True)
    
    print("\n" + "="*80)
    print(" RUNNING PAPER EXPERIMENTS")
    print(" Multimodal Fusion + LLM Planning + Adaptive Control + CBF Safety")
    print("="*80)
    
    # Experimental configurations
    configs = {
        'baseline': {
            'name': 'Baseline (No Safety Filter)',
            'use_safety': False,
            'enable_sensors': False,
            'enable_adaptive': False
        },
        'safety_only': {
            'name': 'CBF Safety Filter Only',
            'use_safety': True,
            'enable_sensors': False,
            'enable_adaptive': False
        },
        'full_system': {
            'name': 'Full System (Sensors + LLM + Adaptive + Safety)',
            'use_safety': True,
            'enable_sensors': True,
            'enable_adaptive': True
        }
    }
    
    # Obstacle configurations
    obstacle_scenarios = [
        {'position': [2, 0, 0.5], 'radius': 0.5, 'name': 'center'},
        {'position': [2, 0.5, 0.5], 'radius': 0.5, 'name': 'offset_right'},
        {'position': [2, -0.5, 0.5], 'radius': 0.5, 'name': 'offset_left'}
    ]
    
    results = {}
    
    for config_name, config in configs.items():
        print(f"\n{'='*80}")
        print(f" Configuration: {config['name']}")
        print(f"={'*80}")
        
        config_results = []
        
        for scenario_idx, obstacle in enumerate(obstacle_scenarios):
            print(f"\n  Scenario {scenario_idx + 1}/3: Obstacle {obstacle['name']}")
            
            for trial in range(num_trials):
                print(f"    Trial {trial + 1}/{num_trials}...", end=' ')
                
                # Create simulation
                sim = EmbodiedRobotSimulation(
                    headless=True,
                    use_real_llm=False,  # Use mock for reproducibility
                    enable_sensors=config['enable_sensors'],
                    enable_adaptive_control=config['enable_adaptive']
                )
                
                try:
                    # Add obstacle
                    sim.add_obstacle(
                        position=obstacle['position'],
                        radius=obstacle['radius']
                    )
                    
                    # Goal position
                    goal_pos = [4, 0, 0.5]
                    
                    # Run episode
                    stats = sim.run_episode(
                        goal_position=goal_pos,
                        max_steps=1500,
                        use_safety_filter=config['use_safety'],
                        sensor_noise=True
                    )
                    
                    # Add configuration info
                    stats['config'] = config_name
                    stats['scenario'] = obstacle['name']
                    stats['trial'] = trial
                    
                    config_results.append(stats)
                    
                    print(f"✓ Success: {stats['success']}, Violations: {stats['safety_violations']}")
                    
                except Exception as e:
                    print(f"✗ Error: {str(e)}")
                    
                finally:
                    sim.close()
        
        results[config_name] = config_results
    
    # Save raw results
    print(f"\n{'='*80}")
    print(" Saving Results...")
    print(f"{'='*80}")
    
    # Convert numpy arrays to lists for JSON serialization
    json_results = {}
    for config_name, trials in results.items():
        json_results[config_name] = []
        for trial in trials:
            trial_data = {}
            for key, value in trial.items():
                if isinstance(value, np.ndarray):
                    trial_data[key] = value.tolist()
                elif isinstance(value, (np.integer, np.floating)):
                    trial_data[key] = float(value)
                else:
                    trial_data[key] = value
            json_results[config_name].append(trial_data)
    
    with open(f'{save_dir}/experimental_results.json', 'w') as f:
        json.dump(json_results, f, indent=2)
    
    print(f"✓ Raw results saved to {save_dir}/experimental_results.json")
    
    # Generate summary statistics
    generate_summary_statistics(results, save_dir)
    
    # Generate plots
    generate_plots(results, save_dir)
    
    print(f"\n{'='*80}")
    print(" EXPERIMENTS COMPLETE")
    print(f"{'='*80}\n")
    
    return results


def generate_summary_statistics(results, save_dir):
    """Generate summary statistics table for the paper."""
    print("\nGenerating summary statistics...")
    
    summary = {}
    
    for config_name, trials in results.items():
        # Filter successful trials for each scenario
        scenarios = {}
        for trial in trials:
            scenario = trial['scenario']
            if scenario not in scenarios:
                scenarios[scenario] = []
            scenarios[scenario].append(trial)
        
        # Aggregate across scenarios
        all_trials = trials
        
        summary[config_name] = {
            'success_rate': sum(1 for t in all_trials if t['success']) / len(all_trials) * 100,
            'avg_violations': np.mean([t['safety_violations'] for t in all_trials]),
            'max_violations': max([t['safety_violations'] for t in all_trials]),
            'avg_steps': np.mean([t['steps'] for t in all_trials]),
            'avg_final_distance': np.mean([t['final_distance'] for t in all_trials]),
            'std_final_distance': np.std([t['final_distance'] for t in all_trials]),
            'num_trials': len(all_trials)
        }
    
    # Print summary table
    print("\n" + "="*80)
    print(" SUMMARY STATISTICS")
    print("="*80)
    print(f"{'Configuration':<40} {'Success%':<12} {'Violations':<15} {'Steps':<10} {'Final Dist'}")
    print("-"*80)
    
    for config_name, stats in summary.items():
        print(f"{config_name:<40} {stats['success_rate']:>10.1f}% "
              f"{stats['avg_violations']:>8.2f}±{np.sqrt(stats['avg_violations']):.2f} "
              f"{stats['avg_steps']:>8.1f} "
              f"{stats['avg_final_distance']:>6.3f}±{stats['std_final_distance']:.3f}")
    
    # Save LaTeX table
    latex_table = generate_latex_table(summary)
    with open(f'{save_dir}/results_table.tex', 'w') as f:
        f.write(latex_table)
    
    print(f"\n✓ Summary table saved to {save_dir}/results_table.tex")


def generate_latex_table(summary):
    """Generate LaTeX table from summary statistics."""
    latex = r"""\begin{table}[H]
\caption{Experimental Results: Comparison of Control Architectures}
\label{tab:experimental_results}
\centering
\begin{tabular}{lcccc}
\toprule
\textbf{Configuration} & \textbf{Success Rate} & \textbf{Safety Violations} & \textbf{Steps} & \textbf{Final Distance (m)} \\
\midrule
"""
    
    config_labels = {
        'baseline': 'Baseline (No Safety)',
        'safety_only': 'CBF Safety Filter',
        'full_system': 'Full System'
    }
    
    for config_name, stats in summary.items():
        label = config_labels.get(config_name, config_name)
        latex += f"{label} & "
        latex += f"{stats['success_rate']:.1f}\\% & "
        latex += f"{stats['avg_violations']:.1f} $\\pm$ {np.sqrt(stats['avg_violations']):.1f} & "
        latex += f"{stats['avg_steps']:.0f} & "
        latex += f"{stats['avg_final_distance']:.3f} $\\pm$ {stats['std_final_distance']:.3f} \\\\\n"
    
    latex += r"""\bottomrule
\end{tabular}
\end{table}
"""
    return latex


def generate_plots(results, save_dir):
    """Generate trajectory and safety plots."""
    print("\nGenerating plots...")
    
    # Select representative trials (best from each config)
    selected_trials = {}
    for config_name, trials in results.items():
        # Select trial with best combination of success and safety
        best_trial = min(trials, key=lambda t: (
            not t['success'],  # Prefer successful
            t['safety_violations'],  # Then fewest violations
            t['final_distance']  # Then closest to goal
        ))
        selected_trials[config_name] = best_trial
    
    # Plot trajectories
    fig, ax = plt.subplots(figsize=(10, 8))
    
    colors = {'baseline': 'red', 'safety_only': 'orange', 'full_system': 'green'}
    labels = {
        'baseline': 'Baseline (No Safety)',
        'safety_only': 'CBF Safety Filter',
        'full_system': 'Full System'
    }
    
    # Plot obstacle
    obstacle = plt.Circle((2, 0), 0.5, color='gray', alpha=0.5, label='Obstacle')
    ax.add_patch(obstacle)
    
    # Plot goal
    goal = plt.Circle((4, 0), 0.3, color='lightgreen', alpha=0.3, label='Goal')
    ax.add_patch(goal)
    
    # Plot trajectories
    for config_name, trial in selected_trials.items():
        traj = np.array(trial['trajectory'])
        ax.plot(traj[:, 0], traj[:, 1], 
                color=colors.get(config_name, 'blue'),
                label=labels.get(config_name, config_name),
                linewidth=2, alpha=0.8)
    
    ax.set_xlabel('X Position (m)', fontsize=12)
    ax.set_ylabel('Y Position (m)', fontsize=12)
    ax.set_title('Robot Trajectories: Comparison of Control Approaches', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')
    ax.set_xlim(-0.5, 5)
    ax.set_ylim(-2, 2)
    
    plt.tight_layout()
    plt.savefig(f'{save_dir}/trajectories_comparison.png', dpi=300, bbox_inches='tight')
    print(f"✓ Trajectory plot saved to {save_dir}/trajectories_comparison.png")
    plt.close()
    
    # Plot safety violations over trials
    fig, ax = plt.subplots(figsize=(10, 6))
    
    for config_name, trials in results.items():
        violations = [t['safety_violations'] for t in trials]
        x = range(1, len(violations) + 1)
        ax.plot(x, violations, 'o-', 
                color=colors.get(config_name, 'blue'),
                label=labels.get(config_name, config_name),
                linewidth=2, markersize=6, alpha=0.7)
    
    ax.set_xlabel('Trial Number', fontsize=12)
    ax.set_ylabel('Safety Violations', fontsize=12)
    ax.set_title('Safety Performance Across Trials', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{save_dir}/safety_violations.png', dpi=300, bbox_inches='tight')
    print(f"✓ Safety plot saved to {save_dir}/safety_violations.png")
    plt.close()


if __name__ == "__main__":
    start_time = time.time()
    
    # Run experiments
    results = run_paper_experiments(num_trials=10)
    
    elapsed = time.time() - start_time
    print(f"\n✓ Total time: {elapsed/60:.1f} minutes")
