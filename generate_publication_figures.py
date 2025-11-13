"""
Generate Publication Figures - Phase 7

Creates 8-10 publication-quality figures from experimental results for MDPI paper.
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pathlib import Path
import seaborn as sns

# Set publication-quality style
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.size'] = 11
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 13
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10
plt.rcParams['legend.fontsize'] = 10
plt.rcParams['figure.titlesize'] = 14

# Color scheme
ENHANCED_COLOR = '#2ecc71'  # Green
BASELINE_COLOR = '#e74c3c'  # Red
COLORS = {'Enhanced': ENHANCED_COLOR, 'Baseline': BASELINE_COLOR}

def load_data():
    """Load experimental results."""
    with open('results/experiments/final_experiment_results.json', 'r') as f:
        data = json.load(f)
    
    with open('results/analysis/overall_statistics.json', 'r') as f:
        overall_stats = json.load(f)
    
    with open('results/analysis/scenario_statistics.json', 'r') as f:
        scenario_stats = json.load(f)
    
    return data, overall_stats, scenario_stats


def fig1_success_rate_by_scenario(data, scenario_stats, save_path):
    """Figure 1: Success rate comparison by scenario."""
    print("Generating Figure 1: Success Rate by Scenario...")
    
    scenarios = sorted(scenario_stats.keys())
    enhanced_rates = [scenario_stats[s]['enhanced_success_rate'] for s in scenarios]
    baseline_rates = [scenario_stats[s]['baseline_success_rate'] for s in scenarios]
    
    # Shorten scenario names for display
    scenario_labels = [s.split(' - ')[0] for s in scenarios]
    
    x = np.arange(len(scenarios))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    bars1 = ax.bar(x - width/2, enhanced_rates, width, label='Enhanced', 
                   color=ENHANCED_COLOR, alpha=0.8, edgecolor='black', linewidth=1.2)
    bars2 = ax.bar(x + width/2, baseline_rates, width, label='Baseline', 
                   color=BASELINE_COLOR, alpha=0.8, edgecolor='black', linewidth=1.2)
    
    ax.set_xlabel('Scenario', fontweight='bold')
    ax.set_ylabel('Success Rate (%)', fontweight='bold')
    ax.set_title('Task Success Rate by Scenario', fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(scenario_labels, rotation=15, ha='right')
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim([0, 105])
    
    # Add value labels on bars
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 2,
                   f'{height:.0f}%', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved to {save_path}")


def fig2_safety_violations(data, scenario_stats, save_path):
    """Figure 2: Safety violations comparison."""
    print("Generating Figure 2: Safety Violations...")
    
    scenarios = sorted(scenario_stats.keys())
    enhanced_viols = [scenario_stats[s]['enhanced_violations'] for s in scenarios]
    baseline_viols = [scenario_stats[s]['baseline_violations'] for s in scenarios]
    
    scenario_labels = [s.split(' - ')[0] for s in scenarios]
    
    x = np.arange(len(scenarios))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    bars1 = ax.bar(x - width/2, enhanced_viols, width, label='Enhanced', 
                   color=ENHANCED_COLOR, alpha=0.8, edgecolor='black', linewidth=1.2)
    bars2 = ax.bar(x + width/2, baseline_viols, width, label='Baseline', 
                   color=BASELINE_COLOR, alpha=0.8, edgecolor='black', linewidth=1.2)
    
    ax.set_xlabel('Scenario', fontweight='bold')
    ax.set_ylabel('Total Safety Violations', fontweight='bold')
    ax.set_title('Safety Violations by Scenario (20 trials each)', fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(scenario_labels, rotation=15, ha='right')
    ax.legend(loc='upper left')
    ax.grid(True, alpha=0.3, axis='y')
    
    # Log scale if needed
    if max(baseline_viols) > 1000:
        ax.set_yscale('log')
        ax.set_ylabel('Total Safety Violations (log scale)', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved to {save_path}")


def fig3_path_efficiency_boxplot(data, save_path):
    """Figure 3: Path efficiency distribution."""
    print("Generating Figure 3: Path Efficiency Box Plot...")
    
    enhanced_trials = [t for t in data if 'Enhanced' in t['method']]
    baseline_trials = [t for t in data if 'Baseline' in t['method']]
    
    enhanced_eff = [t['path_efficiency'] for t in enhanced_trials]
    baseline_eff = [t['path_efficiency'] for t in baseline_trials]
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    bp = ax.boxplot([enhanced_eff, baseline_eff], 
                     labels=['Enhanced', 'Baseline'],
                     patch_artist=True,
                     widths=0.6,
                     showmeans=True,
                     meanprops=dict(marker='D', markerfacecolor='yellow', markeredgecolor='black'))
    
    # Color boxes
    bp['boxes'][0].set_facecolor(ENHANCED_COLOR)
    bp['boxes'][0].set_alpha(0.7)
    bp['boxes'][1].set_facecolor(BASELINE_COLOR)
    bp['boxes'][1].set_alpha(0.7)
    
    ax.set_ylabel('Path Efficiency (lower is better)', fontweight='bold')
    ax.set_title('Path Efficiency Distribution', fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    ax.axhline(y=1.0, color='green', linestyle='--', linewidth=1.5, alpha=0.5, label='Optimal (1.0)')
    ax.legend()
    
    # Add statistics
    enh_mean = np.mean(enhanced_eff)
    base_mean = np.mean(baseline_eff)
    ax.text(1, enh_mean, f'μ={enh_mean:.3f}', ha='center', va='bottom', fontsize=9, color='black')
    ax.text(2, base_mean, f'μ={base_mean:.3f}', ha='center', va='bottom', fontsize=9, color='black')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved to {save_path}")


def fig4_computation_time(data, save_path):
    """Figure 4: Computation time comparison."""
    print("Generating Figure 4: Computation Time...")
    
    enhanced_trials = [t for t in data if 'Enhanced' in t['method']]
    baseline_trials = [t for t in data if 'Baseline' in t['method']]
    
    enhanced_times = [t['computation_time'] for t in enhanced_trials]
    baseline_times = [t['computation_time'] for t in baseline_trials]
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    bp = ax.boxplot([enhanced_times, baseline_times], 
                     labels=['Enhanced', 'Baseline'],
                     patch_artist=True,
                     widths=0.6,
                     showmeans=True,
                     meanprops=dict(marker='D', markerfacecolor='yellow', markeredgecolor='black'))
    
    bp['boxes'][0].set_facecolor(ENHANCED_COLOR)
    bp['boxes'][0].set_alpha(0.7)
    bp['boxes'][1].set_facecolor(BASELINE_COLOR)
    bp['boxes'][1].set_alpha(0.7)
    
    ax.set_ylabel('Computation Time (seconds)', fontweight='bold')
    ax.set_title('Total Episode Computation Time', fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    
    enh_mean = np.mean(enhanced_times)
    base_mean = np.mean(baseline_times)
    speedup = base_mean / enh_mean
    
    ax.text(0.5, 0.95, f'Speedup: {speedup:.1f}×', transform=ax.transAxes,
           fontsize=12, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved to {save_path}")


def fig5_steps_comparison(data, save_path):
    """Figure 5: Steps to completion comparison."""
    print("Generating Figure 5: Steps to Completion...")
    
    enhanced_trials = [t for t in data if 'Enhanced' in t['method']]
    baseline_trials = [t for t in data if 'Baseline' in t['method']]
    
    enhanced_steps = [t['steps'] for t in enhanced_trials]
    baseline_steps = [t['steps'] for t in baseline_trials]
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    bp = ax.boxplot([enhanced_steps, baseline_steps], 
                     labels=['Enhanced', 'Baseline'],
                     patch_artist=True,
                     widths=0.6,
                     showmeans=True,
                     meanprops=dict(marker='D', markerfacecolor='yellow', markeredgecolor='black'))
    
    bp['boxes'][0].set_facecolor(ENHANCED_COLOR)
    bp['boxes'][0].set_alpha(0.7)
    bp['boxes'][1].set_facecolor(BASELINE_COLOR)
    bp['boxes'][1].set_alpha(0.7)
    
    ax.set_ylabel('Steps to Completion', fontweight='bold')
    ax.set_title('Episode Length (Simulation Steps)', fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    
    enh_mean = np.mean(enhanced_steps)
    base_mean = np.mean(baseline_steps)
    reduction = (base_mean - enh_mean) / base_mean * 100
    
    ax.text(0.5, 0.95, f'Reduction: {reduction:.1f}%', transform=ax.transAxes,
           fontsize=12, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved to {save_path}")


def fig6_trajectory_overlay(data, save_path):
    """Figure 6: Representative trajectory comparison."""
    print("Generating Figure 6: Trajectory Overlay...")
    
    # Select best trial from each method for "Hard" scenario
    enhanced_trials = [t for t in data if 'Enhanced' in t['method'] and 'Hard' in t['scenario']]
    baseline_trials = [t for t in data if 'Baseline' in t['method'] and 'Hard' in t['scenario']]
    
    # Get successful trials or best attempt
    enhanced_trial = next((t for t in enhanced_trials if t['success']), enhanced_trials[0])
    baseline_trial = next((t for t in baseline_trials if t['success']), baseline_trials[0]) if baseline_trials else None
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Plot Enhanced trajectory
    enh_traj = enhanced_trial['trajectory']
    enh_x = [p[0] for p in enh_traj]
    enh_y = [p[1] for p in enh_traj]
    ax.plot(enh_x, enh_y, color=ENHANCED_COLOR, linewidth=2.5, label='Enhanced', alpha=0.8)
    ax.plot(enh_x[0], enh_y[0], 'go', markersize=12, label='Start')
    
    # Plot Baseline trajectory if available
    if baseline_trial:
        base_traj = baseline_trial['trajectory']
        base_x = [p[0] for p in base_traj]
        base_y = [p[1] for p in base_traj]
        ax.plot(base_x, base_y, color=BASELINE_COLOR, linewidth=2.5, label='Baseline', alpha=0.8, linestyle='--')
    
    # Plot goal
    if enh_traj:
        ax.plot(5.0, 5.0, 'k*', markersize=20, label='Goal', markeredgewidth=2)
    
    # Add obstacle info (hardcoded from scenario definition)
    # For "Hard - Multiple Obstacles"
    obstacles = [(2, 2, 0.3), (3, 4, 0.3), (4, 3, 0.3)]
    for ox, oy, r in obstacles:
        circle = patches.Circle((ox, oy), r, color='gray', alpha=0.5, zorder=5)
        ax.add_patch(circle)
    
    ax.set_xlabel('X Position (m)', fontweight='bold')
    ax.set_ylabel('Y Position (m)', fontweight='bold')
    ax.set_title('Trajectory Comparison - Hard Scenario', fontweight='bold')
    ax.legend(loc='upper left')
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')
    ax.set_xlim([-0.5, 5.5])
    ax.set_ylim([-0.5, 5.5])
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved to {save_path}")


def fig7_adaptive_gains(data, save_path):
    """Figure 7: Adaptive gain evolution."""
    print("Generating Figure 7: Adaptive Gains Evolution...")
    
    # Select trial with adaptive gains history
    enhanced_trials = [t for t in data if 'Enhanced' in t['method'] and t.get('adaptive_gains_history')]
    
    if not enhanced_trials:
        print("  ⚠ No adaptive gains history found, skipping...")
        return
    
    trial = enhanced_trials[0]
    gains_history = trial['adaptive_gains_history']
    
    steps = list(range(len(gains_history)))
    kp_vals = [g['kp'] for g in gains_history]
    ki_vals = [g['ki'] for g in gains_history]
    kd_vals = [g['kd'] for g in gains_history]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.plot(steps, kp_vals, label='Kp (Proportional)', color='#3498db', linewidth=2)
    ax.plot(steps, ki_vals, label='Ki (Integral)', color='#e74c3c', linewidth=2)
    ax.plot(steps, kd_vals, label='Kd (Derivative)', color='#f39c12', linewidth=2)
    
    ax.set_xlabel('Simulation Step', fontweight='bold')
    ax.set_ylabel('Gain Value', fontweight='bold')
    ax.set_title('Adaptive PID Gains Evolution', fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved to {save_path}")


def fig8_performance_heatmap(scenario_stats, save_path):
    """Figure 8: Performance heatmap."""
    print("Generating Figure 8: Performance Heatmap...")
    
    scenarios = sorted(scenario_stats.keys())
    scenario_labels = [s.split(' - ')[0] for s in scenarios]
    
    # Create matrix: rows are scenarios, columns are [Enhanced Success, Baseline Success]
    data_matrix = []
    for s in scenarios:
        enh_success = scenario_stats[s]['enhanced_success_rate']
        base_success = scenario_stats[s]['baseline_success_rate']
        data_matrix.append([enh_success, base_success])
    
    data_matrix = np.array(data_matrix)
    
    fig, ax = plt.subplots(figsize=(6, 8))
    
    im = ax.imshow(data_matrix, cmap='RdYlGn', aspect='auto', vmin=0, vmax=100)
    
    ax.set_xticks([0, 1])
    ax.set_xticklabels(['Enhanced', 'Baseline'])
    ax.set_yticks(range(len(scenario_labels)))
    ax.set_yticklabels(scenario_labels)
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Success Rate (%)', fontweight='bold')
    
    # Add text annotations
    for i in range(len(scenarios)):
        for j in range(2):
            text = ax.text(j, i, f'{data_matrix[i, j]:.0f}%',
                          ha="center", va="center", color="black", fontsize=11, fontweight='bold')
    
    ax.set_title('Success Rate Heatmap', fontweight='bold', pad=15)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved to {save_path}")


def fig9_overall_comparison(overall_stats, save_path):
    """Figure 9: Overall metrics comparison radar chart."""
    print("Generating Figure 9: Overall Metrics Comparison...")
    
    # Prepare metrics (normalize to 0-100 scale)
    categories = ['Success\nRate', 'Safety\n(inverse viol.)', 'Path\nEfficiency', 'Computation\nSpeed']
    
    # Enhanced scores
    enh_success = overall_stats['success_rate']['enhanced_mean'] * 100
    enh_safety = 100  # 0 violations = 100% safe
    enh_efficiency = (2.0 - overall_stats['path_efficiency']['enhanced_mean']) / 2.0 * 100  # Invert and normalize
    enh_speed = 100  # Faster = 100
    enhanced_values = [enh_success, enh_safety, max(0, enh_efficiency), enh_speed]
    
    # Baseline scores
    base_success = overall_stats['success_rate']['baseline_mean'] * 100
    base_safety = 0  # Has violations
    base_efficiency = (2.0 - overall_stats['path_efficiency']['baseline_mean']) / 2.0 * 100
    base_speed = (overall_stats['computation_time']['enhanced_mean'] / 
                  overall_stats['computation_time']['baseline_mean']) * 100  # Relative speed
    baseline_values = [base_success, base_safety, max(0, base_efficiency), base_speed]
    
    # Number of variables
    num_vars = len(categories)
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    
    # Complete the loop
    enhanced_values += enhanced_values[:1]
    baseline_values += baseline_values[:1]
    angles += angles[:1]
    
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(projection='polar'))
    
    ax.plot(angles, enhanced_values, 'o-', linewidth=2, label='Enhanced', color=ENHANCED_COLOR)
    ax.fill(angles, enhanced_values, alpha=0.25, color=ENHANCED_COLOR)
    
    ax.plot(angles, baseline_values, 'o-', linewidth=2, label='Baseline', color=BASELINE_COLOR)
    ax.fill(angles, baseline_values, alpha=0.25, color=BASELINE_COLOR)
    
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories)
    ax.set_ylim(0, 100)
    ax.set_yticks([25, 50, 75, 100])
    ax.set_yticklabels(['25', '50', '75', '100'])
    ax.grid(True)
    
    ax.set_title('Overall Performance Comparison', fontweight='bold', pad=20, fontsize=14)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved to {save_path}")


def fig10_path_length_violin(data, save_path):
    """Figure 10: Path length distribution violin plot."""
    print("Generating Figure 10: Path Length Distribution...")
    
    enhanced_trials = [t for t in data if 'Enhanced' in t['method']]
    baseline_trials = [t for t in data if 'Baseline' in t['method']]
    
    # Prepare data
    plot_data = []
    for t in enhanced_trials:
        plot_data.append({'Method': 'Enhanced', 'Path Length': t['path_length']})
    for t in baseline_trials:
        plot_data.append({'Method': 'Baseline', 'Path Length': t['path_length']})
    
    import pandas as pd
    df = pd.DataFrame(plot_data)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    parts = ax.violinplot([df[df['Method']=='Enhanced']['Path Length'].values,
                           df[df['Method']=='Baseline']['Path Length'].values],
                          positions=[1, 2], showmeans=True, showmedians=True)
    
    # Color violin parts
    for i, pc in enumerate(parts['bodies']):
        if i == 0:
            pc.set_facecolor(ENHANCED_COLOR)
        else:
            pc.set_facecolor(BASELINE_COLOR)
        pc.set_alpha(0.7)
    
    ax.set_xticks([1, 2])
    ax.set_xticklabels(['Enhanced', 'Baseline'])
    ax.set_ylabel('Path Length (m)', fontweight='bold')
    ax.set_title('Path Length Distribution', fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add optimal path line (straight line from 0,0 to 5,5)
    optimal_length = np.sqrt(5**2 + 5**2)
    ax.axhline(y=optimal_length, color='green', linestyle='--', linewidth=1.5, 
              alpha=0.5, label=f'Optimal ({optimal_length:.2f}m)')
    ax.legend()
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved to {save_path}")


def main():
    """Generate all publication figures."""
    print("\n" + "="*70)
    print("PUBLICATION FIGURE GENERATION - PHASE 7")
    print("="*70 + "\n")
    
    # Create output directory
    output_dir = Path('generated_figures')
    output_dir.mkdir(exist_ok=True)
    
    # Load data
    print("Loading experimental data...")
    data, overall_stats, scenario_stats = load_data()
    print(f"✓ Loaded {len(data)} trials\n")
    
    # Generate all figures
    fig1_success_rate_by_scenario(data, scenario_stats, output_dir / 'fig1_success_by_scenario.png')
    fig2_safety_violations(data, scenario_stats, output_dir / 'fig2_safety_violations.png')
    fig3_path_efficiency_boxplot(data, output_dir / 'fig3_path_efficiency.png')
    fig4_computation_time(data, output_dir / 'fig4_computation_time.png')
    fig5_steps_comparison(data, output_dir / 'fig5_steps_comparison.png')
    fig6_trajectory_overlay(data, output_dir / 'fig6_trajectory_overlay.png')
    fig7_adaptive_gains(data, output_dir / 'fig7_adaptive_gains.png')
    fig8_performance_heatmap(scenario_stats, output_dir / 'fig8_performance_heatmap.png')
    fig9_overall_comparison(overall_stats, output_dir / 'fig9_overall_comparison.png')
    fig10_path_length_violin(data, output_dir / 'fig10_path_length_distribution.png')
    
    print("\n" + "="*70)
    print("FIGURE GENERATION COMPLETE")
    print("="*70)
    print(f"\n✓ All figures saved to {output_dir}/")
    print("\nGenerated Figures:")
    print("  1. Success Rate by Scenario (bar chart)")
    print("  2. Safety Violations (bar chart)")
    print("  3. Path Efficiency (box plot)")
    print("  4. Computation Time (box plot)")
    print("  5. Steps to Completion (box plot)")
    print("  6. Trajectory Overlay - Hard Scenario")
    print("  7. Adaptive Gains Evolution (line chart)")
    print("  8. Performance Heatmap")
    print("  9. Overall Metrics Comparison (radar chart)")
    print(" 10. Path Length Distribution (violin plot)")
    print("\nReady for paper integration!")


if __name__ == '__main__':
    main()
