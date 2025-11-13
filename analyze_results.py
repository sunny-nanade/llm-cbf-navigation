"""
Statistical Analysis of Experimental Results
Analyzes 200 trials (100 Enhanced + 100 Baseline) across 5 scenarios
Computes means, standard deviations, t-tests, and effect sizes for paper
"""

import json
import numpy as np
from scipy import stats
from pathlib import Path


def load_experimental_data(filepath='results/experiments/final_experiment_results.json'):
    """Load experimental results from JSON file."""
    print(f"\n{'='*70}")
    print(f"Loading Experimental Data")
    print(f"{'='*70}")
    
    with open(filepath, 'r') as f:
        data = json.load(f)
    
    print(f"✓ Loaded {len(data)} trials")
    
    # Split by method (use substring match to handle full method names)
    enhanced = [t for t in data if 'Enhanced' in t['method']]
    baseline = [t for t in data if 'Baseline' in t['method']]
    
    print(f"  • Enhanced: {len(enhanced)} trials")
    print(f"  • Baseline: {len(baseline)} trials")
    
    return data, enhanced, baseline


def compute_summary_statistics(enhanced_trials, baseline_trials):
    """Compute summary statistics for all metrics."""
    print(f"\n{'='*70}")
    print(f"Computing Summary Statistics")
    print(f"{'='*70}\n")
    
    metrics = {
        'success_rate': lambda t: t['success'],
        'safety_violations': lambda t: t['safety_violations'],
        'path_length': lambda t: t['path_length'],
        'path_efficiency': lambda t: t['path_efficiency'],
        'min_obstacle_distance': lambda t: t['min_obstacle_distance'],
        'steps': lambda t: t['steps'],
        'computation_time': lambda t: t['computation_time']
    }
    
    results = {}
    
    for metric_name, metric_fn in metrics.items():
        enhanced_values = [metric_fn(t) for t in enhanced_trials]
        baseline_values = [metric_fn(t) for t in baseline_trials]
        
        # Compute statistics
        enhanced_mean = np.mean(enhanced_values)
        enhanced_std = np.std(enhanced_values, ddof=1)
        baseline_mean = np.mean(baseline_values)
        baseline_std = np.std(baseline_values, ddof=1)
        
        # Perform t-test
        if metric_name == 'success_rate':
            # Use chi-square test for binary success/failure
            enhanced_successes = sum(enhanced_values)
            baseline_successes = sum(baseline_values)
            enhanced_failures = len(enhanced_values) - enhanced_successes
            baseline_failures = len(baseline_values) - baseline_successes
            
            # Contingency table
            contingency = np.array([
                [enhanced_successes, enhanced_failures],
                [baseline_successes, baseline_failures]
            ])
            chi2, p_value = stats.chi2_contingency(contingency)[:2]
            t_stat = None
        else:
            t_stat, p_value = stats.ttest_ind(enhanced_values, baseline_values)
        
        # Compute Cohen's d (effect size)
        pooled_std = np.sqrt(((len(enhanced_values)-1)*enhanced_std**2 + 
                             (len(baseline_values)-1)*baseline_std**2) / 
                            (len(enhanced_values) + len(baseline_values) - 2))
        cohens_d = (enhanced_mean - baseline_mean) / pooled_std if pooled_std > 0 else 0
        
        results[metric_name] = {
            'enhanced_mean': enhanced_mean,
            'enhanced_std': enhanced_std,
            'baseline_mean': baseline_mean,
            'baseline_std': baseline_std,
            't_statistic': t_stat,
            'p_value': p_value,
            'cohens_d': cohens_d,
            'enhanced_values': enhanced_values,
            'baseline_values': baseline_values
        }
        
        # Print formatted results
        print(f"{metric_name.replace('_', ' ').title()}:")
        print(f"  Enhanced: {enhanced_mean:.4f} ± {enhanced_std:.4f}")
        print(f"  Baseline: {baseline_mean:.4f} ± {baseline_std:.4f}")
        print(f"  Difference: {enhanced_mean - baseline_mean:+.4f}")
        print(f"  p-value: {p_value:.6f} {'***' if p_value < 0.001 else '**' if p_value < 0.01 else '*' if p_value < 0.05 else 'ns'}")
        print(f"  Cohen's d: {cohens_d:.4f} ({'large' if abs(cohens_d) > 0.8 else 'medium' if abs(cohens_d) > 0.5 else 'small'})")
        print()
    
    return results


def compute_per_scenario_statistics(all_trials):
    """Compute statistics broken down by scenario."""
    print(f"\n{'='*70}")
    print(f"Per-Scenario Analysis")
    print(f"{'='*70}\n")
    
    # Get actual scenario names from data
    scenarios = sorted(set(t['scenario'] for t in all_trials))
    scenario_results = {}
    
    for scenario in scenarios:
        scenario_data = [t for t in all_trials if t['scenario'] == scenario]
        enhanced = [t for t in scenario_data if 'Enhanced' in t['method']]
        baseline = [t for t in scenario_data if 'Baseline' in t['method']]
        
        enhanced_success_rate = sum(t['success'] for t in enhanced) / len(enhanced) * 100
        baseline_success_rate = sum(t['success'] for t in baseline) / len(baseline) * 100
        
        enhanced_violations = sum(t['safety_violations'] for t in enhanced)
        baseline_violations = sum(t['safety_violations'] for t in baseline)
        
        scenario_results[scenario] = {
            'enhanced_success_rate': enhanced_success_rate,
            'baseline_success_rate': baseline_success_rate,
            'enhanced_violations': enhanced_violations,
            'baseline_violations': baseline_violations,
            'enhanced_trials': len(enhanced),
            'baseline_trials': len(baseline)
        }
        
        print(f"{scenario.capitalize()}:")
        print(f"  Trials: {len(enhanced)} Enhanced, {len(baseline)} Baseline")
        print(f"  Success Rate: {enhanced_success_rate:.1f}% (Enhanced) vs {baseline_success_rate:.1f}% (Baseline)")
        print(f"  Safety Violations: {enhanced_violations} (Enhanced) vs {baseline_violations} (Baseline)")
        print()
    
    return scenario_results


def save_analysis_results(overall_stats, scenario_stats, output_dir='results/analysis'):
    """Save analysis results to JSON for later use."""
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Prepare data for JSON (remove numpy arrays)
    overall_export = {}
    for metric, data in overall_stats.items():
        overall_export[metric] = {
            'enhanced_mean': float(data['enhanced_mean']),
            'enhanced_std': float(data['enhanced_std']),
            'baseline_mean': float(data['baseline_mean']),
            'baseline_std': float(data['baseline_std']),
            't_statistic': float(data['t_statistic']) if data['t_statistic'] is not None else None,
            'p_value': float(data['p_value']),
            'cohens_d': float(data['cohens_d'])
        }
    
    # Save overall statistics
    with open(f'{output_dir}/overall_statistics.json', 'w') as f:
        json.dump(overall_export, f, indent=2)
    
    # Save scenario statistics
    with open(f'{output_dir}/scenario_statistics.json', 'w') as f:
        json.dump(scenario_stats, f, indent=2)
    
    print(f"\n✓ Analysis results saved to {output_dir}/")
    print(f"  • overall_statistics.json")
    print(f"  • scenario_statistics.json")
    
    return overall_export, scenario_stats


def generate_latex_table(overall_stats, scenario_stats):
    """Generate LaTeX table for paper."""
    print(f"\n{'='*70}")
    print(f"LaTeX Table Generation")
    print(f"{'='*70}\n")
    
    # Overall metrics table
    latex = "\\begin{table}[H]\n"
    latex += "\\caption{Comparison of Enhanced vs Baseline Controller Performance}\n"
    latex += "\\label{tab:metrics_summary}\n"
    latex += "\\centering\n"
    latex += "\\begin{tabular}{lrrrc}\n"
    latex += "\\toprule\n"
    latex += "\\textbf{Metric} & \\textbf{Enhanced} & \\textbf{Baseline} & \\textbf{$p$-value} & \\textbf{Cohen's $d$} \\\\\n"
    latex += "\\midrule\n"
    
    # Success rate (as percentage)
    sr_enh = overall_stats['success_rate']['enhanced_mean'] * 100
    sr_base = overall_stats['success_rate']['baseline_mean'] * 100
    sr_p = overall_stats['success_rate']['p_value']
    sr_d = overall_stats['success_rate']['cohens_d']
    latex += f"Success Rate (\\%) & {sr_enh:.1f} & {sr_base:.1f} & {sr_p:.4f} & {sr_d:.2f} \\\\\n"
    
    # Safety violations (total)
    sv_enh = overall_stats['safety_violations']['enhanced_mean'] * 100  # Total across 100 trials
    sv_base = overall_stats['safety_violations']['baseline_mean'] * 100
    sv_p = overall_stats['safety_violations']['p_value']
    sv_d = overall_stats['safety_violations']['cohens_d']
    latex += f"Safety Violations (total) & {sv_enh:.0f} & {sv_base:.0f} & {sv_p:.4f} & {sv_d:.2f} \\\\\n"
    
    # Path efficiency
    pe_enh = overall_stats['path_efficiency']['enhanced_mean']
    pe_enh_std = overall_stats['path_efficiency']['enhanced_std']
    pe_base = overall_stats['path_efficiency']['baseline_mean']
    pe_base_std = overall_stats['path_efficiency']['baseline_std']
    pe_p = overall_stats['path_efficiency']['p_value']
    pe_d = overall_stats['path_efficiency']['cohens_d']
    latex += f"Path Efficiency & {pe_enh:.3f}$\\pm${pe_enh_std:.3f} & {pe_base:.3f}$\\pm${pe_base_std:.3f} & {pe_p:.4f} & {pe_d:.2f} \\\\\n"
    
    # Min obstacle distance
    mod_enh = overall_stats['min_obstacle_distance']['enhanced_mean']
    mod_enh_std = overall_stats['min_obstacle_distance']['enhanced_std']
    mod_base = overall_stats['min_obstacle_distance']['baseline_mean']
    mod_base_std = overall_stats['min_obstacle_distance']['baseline_std']
    mod_p = overall_stats['min_obstacle_distance']['p_value']
    mod_d = overall_stats['min_obstacle_distance']['cohens_d']
    latex += f"Min Obstacle Dist (m) & {mod_enh:.3f}$\\pm${mod_enh_std:.3f} & {mod_base:.3f}$\\pm${mod_base_std:.3f} & {mod_p:.4f} & {mod_d:.2f} \\\\\n"
    
    latex += "\\bottomrule\n"
    latex += "\\end{tabular}\n"
    latex += "\\end{table}\n"
    
    # Save to file
    Path('generated_figures').mkdir(exist_ok=True)
    with open('generated_figures/metrics_summary_real.tex', 'w') as f:
        f.write(latex)
    
    print("✓ LaTeX table generated: generated_figures/metrics_summary_real.tex")
    print("\nTable preview:")
    print(latex)
    
    return latex


def main():
    """Run complete statistical analysis."""
    print("\n" + "="*70)
    print(" STATISTICAL ANALYSIS - PHASE 7")
    print("="*70)
    
    # Load data
    all_trials, enhanced, baseline = load_experimental_data()
    
    # Overall statistics
    overall_stats = compute_summary_statistics(enhanced, baseline)
    
    # Per-scenario statistics
    scenario_stats = compute_per_scenario_statistics(all_trials)
    
    # Save results
    save_analysis_results(overall_stats, scenario_stats)
    
    # Generate LaTeX table
    generate_latex_table(overall_stats, scenario_stats)
    
    print(f"\n{'='*70}")
    print(" ANALYSIS COMPLETE")
    print(f"{'='*70}\n")
    
    return overall_stats, scenario_stats


if __name__ == "__main__":
    overall_stats, scenario_stats = main()
