"""
Comparative Experiment Runner

Runs systematic experiments comparing baseline vs enhanced controller
across multiple scenarios with statistical analysis.
"""
import numpy as np
import json
import time
from datetime import datetime
from pathlib import Path


class ExperimentRunner:
    """Runs comparative experiments and collects metrics."""
    
    def __init__(self, output_dir="results/experiments"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.results = []
        
    def run_single_trial(self, controller, scenario, trial_num, method_name):
        """Run a single trial and collect metrics."""
        print(f"\n{'='*70}")
        print(f"Method: {method_name} | Scenario: {scenario['name']} | Trial: {trial_num}")
        print(f"{'='*70}")
        
        start_time = time.time()
        
        # Run episode
        stats = controller.run_episode(
            goal_position=scenario['goal'],
            max_steps=5000
        )
        
        computation_time = time.time() - start_time
        
        # Collect metrics
        result = {
            'method': method_name,
            'scenario': scenario['name'],
            'trial': trial_num,
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            
            # Performance metrics
            'success': stats['success'],
            'final_distance': stats['final_distance'],
            'steps': stats['steps'],
            'path_length': stats['path_length'],
            'computation_time': computation_time,
            
            # Efficiency metrics
            'direct_distance': np.linalg.norm(
                np.array(scenario['goal'][:2]) - np.array(scenario['start'][:2])
            ),
            'path_efficiency': 0.0,  # Will compute below
            
            # Safety metrics (from controller if available)
            'safety_violations': stats.get('safety_violations', 0),
            'min_obstacle_distance': stats.get('min_obstacle_distance', float('inf')),
            
            # Enhanced controller metrics (for multimodal analysis)
            'adaptive_gains_history': stats.get('adaptive_gains_history', []),
            'sensor_data_samples': stats.get('sensor_data_samples', []),
            'llm_interventions': stats.get('llm_interventions', 0),
            'llm_confidence': stats.get('llm_confidence', 0.0),
            
            # Trajectory for detailed analysis
            'trajectory': stats.get('trajectory', [])
        }
        
        # Compute path efficiency (closer to 1.0 is better)
        if result['path_length'] > 0:
            result['path_efficiency'] = result['direct_distance'] / result['path_length']
        
        self.results.append(result)
        
        print(f"\nTrial {trial_num} Results:")
        print(f"  Success: {result['success']}")
        print(f"  Final Distance: {result['final_distance']:.3f}m")
        print(f"  Steps: {result['steps']}")
        print(f"  Path Length: {result['path_length']:.2f}m")
        print(f"  Path Efficiency: {result['path_efficiency']:.3f}")
        print(f"  Computation Time: {computation_time:.2f}s")
        
        return result
    
    def run_scenario_comparison(self, baseline_controller, enhanced_controller, 
                               scenario, num_trials=20):
        """
        Run comparison experiments for a single scenario.
        
        Args:
            baseline_controller: BaselineController instance
            enhanced_controller: EmbodiedSimulation instance
            scenario: Scenario dictionary
            num_trials: Number of trials to run for each method
        """
        print(f"\n{'#'*70}")
        print(f"# SCENARIO: {scenario['name']}")
        print(f"# Running {num_trials} trials for each method (Total: {num_trials*2} trials)")
        print(f"{'#'*70}\n")
        
        # Run baseline trials
        print(f"\n{'*'*70}")
        print(f"* BASELINE METHOD")
        print(f"{'*'*70}")
        
        for trial in range(1, num_trials + 1):
            # Reset robot position
            # (This would need to be implemented in the controllers)
            self.run_single_trial(
                baseline_controller, 
                scenario, 
                trial, 
                "Baseline (PID only)"
            )
        
        # Run enhanced trials
        print(f"\n{'*'*70}")
        print(f"* ENHANCED METHOD")
        print(f"{'*'*70}")
        
        for trial in range(1, num_trials + 1):
            # Reset robot position
            self.run_single_trial(
                enhanced_controller,
                scenario,
                trial,
                "Enhanced (Sensors+LLM+Adaptive+Safety)"
            )
    
    def save_results(self, filename=None):
        """Save results to JSON file."""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"experiment_results_{timestamp}.json"
        
        filepath = self.output_dir / filename
        
        # Convert any numpy types to native Python types
        def convert_numpy(obj):
            """Recursively convert numpy types to Python types."""
            import numpy as np
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, (np.integer, np.floating)):
                return float(obj)
            elif isinstance(obj, np.bool_):
                return bool(obj)
            elif isinstance(obj, dict):
                return {key: convert_numpy(value) for key, value in obj.items()}
            elif isinstance(obj, (list, tuple)):
                return [convert_numpy(item) for item in obj]
            else:
                return obj
        
        clean_results = convert_numpy(self.results)
        
        with open(filepath, 'w') as f:
            json.dump(clean_results, f, indent=2)
        
        print(f"\n[OK] Results saved to: {filepath}")
        return filepath
    
    def print_summary(self):
        """Print summary statistics."""
        if not self.results:
            print("No results to summarize")
            return
        
        print(f"\n{'='*70}")
        print("EXPERIMENT SUMMARY")
        print(f"{'='*70}\n")
        
        # Group by method
        methods = {}
        for result in self.results:
            method = result['method']
            if method not in methods:
                methods[method] = []
            methods[method].append(result)
        
        # Print summary for each method
        for method_name, method_results in methods.items():
            print(f"\n{method_name}:")
            print(f"  Total Trials: {len(method_results)}")
            
            # Success rate
            successes = sum(1 for r in method_results if r['success'])
            success_rate = successes / len(method_results) * 100
            print(f"  Success Rate: {successes}/{len(method_results)} ({success_rate:.1f}%)")
            
            # Average metrics (only for successful trials)
            successful = [r for r in method_results if r['success']]
            if successful:
                avg_steps = np.mean([r['steps'] for r in successful])
                avg_path = np.mean([r['path_length'] for r in successful])
                avg_efficiency = np.mean([r['path_efficiency'] for r in successful])
                avg_time = np.mean([r['computation_time'] for r in successful])
                
                print(f"  Avg Steps (successful): {avg_steps:.1f}")
                print(f"  Avg Path Length: {avg_path:.2f}m")
                print(f"  Avg Path Efficiency: {avg_efficiency:.3f}")
                print(f"  Avg Computation Time: {avg_time:.2f}s")
            
            # Safety violations (only for enhanced method)
            if any('safety_violations' in r for r in method_results):
                total_violations = sum(r.get('safety_violations', 0) for r in method_results)
                print(f"  Total Safety Violations: {total_violations}")
        
        print(f"\n{'='*70}\n")
    
    def generate_comparison_table(self):
        """Generate comparison table data."""
        if not self.results:
            return None
        
        # Group by method and scenario
        comparison = {}
        
        for result in self.results:
            method = result['method']
            scenario = result['scenario']
            
            key = (method, scenario)
            if key not in comparison:
                comparison[key] = []
            comparison[key].append(result)
        
        # Compute statistics
        table_data = []
        for (method, scenario), results in comparison.items():
            successes = sum(1 for r in results if r['success'])
            success_rate = successes / len(results) * 100
            
            successful = [r for r in results if r['success']]
            if successful:
                avg_steps = np.mean([r['steps'] for r in successful])
                avg_path = np.mean([r['path_length'] for r in successful])
                std_path = np.std([r['path_length'] for r in successful])
                # Only calculate violations for enhanced method (baseline doesn't have this field)
                avg_violations = np.mean([r.get('safety_violations', 0) for r in successful])
            else:
                avg_steps = 0
                avg_path = 0
                std_path = 0
                avg_violations = 0
            
            table_data.append({
                'method': method,
                'scenario': scenario,
                'trials': len(results),
                'success_rate': success_rate,
                'avg_steps': avg_steps,
                'avg_path_length': avg_path,
                'std_path_length': std_path,
                'avg_violations': avg_violations,
            })
        
        return table_data


if __name__ == "__main__":
    # Example usage
    runner = ExperimentRunner()
    print("Experiment Runner initialized")
    print(f"Output directory: {runner.output_dir}")
