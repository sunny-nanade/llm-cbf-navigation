from simulation import runner, analyze

def main():
    """
    Runs the full simulation and analysis pipeline.
    """
    print("--- Starting Simulation Pipeline ---")
    
    # 1. Run the simulation scenarios
    runner.main()
    
    # 2. Analyze the results and generate figures/tables
    analyze.main()
    
    print("\n--- Pipeline Complete ---")
    print("Results have been generated in 'simulation_results/'")
    print("Figures and tables have been saved to 'generated_figures/'")

if __name__ == "__main__":
    main()
