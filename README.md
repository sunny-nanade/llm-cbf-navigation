# llm-cbf-navigation

This repository contains the complete implementation and experimental code to reproduce the results from the paper:

**"Large Language Model-Driven Multimodal Sensor Fusion and Adaptive Control for Safety-Critical Embodied Autonomous Robots"**

Published in MDPI Sensors, 2025.

## Structure

- `simulation/`: Contains the core simulation and control logic.
  - `main.py`: Main PyBullet simulation setup.
  - `control.py`: PID controller and CBF-QP Safety Filter implementation.
  - `runner.py`: Script to run the different experimental scenarios.
  - `analyze.py`: Script to analyze results and generate plots/tables.
- `simulation_results/`: Directory where raw JSON results from the runner are stored.
- `generated_figures/`: Directory where plots and LaTeX tables from the analysis script are saved.
- `run_simulation.py`: Top-level script to execute the entire pipeline.

## Requirements

The simulation requires Python 3.10+ and the following packages:
- `numpy==2.2.6`
- `pybullet==3.2.7`
- `cvxpy` (for QP solver)
- `matplotlib==3.10.7`
- `pandas==2.3.3`
- `scipy==1.14.1`

Install all dependencies using:
```bash
pip install -r requirements.txt
```

## How to Run

**Run the full simulation and analysis pipeline:**
```bash
python run_simulation.py
```

This will run the baseline and safety-filtered simulation scenarios, save results to `simulation_results/`, and generate figures and tables in `generated_figures/`.

## Data Availability

The raw experimental data and analysis scripts used in this study are available in this repository. The generated figures and tables can be fully reproduced by running the provided scripts.

## Citation

If you use this code in your research, please cite:

```bibtex
@article{llm-cbf-2025,
  title={Large Language Model-Driven Multimodal Sensor Fusion and Adaptive Control for Safety-Critical Embodied Autonomous Robots},
  author={Sunny Nanade and Koteswara Rao Anne},
  journal={MDPI Sensors},
  year={2025}
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contact

For questions or issues, please open an issue on GitHub.

---
