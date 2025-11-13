# Large Language Model-Driven Multimodal Sensor Fusion and Adaptive Control for Safety-Critical Embodied Autonomous Robots

## Overview

This repository contains the complete implementation for our MDPI Sensors journal submission. The codebase demonstrates a hierarchical control architecture combining high-level planning with runtime assurance for safe robot navigation.

## System Architecture

The system implements the following components:

### Core Components (Validated in Experiments)
- **PID Controller** (`control.py`): Position tracking controller
- **Control Barrier Function (CBF) Safety Filter** (`control.py`): QP-based safety projection
- **Mock LLM Planner** (`control.py`): Deterministic waypoint generator for reproducible experiments
- **PyBullet Simulation** (`main.py`, `runner.py`): Physics-based validation environment

### Enhanced Components (Implemented, Available for Future Work)
- **Multimodal Sensor Suite** (`sensors.py`):
  - RGB-D Camera with noise modeling
  - 2D LiDAR with 72-360 rays
  - Tactile/contact sensors
  - Proprioceptive sensors (encoders, IMU)
  - Extended Kalman Filter for sensor fusion
  
- **LLM Integration** (`llm_planner.py`):
  - OpenAI GPT-4 API integration
  - Confidence gating mechanism
  - Fallback policy handling
  - Prompt engineering for safe planning
  
- **Adaptive Control** (`llm_planner.py`):
  - Online gain scheduling
  - Performance-based adaptation
  - Automatic PID tuning

- **Full Embodied System** (`embodied_sim.py`):
  - Integrated perception-planning-control pipeline
  - Configurable sensor modalities
  - Real-time safety filtering

## Repository Structure

```
simulation/
├── README.md                           # This file
├── DEVELOPMENT_LOG.md                  # Development history
├── control.py                          # Core control & CBF safety filter
├── sensors.py                          # Multimodal sensor implementations
├── llm_planner.py                      # LLM integration & adaptive control
├── main.py                             # Basic PyBullet environment
├── runner.py                           # Baseline experiment runner
├── analyze.py                          # Results analysis
├── embodied_sim.py                     # Enhanced full system
├── run_paper_experiments.py           # Complete experimental suite
├── demo_visual.py                      # Visual demonstration
├── test_embodied_system.py            # Component validation tests
├── run_visual_demo.py                 # Original working demo
└── generated_data/                     # Experimental results
    ├── baseline.json
    ├── safe.json
    ├── trajectory.png
    ├── safety.png
    └── metrics_summary_real.tex
```

## Validated Experimental Results

The paper presents results from the **baseline system** (`runner.py` + `control.py`), which has been thoroughly validated:

### Key Findings
- **Baseline Controller**: 100% collision rate (unsafe)
- **CBF Safety Filter**: 0% collision rate (safe)
- **Success**: Demonstrates effectiveness of runtime assurance layer

### Running the Validated Experiments

```bash
# Activate virtual environment
.\.venv\Scripts\Activate.ps1

# Run baseline vs. safety-filtered comparison
python runner.py

# Analyze and generate figures
python analyze.py

# Visual demonstration
python run_visual_demo.py
```

## Enhanced System (Available for Future Validation)

The enhanced components are fully implemented and can be tested:

```bash
# Test individual components
python test_embodied_system.py

# Visual demo of full system
python demo_visual.py

# Run full experimental suite (requires debugging)
python run_paper_experiments.py
```

## Dependencies

```bash
pip install pybullet numpy matplotlib cvxpy scipy
```

Optional for real LLM integration:
```bash
pip install openai
export OPENAI_API_KEY="your-api-key"
```

## Configuration

### Baseline Experiments (Validated)
- Simple point-mass robot model
- Direct PyBullet state access
- Deterministic mock LLM planner
- Fixed PID gains
- CBF-QP safety filter

### Enhanced System (Implemented)
- Multimodal sensor fusion with noise
- Real LLM API integration option
- Adaptive gain scheduling
- Advanced state estimation (EKF)

## Implementation Notes

### Design Decisions

1. **Mock LLM for Reproducibility**: The validated experiments use a deterministic waypoint generator rather than real LLM API calls to ensure reproducibility. The real LLM integration is implemented and available in `llm_planner.py`.

2. **Simplified State Estimation**: Baseline experiments use direct PyBullet state for deterministic validation. Enhanced sensor fusion with EKF is implemented in `sensors.py`.

3. **Fixed Control Gains**: Validated results use fixed PID gains. Adaptive control is implemented in `llm_planner.py` for future work.

### Why This Approach?

**Scientific Reproducibility**: By using deterministic components in the validated experiments, we ensure:
- Bit-exact reproducibility
- No dependency on external API availability
- Consistent results across runs
- Clear isolation of the safety filter's contribution

**Future Extensibility**: All enhanced components are implemented and ready for:
- Integration with real LLM services
- Physical robot deployment
- Advanced sensor configurations
- Real-world validation

## Reproducing Paper Results

To reproduce the exact results in the paper:

```bash
cd simulation
python runner.py          # Runs baseline vs. safe experiments
python analyze.py         # Generates figures and tables
```

Expected output:
- `baseline.json`: Raw data from unfiltered controller
- `safe.json`: Raw data from safety-filtered controller  
- `trajectory.png`: Trajectory comparison plot
- `safety.png`: Safety metrics visualization
- `metrics_summary_real.tex`: LaTeX table for paper

## Future Work

The enhanced components provide a foundation for:

1. **Real LLM Integration**: Replace mock planner with GPT-4 API calls
2. **Physical Robot Testing**: Deploy on hardware with real sensors
3. **Dynamic Obstacles**: Extend to moving hazards
4. **Multi-Robot Systems**: Scale to collaborative scenarios
5. **Articulated Models**: Replace point-mass with mobile manipulator

## Citation

If you use this code, please cite:

```bibtex
@article{nanade2025llm,
  title={Large Language Model-Driven Multimodal Sensor Fusion and Adaptive Control for Safety-Critical Embodied Autonomous Robots},
  author={Nanade, Sunny and Anne, Koteswara Rao},
  journal={Sensors},
  year={2025},
  publisher={MDPI}
}
```

## License

This code is released under the MIT License for academic and research use.

## Contact

For questions or collaboration:
- Sunny Nanade: sunny.nanade@nmims.edu
- Koteswara Rao Anne: Koteswararao.Anne@nmims.edu

## Acknowledgments

This work was supported by Mukesh Patel School of Technology Management & Engineering, SVKM's NMIMS, Mumbai, India.
