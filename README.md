QFTT Bootstrap Loop Simulation Package

A peer-review-grade, fully reproducible simulation package for the Quantum Field Theory of Time (QFTT) Bootstrap Loop using a minimal 3-qubit model (Clock, System, Environment).
Overview
This repository provides a complete simulation framework demonstrating how an internal quantum "clock" can produce an emergent arrow of time via spontaneous collapse events, even though the total system is globally static (no external time). The simulation tracks 1000+ collapse events and analyzes the resulting entropy growth and temporal dynamics.
Key Features

3-Qubit Quantum Simulation: Clock-System-Environment model with configurable couplings
Spontaneous Collapse Mechanism: Time emerges through quantum state reduction when clock uncertainty exceeds threshold
Comprehensive Analysis Tools: Entropy tracking, interval distributions, parameter sweeps
Full Reproducibility: Fixed random seeds, Docker support, automated tests
Publication Ready: Generate all figures and statistics for papers

Quick Start
Installation
bash# Clone the repository
git clone https://github.com/LukeGr1ndhouse/qftt-bootstrap-loop-simulation.git
cd qftt-bootstrap-loop-simulation

# Install dependencies
pip install -r requirements.txt

# Install package
pip install -e .
Run a Basic Simulation
pythonfrom src.qftt_simulator import QFTTSimulator

# Create simulator with default parameters
sim = QFTTSimulator(max_events=1000)

# Run simulation
df = sim.run(random_seed=42)

# Results are saved to qftt_events.csv and qftt_events.h5
print(f"Recorded {len(df)} collapse events")
print(f"Final entropy: {df.iloc[-1]['entropy']:.3f} bits")
Or use the command line:
bashpython scripts/run_simulation.py --max_events 1000 --g 0.2 --threshold 0.5
Analyze Results
pythonfrom src import qftt_analysis as qa

# Load and visualize results
df = qa.load_events("qftt_events.csv")
qa.plot_entropy_vs_event(df)
qa.plot_interval_histogram(df)
Repository Structure
qftt-bootstrap-loop-simulation/
├── src/                          # Core simulation code
│   ├── qftt_simulator.py         # Main simulation engine
│   ├── qftt_analysis.py          # Analysis and visualization
│   └── qftt_validation.py        # Physical invariant checks
├── notebooks/                    # Jupyter notebooks
│   ├── 01_basic_simulation.ipynb
│   ├── 02_analysis_visualization.ipynb
│   └── 03_parameter_exploration.ipynb
├── tests/                        # Unit tests
├── scripts/                      # Command-line tools
├── docs/                         # Documentation
├── data/                         # Sample outputs
└── figures/                      # Generated figures
Theoretical Background
The QFTT Bootstrap Loop implements a Wheeler-DeWitt type universe where:

Initial State: Clock and system in entangled superposition with total energy ≈ 0
Time Emergence: Clock's time uncertainty triggers spontaneous collapses
Entropy Growth: Each collapse increases system entropy, creating an arrow of time
No External Time: Evolution uses internal consistency, not external time parameter

See docs/theory_overview.md for complete theoretical details.
Key Results
Running the default simulation produces:

~1000 collapse events with exponentially distributed intervals
Monotonic entropy growth from 0 to ~1 bit (maximum for 2-level system)
Balanced eigentime outcomes (+1/-1) demonstrating unbiased clock ticks
Parameter phase diagram showing stable vs. no-collapse regimes

Docker Support
Run simulations in a consistent environment:
bash# Build container
docker build -t qftt-simulation .

# Run simulation
docker run -v $(pwd)/data:/app/data qftt-simulation

# Or run notebooks
docker run -p 8888:8888 qftt-simulation jupyter notebook --ip=0.0.0.0
Testing
Comprehensive test suite ensures physical correctness:
bash# Run all tests
pytest

# With coverage
pytest --cov=src --cov-report=html

# Run specific validation
python -m pytest tests/test_validation.py -v
Citation
If you use this package in your research, please cite:
bibtex@software{qftt_bootstrap_2024,
  author = {Luca Casagrande},
  title = {QFTT Bootstrap Loop Simulation Package},
  year = {2024},
  publisher = {GitHub},
  url = {https://github.com/LukeGr1ndhouse/qftt-bootstrap-loop-simulation}
}
License
This project is licensed under the MIT License - see the LICENSE file for details.
Contributing
Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.
Acknowledgments
This work implements the theoretical framework of the Quantum Field Theory of Time (QFTT) Bootstrap Loop.

For questions or issues, please open a GitHub issue.
