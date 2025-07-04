#!/usr/bin/env python
"""
Command-line script to run QFTT simulation.

Usage:
    python run_simulation.py [options]

© 2024 - MIT License
"""

import argparse
import sys
from pathlib import Path
import time

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.qftt_simulator import QFTTSimulator


def main():
    parser = argparse.ArgumentParser(
        description='Run QFTT Bootstrap Loop simulation',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Simulation parameters
    parser.add_argument('--omega', type=float, default=1.0,
                       help='Energy splitting for clock and system qubits')
    parser.add_argument('--g', type=float, default=0.2,
                       help='Clock-system coupling strength')
    parser.add_argument('--lambda_se', type=float, default=0.1,
                       help='System-environment coupling strength')
    parser.add_argument('--dt', type=float, default=0.05,
                       help='Time step for evolution')
    parser.add_argument('--threshold', type=float, default=0.5,
                       help='Collapse threshold for time uncertainty (0-1)')
    parser.add_argument('--max_events', type=int, default=1000,
                       help='Maximum number of collapse events')
    parser.add_argument('--t_max', type=float, default=200.0,
                       help='Maximum simulation time')
    
    # Output options
    parser.add_argument('--output', type=str, default='qftt_events',
                       help='Output filename (without extension)')
    parser.add_argument('--output_dir', type=str, default='.',
                       help='Output directory')
    
    # Other options
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed (use -1 for random)')
    parser.add_argument('--quiet', action='store_true',
                       help='Suppress progress output')
    
    args = parser.parse_args()
    
    # Validate parameters
    if not 0 <= args.threshold <= 1:
        parser.error("threshold must be between 0 and 1")
    
    # Set random seed
    seed = None if args.seed == -1 else args.seed
    
    # Create output directory if needed
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Print configuration
    if not args.quiet:
        print("=== QFTT Bootstrap Loop Simulation ===")
        print(f"Parameters:")
        print(f"  Omega: {args.omega}")
        print(f"  g: {args.g}")
        print(f"  lambda_se: {args.lambda_se}")
        print(f"  dt: {args.dt}")
        print(f"  threshold: {args.threshold}")
        print(f"  max_events: {args.max_events}")
        print(f"  t_max: {args.t_max}")
        print(f"  seed: {seed if seed is not None else 'random'}")
        print()
    
    # Run simulation
    start_time = time.time()
    
    # Create simulator
    sim = QFTTSimulator(
        Omega=args.omega,
        g=args.g,
        lambda_se=args.lambda_se,
        dt=args.dt,
        threshold=args.threshold,
        max_events=args.max_events,
        t_max=args.t_max
    )
    
    # Run simulation
    if not args.quiet:
        print("Running simulation...")
    
    df = sim.run(random_seed=seed)
    
    elapsed_time = time.time() - start_time
    
    # Save with custom filename
    csv_path = output_dir / f"{args.output}.csv"
    h5_path = output_dir / f"{args.output}.h5"
    
    df.to_csv(csv_path, index=False)
    df.to_hdf(h5_path, key='events', mode='w')
    
    # Print summary
    if not args.quiet:
        print(f"\nSimulation completed in {elapsed_time:.2f} seconds")
        print(f"Events recorded: {len(df)}")
        if len(df) > 0:
            print(f"Average Δt: {df['delta_t'].mean():.3f} ± {df['delta_t'].std():.3f}")
            print(f"Final entropy: {df.iloc[-1]['entropy']:.3f} bits")
            print(f"Eigentime balance: +1: {(df['eigentime'] == 1).sum()}, -1: {(df['eigentime'] == -1).sum()}")
        print(f"\nResults saved to:")
        print(f"  {csv_path}")
        print(f"  {h5_path}")


if __name__ == "__main__":
    main()