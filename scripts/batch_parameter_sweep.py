#!/usr/bin/env python
"""
Batch parameter sweep for QFTT simulations.

This script runs multiple simulations over a parameter grid and saves summary statistics.

© 2024 - MIT License
"""

import argparse
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from multiprocessing import Pool, cpu_count
import time
from itertools import product
import json

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.qftt_simulator import QFTTSimulator
from src.qftt_analysis import get_interval_stats, entropy_rate_analysis


def run_single_simulation(params):
    """
    Run a single simulation with given parameters.
    
    Returns summary statistics instead of full event data.
    """
    param_dict, seed = params
    
    try:
        # Create and run simulator
        sim = QFTTSimulator(**param_dict)
        df = sim.run(random_seed=seed)
        
        if len(df) == 0:
            # No events case
            return {
                **param_dict,
                'seed': seed,
                'n_events': 0,
                'simulation_time': sim.t,
                'avg_delta_t': np.nan,
                'std_delta_t': np.nan,
                'final_entropy': 0.0,
                'avg_entropy': 0.0,
                'entropy_rate': 0.0,
                'eigentime_balance': np.nan,
                'success': True,
                'error': None
            }
        
        # Compute statistics
        interval_stats = get_interval_stats(df)
        entropy_stats = entropy_rate_analysis(df)
        
        plus_count = (df['eigentime'] == 1).sum()
        eigentime_balance = plus_count / len(df)
        
        return {
            **param_dict,
            'seed': seed,
            'n_events': len(df),
            'simulation_time': df['delta_t'].sum(),
            'avg_delta_t': interval_stats['mean'],
            'std_delta_t': interval_stats['std'],
            'final_entropy': df.iloc[-1]['entropy'],
            'avg_entropy': df['entropy'].mean(),
            'entropy_rate': entropy_stats['avg_entropy_rate'],
            'entropy_per_event': entropy_stats['entropy_per_event'],
            'eigentime_balance': eigentime_balance,
            'is_exponential': interval_stats['is_exponential'],
            'success': True,
            'error': None
        }
        
    except Exception as e:
        # Return error info
        return {
            **param_dict,
            'seed': seed,
            'n_events': -1,
            'success': False,
            'error': str(e)
        }


def main():
    parser = argparse.ArgumentParser(
        description='Run parameter sweep for QFTT simulations',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Parameter ranges
    parser.add_argument('--g_values', type=str, default='0.1,0.2,0.4',
                       help='Comma-separated list of g values')
    parser.add_argument('--threshold_values', type=str, default='0.3,0.5,0.7',
                       help='Comma-separated list of threshold values')
    parser.add_argument('--lambda_se_values', type=str, default='0.1',
                       help='Comma-separated list of lambda_se values')
    
    # Fixed parameters
    parser.add_argument('--omega', type=float, default=1.0,
                       help='Fixed omega value')
    parser.add_argument('--dt', type=float, default=0.05,
                       help='Fixed dt value')
    parser.add_argument('--max_events', type=int, default=1000,
                       help='Maximum events per simulation')
    parser.add_argument('--t_max', type=float, default=200.0,
                       help='Maximum time per simulation')
    
    # Sweep options
    parser.add_argument('--n_seeds', type=int, default=3,
                       help='Number of random seeds per parameter combination')
    parser.add_argument('--n_workers', type=int, default=None,
                       help='Number of parallel workers (default: CPU count)')
    
    # Output options
    parser.add_argument('--output', type=str, default='parameter_sweep_results.csv',
                       help='Output filename')
    parser.add_argument('--output_dir', type=str, default='data/benchmarks',
                       help='Output directory')
    
    args = parser.parse_args()
    
    # Parse parameter values
    g_values = [float(x) for x in args.g_values.split(',')]
    threshold_values = [float(x) for x in args.threshold_values.split(',')]
    lambda_se_values = [float(x) for x in args.lambda_se_values.split(',')]
    
    # Create parameter grid
    param_grid = list(product(g_values, threshold_values, lambda_se_values))
    
    # Prepare simulation tasks
    tasks = []
    base_seed = 42
    
    for i, (g, threshold, lambda_se) in enumerate(param_grid):
        param_dict = {
            'Omega': args.omega,
            'g': g,
            'lambda_se': lambda_se,
            'dt': args.dt,
            'threshold': threshold,
            'max_events': args.max_events,
            't_max': args.t_max
        }
        
        # Add multiple seeds for each parameter combination
        for j in range(args.n_seeds):
            seed = base_seed + i * args.n_seeds + j
            tasks.append((param_dict, seed))
    
    print(f"=== QFTT Parameter Sweep ===")
    print(f"Parameter combinations: {len(param_grid)}")
    print(f"Seeds per combination: {args.n_seeds}")
    print(f"Total simulations: {len(tasks)}")
    print(f"Parameters:")
    print(f"  g values: {g_values}")
    print(f"  threshold values: {threshold_values}")
    print(f"  lambda_se values: {lambda_se_values}")
    print()
    
    # Run simulations in parallel
    n_workers = args.n_workers or cpu_count()
    print(f"Running with {n_workers} workers...")
    
    start_time = time.time()
    
    with Pool(n_workers) as pool:
        results = pool.map(run_single_simulation, tasks)
    
    elapsed_time = time.time() - start_time
    
    # Convert to DataFrame
    df_results = pd.DataFrame(results)
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save results
    output_path = output_dir / args.output
    df_results.to_csv(output_path, index=False)
    
    # Save metadata
    metadata = {
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'elapsed_time': elapsed_time,
        'n_simulations': len(tasks),
        'n_workers': n_workers,
        'parameter_ranges': {
            'g': g_values,
            'threshold': threshold_values,
            'lambda_se': lambda_se_values
        },
        'fixed_parameters': {
            'omega': args.omega,
            'dt': args.dt,
            'max_events': args.max_events,
            't_max': args.t_max
        }
    }
    
    metadata_path = output_path.with_suffix('.json')
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    # Print summary
    print(f"\nCompleted in {elapsed_time:.2f} seconds")
    print(f"Results saved to: {output_path}")
    print(f"Metadata saved to: {metadata_path}")
    
    # Print aggregate statistics
    successful = df_results[df_results['success']].copy()
    if len(successful) > 0:
        print("\nAggregate statistics:")
        
        # Group by parameters (excluding seed)
        grouped = successful.groupby(['g', 'threshold', 'lambda_se'])
        
        print("\nAverage events per parameter combination:")
        avg_events = grouped['n_events'].mean().sort_values(ascending=False)
        for params, n_events in avg_events.head(10).items():
            print(f"  g={params[0]:.2f}, threshold={params[1]:.2f}, λ_se={params[2]:.2f}: {n_events:.1f} events")
        
        # Identify failure modes
        no_events = successful[successful['n_events'] == 0]
        if len(no_events) > 0:
            print(f"\nParameter combinations with no events: {len(no_events)}")
            
    else:
        print("\nWarning: No successful simulations!")


if __name__ == "__main__":
    main()