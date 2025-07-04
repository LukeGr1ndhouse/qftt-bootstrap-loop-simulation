#!/usr/bin/env python
"""
Generate publication-quality figures from QFTT simulation data.

© 2024 - MIT License
"""

import argparse
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.qftt_simulator import QFTTSimulator
from src.qftt_analysis import (
    load_events, plot_entropy_vs_event, plot_interval_histogram,
    plot_qq_exponential, plot_time_series
)


# Set publication-quality defaults
plt.rcParams.update({
    'font.size': 12,
    'axes.labelsize': 14,
    'axes.titlesize': 16,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'legend.fontsize': 12,
    'figure.dpi': 100,
    'savefig.dpi': 300,
    'lines.linewidth': 2,
    'lines.markersize': 8,
    'figure.figsize': (10, 6),
    'figure.constrained_layout.use': True
})


def create_parameter_heatmap(sweep_results_path, output_dir):
    """Create heatmap from parameter sweep results."""
    df = pd.read_csv(sweep_results_path)
    
    # Average over seeds
    grouped = df.groupby(['g', 'threshold']).agg({
        'n_events': 'mean',
        'final_entropy': 'mean',
        'entropy_rate': 'mean'
    }).reset_index()
    
    # Create pivot tables
    events_pivot = grouped.pivot(index='threshold', columns='g', values='n_events')
    entropy_pivot = grouped.pivot(index='threshold', columns='g', values='final_entropy')
    rate_pivot = grouped.pivot(index='threshold', columns='g', values='entropy_rate')
    
    # Create figure with subplots
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # Events heatmap
    im1 = axes[0].imshow(events_pivot, cmap='YlOrRd', aspect='auto', origin='lower')
    axes[0].set_xlabel('Clock-System Coupling (g)')
    axes[0].set_ylabel('Collapse Threshold')
    axes[0].set_title('Average Number of Events')
    
    # Add colorbar
    cbar1 = plt.colorbar(im1, ax=axes[0])
    cbar1.set_label('Events')
    
    # Final entropy heatmap
    im2 = axes[1].imshow(entropy_pivot, cmap='viridis', aspect='auto', origin='lower')
    axes[1].set_xlabel('Clock-System Coupling (g)')
    axes[1].set_ylabel('Collapse Threshold')
    axes[1].set_title('Final System Entropy')
    
    cbar2 = plt.colorbar(im2, ax=axes[1])
    cbar2.set_label('Entropy (bits)')
    
    # Entropy rate heatmap
    im3 = axes[2].imshow(rate_pivot, cmap='plasma', aspect='auto', origin='lower')
    axes[2].set_xlabel('Clock-System Coupling (g)')
    axes[2].set_ylabel('Collapse Threshold')
    axes[2].set_title('Entropy Growth Rate')
    
    cbar3 = plt.colorbar(im3, ax=axes[2])
    cbar3.set_label('Bits/time')
    
    # Set tick labels
    for ax, pivot in zip(axes, [events_pivot, entropy_pivot, rate_pivot]):
        ax.set_xticks(range(len(pivot.columns)))
        ax.set_xticklabels([f'{x:.2f}' for x in pivot.columns])
        ax.set_yticks(range(len(pivot.index)))
        ax.set_yticklabels([f'{y:.2f}' for y in pivot.index])
    
    # Add grid
    for ax in axes:
        ax.grid(True, alpha=0.3, color='white', linewidth=0.5)
    
    plt.suptitle('QFTT Parameter Space Analysis', fontsize=18, fontweight='bold')
    
    # Save
    output_path = output_dir / 'param_heatmap_combined.png'
    fig.savefig(output_path, bbox_inches='tight')
    plt.close()
    
    print(f"Created: {output_path}")


def create_all_figures(data_path, output_dir, run_new_simulation=False):
    """Generate all standard figures."""
    
    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load or generate data
    if run_new_simulation:
        print("Running new simulation...")
        sim = QFTTSimulator(max_events=1000)
        df = sim.run(random_seed=42)
    else:
        print(f"Loading data from {data_path}...")
        df = load_events(data_path)
    
    print(f"Loaded {len(df)} events")
    
    # Figure 1: Entropy vs Event
    print("Creating entropy vs event plot...")
    fig1 = plot_entropy_vs_event(df)
    fig1.savefig(output_dir / 'entropy_vs_event.png', bbox_inches='tight')
    fig1.savefig(output_dir / 'entropy_vs_event.pdf', bbox_inches='tight')
    plt.close(fig1)
    
    # Figure 2: Interval histogram
    print("Creating interval histogram...")
    fig2 = plot_interval_histogram(df, n_bins=50, fit_exponential=True)
    fig2.savefig(output_dir / 'delta_t_distribution.png', bbox_inches='tight')
    fig2.savefig(output_dir / 'delta_t_distribution.pdf', bbox_inches='tight')
    plt.close(fig2)
    
    # Figure 3: Q-Q plot
    print("Creating Q-Q plot...")
    fig3 = plot_qq_exponential(df)
    fig3.savefig(output_dir / 'qq_plot_exponential.png', bbox_inches='tight')
    plt.close(fig3)
    
    # Figure 4: Time series
    print("Creating time series plots...")
    fig4 = plot_time_series(df, quantities=['entropy', 'energy'])
    fig4.savefig(output_dir / 'time_series.png', bbox_inches='tight')
    plt.close(fig4)
    
    # Figure 5: Eigentime balance
    print("Creating eigentime analysis plot...")
    fig5, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Eigentime sequence
    ax1.plot(df['event_id'], df['eigentime'], 'b-', alpha=0.5)
    ax1.scatter(df['event_id'], df['eigentime'], c=['red' if x == 1 else 'blue' for x in df['eigentime']], 
                s=20, alpha=0.7)
    ax1.set_xlabel('Event Number')
    ax1.set_ylabel('Eigentime')
    ax1.set_title('Clock Eigentime Sequence')
    ax1.set_ylim(-1.5, 1.5)
    ax1.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax1.grid(True, alpha=0.3)
    
    # Running balance
    cumsum = np.cumsum(df['eigentime'].values)
    balance = cumsum / np.arange(1, len(cumsum) + 1)
    ax2.plot(df['event_id'], balance, 'g-', linewidth=2)
    ax2.axhline(y=0, color='red', linestyle='--', alpha=0.5, label='Perfect balance')
    ax2.fill_between(df['event_id'], -0.1, 0.1, alpha=0.2, color='gray', label='±0.1 band')
    ax2.set_xlabel('Event Number')
    ax2.set_ylabel('Running Average')
    ax2.set_title('Eigentime Balance Evolution')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    fig5.suptitle('Clock Eigentime Analysis', fontsize=16, fontweight='bold')
    fig5.savefig(output_dir / 'eigentime_analysis.png', bbox_inches='tight')
    plt.close(fig5)
    
    print(f"\nAll figures saved to {output_dir}/")


def main():
    parser = argparse.ArgumentParser(
        description='Generate figures from QFTT simulation data',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument('--data', type=str, default='qftt_events.csv',
                       help='Path to event data file')
    parser.add_argument('--sweep_results', type=str, 
                       default='data/benchmarks/parameter_sweep_results.csv',
                       help='Path to parameter sweep results')
    parser.add_argument('--output_dir', type=str, default='figures',
                       help='Output directory for figures')
    parser.add_argument('--run_new', action='store_true',
                       help='Run new simulation instead of loading data')
    parser.add_argument('--style', type=str, default='seaborn-v0_8-darkgrid',
                       help='Matplotlib style')
    
    args = parser.parse_args()
    
    # Set style
    plt.style.use(args.style)
    
    # Generate standard figures
    create_all_figures(args.data, args.output_dir, args.run_new)
    
    # Generate heatmap if sweep results exist
    sweep_path = Path(args.sweep_results)
    if sweep_path.exists():
        print("\nCreating parameter heatmaps...")
        create_parameter_heatmap(sweep_path, Path(args.output_dir))
    else:
        print(f"\nSkipping heatmap (no sweep results at {sweep_path})")


if __name__ == "__main__":
    main()