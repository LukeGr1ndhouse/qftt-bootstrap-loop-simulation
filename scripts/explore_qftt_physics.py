#!/usr/bin/env python
"""
Explore QFTT Bootstrap Loop physics and compare with theoretical predictions.

This script runs autonomous quantum simulations to discover emergent time behavior.
While it uses parameters from the theoretical paper as starting points, all results
emerge naturally from the quantum dynamics - nothing is pre-programmed to match.

The simulation will reveal whether:
1. Time truly emerges from quantum collapse
2. Environment reset is necessary for arrow of time
3. The dynamics match or differ from theoretical predictions

© 2024 - MIT License
"""

import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.qftt_simulator import QFTTSimulator
from src import qftt_analysis as qa


def analyze_results(df, scenario_name):
    """Perform comprehensive analysis and draw autonomous conclusions."""
    print(f"\n{'='*60}")
    print(f"AUTONOMOUS ANALYSIS: {scenario_name}")
    print(f"{'='*60}")
    
    # Basic statistics
    n_events = len(df)
    print(f"\nBasic Statistics:")
    print(f"  Total events: {n_events}")
    
    if n_events == 0:
        print("  NO EVENTS - System failed to produce time emergence!")
        return {}
    
    # Time intervals analysis
    dt_values = df['delta_t'][df['event_id'] > 1].values
    print(f"\nTime Interval Analysis:")
    print(f"  Mean Δt: {np.mean(dt_values):.3f} ± {np.std(dt_values):.3f}")
    print(f"  Coefficient of variation: {np.std(dt_values)/np.mean(dt_values):.3f}")
    
    # Test for exponential distribution (Poisson process)
    if len(dt_values) > 20:
        ks_stat, ks_pvalue = stats.kstest(dt_values, 
                                         lambda x: stats.expon.cdf(x, scale=np.mean(dt_values)))
        if ks_pvalue > 0.05:
            print(f"  ✓ Intervals follow exponential distribution (p={ks_pvalue:.3f})")
            print(f"    → Collapse process is MEMORYLESS (Poisson-like)")
        else:
            print(f"  ✗ Intervals NOT exponential (p={ks_pvalue:.3f})")
            print(f"    → Collapse process has MEMORY or PERIODICITY")
    
    # Entropy analysis
    print(f"\nEntropy Evolution:")
    initial_entropy = df.iloc[0]['entropy']
    final_entropy = df.iloc[-1]['entropy']
    max_entropy = df['entropy'].max()
    
    print(f"  Initial: {initial_entropy:.6f} bits")
    print(f"  Final: {final_entropy:.6f} bits")
    print(f"  Maximum reached: {max_entropy:.6f} bits")
    
    # Detect entropy trend
    if n_events > 10:
        entropy_slope = np.polyfit(df['event_id'].values, df['entropy'].values, 1)[0]
        print(f"  Linear trend: {entropy_slope:.6f} bits/event")
        
        if entropy_slope > 0.0001:
            print(f"  ✓ ARROW OF TIME DETECTED - Entropy increases!")
        elif entropy_slope < -0.0001:
            print(f"  ✗ REVERSED ARROW - Entropy decreases (unphysical?)")
        else:
            print(f"  ○ NO ARROW OF TIME - Entropy saturated/oscillating")
    
    # Check for oscillations
    if n_events > 50:
        entropy_diff = np.diff(df['entropy'].values)
        sign_changes = np.sum(np.diff(np.sign(entropy_diff)) != 0)
        oscillation_rate = sign_changes / len(entropy_diff)
        
        if oscillation_rate > 0.8:
            print(f"  ! High oscillation rate ({oscillation_rate:.2f}) - System in quasi-equilibrium")
    
    # Eigentime fairness
    print(f"\nClock Eigentime Analysis:")
    plus_count = (df['eigentime'] == 1).sum()
    minus_count = (df['eigentime'] == -1).sum()
    
    print(f"  +1 outcomes: {plus_count} ({100*plus_count/n_events:.1f}%)")
    print(f"  -1 outcomes: {minus_count} ({100*minus_count/n_events:.1f}%)")
    
    # Test for bias
    binom_test = stats.binom_test(plus_count, n_events, p=0.5, alternative='two-sided')
    if binom_test > 0.05:
        print(f"  ✓ Clock ticks are UNBIASED (p={binom_test:.3f})")
    else:
        print(f"  ✗ Clock shows BIAS (p={binom_test:.3f})")
    
    # Energy flow
    print(f"\nEnergy Dynamics:")
    initial_energy = df.iloc[0]['energy']
    final_energy = df.iloc[-1]['energy']
    print(f"  System energy: {initial_energy:.3f} → {final_energy:.3f}")
    
    if final_energy > initial_energy + 0.1:
        print(f"  → System GAINED energy (excited)")
    elif final_energy < initial_energy - 0.1:
        print(f"  → System LOST energy (cooled)")
    else:
        print(f"  → System energy STABLE")
    
    # Final conclusions
    print(f"\nEMERGENT CONCLUSIONS:")
    
    conclusions = {
        'time_emerged': n_events > 10,
        'arrow_of_time': entropy_slope > 0.0001 if n_events > 10 else False,
        'memoryless': ks_pvalue > 0.05 if len(dt_values) > 20 else None,
        'unbiased_clock': binom_test > 0.05,
        'entropy_saturated': abs(entropy_slope) < 0.0001 if n_events > 10 else False,
        'oscillating': oscillation_rate > 0.8 if n_events > 50 else False
    }
    
    if conclusions['time_emerged']:
        print(f"  ✓ TIME SUCCESSFULLY EMERGED from quantum dynamics")
    else:
        print(f"  ✗ TIME FAILED TO EMERGE - check parameters")
    
    if conclusions['arrow_of_time']:
        print(f"  ✓ THERMODYNAMIC ARROW established")
    elif conclusions['entropy_saturated']:
        print(f"  ○ THERMAL EQUILIBRIUM reached - no arrow")
    
    return conclusions