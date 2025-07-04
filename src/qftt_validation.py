"""
QFTT Validation Module

Validation checks for physical invariants and consistency of simulation outputs.

© 2024 - MIT License
"""

import numpy as np
import pandas as pd
from typing import Dict, Tuple, Optional
from scipy import stats


def check_initial_energy(sim, tolerance: float = 1e-6) -> bool:
    """
    Check that initial state has total energy ≈ 0.
    
    Parameters
    ----------
    sim : QFTTSimulator
        Simulator instance
    tolerance : float
        Tolerance for energy check
        
    Returns
    -------
    bool
        True if initial energy is within tolerance of zero
    """
    return abs(sim.initial_energy) < tolerance


def check_entropy_monotonic(events_df: pd.DataFrame, 
                           tolerance: float = 1e-6) -> Tuple[bool, Dict]:
    """
    Verify that system entropy is non-decreasing (within tolerance).
    
    Parameters
    ----------
    events_df : pd.DataFrame
        Event data with 'entropy' column
    tolerance : float
        Allowed decrease in entropy
        
    Returns
    -------
    bool
        True if entropy is monotonic within tolerance
    dict
        Diagnostic information
    """
    entropy = events_df['entropy'].values
    entropy_diff = np.diff(entropy)
    
    # Find violations
    violations = np.where(entropy_diff < -tolerance)[0]
    max_decrease = np.min(entropy_diff) if len(entropy_diff) > 0 else 0
    
    diagnostics = {
        'n_violations': len(violations),
        'violation_indices': violations + 1,  # +1 for event numbering
        'max_decrease': max_decrease,
        'mean_increase': np.mean(entropy_diff[entropy_diff > 0]) if any(entropy_diff > 0) else 0
    }
    
    return len(violations) == 0, diagnostics


def check_eigentime_values(events_df: pd.DataFrame, 
                          significance_level: float = 0.05) -> Tuple[bool, Dict]:
    """
    Check that eigentime values (+1/-1) are well-distributed.
    
    Parameters
    ----------
    events_df : pd.DataFrame
        Event data with 'eigentime' column
    significance_level : float
        Significance level for statistical test
        
    Returns
    -------
    bool
        True if eigentime distribution appears fair
    dict
        Statistical test results
    """
    eigentime = events_df['eigentime'].values
    plus_count = np.sum(eigentime == 1)
    minus_count = np.sum(eigentime == -1)
    total = len(eigentime)
    
    # Check all values are ±1
    valid_values = np.all(np.isin(eigentime, [1, -1]))
    
    # Binomial test for fairness
    p_value = stats.binom_test(plus_count, total, p=0.5, alternative='two-sided')
    
    # Check for long runs (potential bias)
    max_run = 1
    current_run = 1
    for i in range(1, len(eigentime)):
        if eigentime[i] == eigentime[i-1]:
            current_run += 1
            max_run = max(max_run, current_run)
        else:
            current_run = 1
            
    # Expected max run length (rough approximation)
    expected_max_run = np.log2(total) * 2
    
    diagnostics = {
        'plus_count': plus_count,
        'minus_count': minus_count,
        'plus_fraction': plus_count / total,
        'valid_values': valid_values,
        'binomial_p_value': p_value,
        'max_run_length': max_run,
        'expected_max_run': expected_max_run,
        'suspicious_runs': max_run > expected_max_run * 1.5
    }
    
    return valid_values and p_value > significance_level and not diagnostics['suspicious_runs'], diagnostics


def check_environment_reset(events_df: pd.DataFrame) -> Tuple[bool, Dict]:
    """
    Verify environment reset is working by checking system state properties.
    
    Parameters
    ----------
    events_df : pd.DataFrame
        Event data with 'system_state' column
        
    Returns
    -------
    bool
        True if no anomalies detected
    dict
        Diagnostic information
    """
    # Parse system state probabilities
    p0_list = []
    p1_list = []
    
    for state_str in events_df['system_state']:
        # Parse "P(0)=0.430, P(1)=0.570" format
        parts = state_str.split(', ')
        p0 = float(parts[0].split('=')[1])
        p1 = float(parts[1].split('=')[1])
        p0_list.append(p0)
        p1_list.append(p1)
        
    p0_array = np.array(p0_list)
    p1_array = np.array(p1_list)
    
    # Check normalization
    norm_check = np.allclose(p0_array + p1_array, 1.0, atol=1e-3)
    
    # Check for reasonable probability ranges
    valid_probs = np.all((p0_array >= 0) & (p0_array <= 1) & 
                        (p1_array >= 0) & (p1_array <= 1))
    
    # Compute purity trend
    purity = p0_array**2 + p1_array**2
    
    diagnostics = {
        'normalization_ok': norm_check,
        'valid_probabilities': valid_probs,
        'mean_purity': np.mean(purity),
        'final_purity': purity[-1],
        'purity_trend': np.polyfit(range(len(purity)), purity, 1)[0]  # slope
    }
    
    return norm_check and valid_probs, diagnostics


def check_event_sequence(events_df: pd.DataFrame) -> Tuple[bool, Dict]:
    """
    Ensure event IDs are sequential with no gaps or repeats.
    
    Parameters
    ----------
    events_df : pd.DataFrame
        Event data
        
    Returns
    -------
    bool
        True if sequence is valid
    dict
        Diagnostic information
    """
    event_ids = events_df['event_id'].values
    expected_ids = np.arange(1, len(event_ids) + 1)
    
    sequence_ok = np.array_equal(event_ids, expected_ids)
    
    # Check delta_t consistency
    first_delta_t = events_df.iloc[0]['delta_t']
    delta_t_positive = np.all(events_df['delta_t'] > 0)
    
    diagnostics = {
        'sequence_valid': sequence_ok,
        'n_events': len(event_ids),
        'delta_t_positive': delta_t_positive,
        'first_delta_t': first_delta_t,
        'gaps': [] if sequence_ok else np.where(np.diff(event_ids) != 1)[0].tolist()
    }
    
    return sequence_ok and delta_t_positive, diagnostics


def check_interval_distribution(events_df: pd.DataFrame, 
                               test: str = 'ks',
                               significance_level: float = 0.05) -> Tuple[bool, Dict]:
    """
    Test if collapse intervals follow expected distribution (exponential).
    
    Parameters
    ----------
    events_df : pd.DataFrame
        Event data
    test : str
        Statistical test to use ('ks' for Kolmogorov-Smirnov)
    significance_level : float
        Significance level
        
    Returns
    -------
    bool
        True if distribution matches expected
    dict
        Test results
    """
    delta_t = events_df['delta_t'][events_df['event_id'] > 1].values
    
    if len(delta_t) < 10:
        return True, {'test_skipped': True, 'reason': 'insufficient data'}
    
    # Fit exponential
    mean_dt = np.mean(delta_t)
    
    if test == 'ks':
        # Kolmogorov-Smirnov test
        statistic, p_value = stats.kstest(delta_t, 
                                         lambda x: stats.expon.cdf(x, scale=mean_dt))
    else:
        raise ValueError(f"Unknown test: {test}")
        
    diagnostics = {
        'test': test,
        'statistic': statistic,
        'p_value': p_value,
        'mean_interval': mean_dt,
        'std_interval': np.std(delta_t),
        'cv': np.std(delta_t) / mean_dt  # Coefficient of variation (should be ~1 for exponential)
    }
    
    return p_value > significance_level, diagnostics


def validate_simulation_output(events_df: pd.DataFrame, 
                              sim: Optional = None,
                              verbose: bool = True) -> Dict[str, Tuple[bool, Dict]]:
    """
    Run all validation checks on simulation output.
    
    Parameters
    ----------
    events_df : pd.DataFrame
        Event data
    sim : QFTTSimulator, optional
        Simulator instance (for initial energy check)
    verbose : bool
        Whether to print results
        
    Returns
    -------
    dict
        Results of all validation checks
    """
    results = {}
    
    # Check initial energy if simulator provided
    if sim is not None:
        energy_ok = check_initial_energy(sim)
        results['initial_energy'] = (energy_ok, {'energy': sim.initial_energy})
    
    # Run all checks
    results['entropy_monotonic'] = check_entropy_monotonic(events_df)
    results['eigentime_distribution'] = check_eigentime_values(events_df)
    results['environment_reset'] = check_environment_reset(events_df)
    results['event_sequence'] = check_event_sequence(events_df)
    results['interval_distribution'] = check_interval_distribution(events_df)
    
    # Summary
    all_passed = all(result[0] for result in results.values())
    
    if verbose:
        print("=== Validation Results ===")
        for check_name, (passed, diagnostics) in results.items():
            status = "✓ PASSED" if passed else "✗ FAILED"
            print(f"{check_name}: {status}")
            
            # Print key diagnostics for failed checks
            if not passed:
                for key, value in diagnostics.items():
                    if key not in ['violation_indices', 'gaps']:  # Skip long lists
                        print(f"  - {key}: {value}")
                        
        print(f"\nOverall: {'ALL CHECKS PASSED' if all_passed else 'SOME CHECKS FAILED'}")
        
    return results


def detect_failure_modes(events_df: pd.DataFrame, 
                        params: Dict,
                        expected_events: int = 1000) -> Dict[str, bool]:
    """
    Detect known failure modes based on parameters and results.
    
    Parameters
    ----------
    events_df : pd.DataFrame
        Event data
    params : dict
        Simulation parameters
    expected_events : int
        Expected number of events
        
    Returns
    -------
    dict
        Detected failure modes
    """
    n_events = len(events_df)
    
    failures = {
        'no_collapse': n_events == 0,
        'insufficient_events': 0 < n_events < expected_events * 0.1,
        'zeno_regime': False,
        'no_time_emergence': False,
        'rapid_thermalization': False
    }
    
    if n_events > 0:
        mean_dt = events_df['delta_t'].mean()
        
        # Zeno regime: collapses too frequent
        failures['zeno_regime'] = mean_dt < params.get('dt', 0.05) * 2
        
        # No time emergence: threshold too high
        failures['no_time_emergence'] = (params.get('threshold', 0.5) > 0.9 and 
                                         n_events < expected_events * 0.5)
        
        # Rapid thermalization: system saturates quickly
        final_entropy = events_df.iloc[-1]['entropy']
        halfway_entropy = events_df.iloc[n_events//2]['entropy'] if n_events > 2 else 0
        failures['rapid_thermalization'] = (final_entropy > 0.95 and 
                                           halfway_entropy > 0.9)
        
    return failures