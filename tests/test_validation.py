"""
Unit tests for QFTT validation module.

© 2024 - MIT License
"""

import pytest
import numpy as np
import pandas as pd
from unittest.mock import Mock

from qftt_bootstrap_loop.src import qftt_validation as val
from qftt_bootstrap_loop.src.qftt_simulator import QFTTSimulator


class TestInitialEnergyCheck:
    """Test initial energy validation."""
    
    def test_valid_initial_energy(self):
        """Test that properly initialized simulator has zero energy."""
        sim = QFTTSimulator()
        assert val.check_initial_energy(sim, tolerance=1e-6) == True
        
    def test_invalid_initial_energy(self):
        """Test detection of non-zero initial energy."""
        # Create mock simulator with non-zero energy
        mock_sim = Mock()
        mock_sim.initial_energy = 0.1
        
        assert val.check_initial_energy(mock_sim, tolerance=1e-6) == False
        assert val.check_initial_energy(mock_sim, tolerance=0.2) == True


class TestEntropyMonotonic:
    """Test entropy monotonicity validation."""
    
    def test_monotonic_entropy(self):
        """Test detection of properly monotonic entropy."""
        df = pd.DataFrame({
            'entropy': [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
        })
        
        is_valid, diagnostics = val.check_entropy_monotonic(df)
        assert is_valid == True
        assert diagnostics['n_violations'] == 0
        
    def test_non_monotonic_entropy(self):
        """Test detection of entropy violations."""
        df = pd.DataFrame({
            'entropy': [0.0, 0.2, 0.1, 0.3, 0.25, 0.4]  # Decreases at indices 2 and 4
        })
        
        is_valid, diagnostics = val.check_entropy_monotonic(df, tolerance=1e-6)
        assert is_valid == False
        assert diagnostics['n_violations'] == 2
        assert 2 in diagnostics['violation_indices']
        assert 4 in diagnostics['violation_indices']
        
    def test_small_fluctuations_allowed(self):
        """Test that small fluctuations within tolerance are allowed."""
        df = pd.DataFrame({
            'entropy': [0.0, 0.1, 0.099999, 0.2]  # Tiny decrease
        })
        
        is_valid, diagnostics = val.check_entropy_monotonic(df, tolerance=1e-5)
        assert is_valid == True  # Within tolerance
        
        is_valid_strict, _ = val.check_entropy_monotonic(df, tolerance=1e-7)
        assert is_valid_strict == False  # Not within stricter tolerance


class TestEigentimeDistribution:
    """Test eigentime value validation."""
    
    def test_balanced_eigentime(self):
        """Test detection of balanced eigentime distribution."""
        # Create perfectly balanced data
        df = pd.DataFrame({
            'eigentime': [1, -1] * 50  # 50 of each
        })
        
        is_valid, diagnostics = val.check_eigentime_values(df)
        assert is_valid == True
        assert diagnostics['valid_values'] == True
        assert diagnostics['plus_fraction'] == pytest.approx(0.5)
        assert diagnostics['binomial_p_value'] > 0.05
        
    def test_unbalanced_eigentime(self):
        """Test detection of unbalanced eigentime."""
        # Create very unbalanced data
        df = pd.DataFrame({
            'eigentime': [1] * 90 + [-1] * 10
        })
        
        is_valid, diagnostics = val.check_eigentime_values(df)
        assert is_valid == False
        assert diagnostics['plus_fraction'] == pytest.approx(0.9)
        assert diagnostics['binomial_p_value'] < 0.05
        
    def test_invalid_eigentime_values(self):
        """Test detection of invalid eigentime values."""
        df = pd.DataFrame({
            'eigentime': [1, -1, 0, 2, -1]  # Contains invalid 0 and 2
        })
        
        is_valid, diagnostics = val.check_eigentime_values(df)
        assert is_valid == False
        assert diagnostics['valid_values'] == False
        
    def test_suspicious_runs(self):
        """Test detection of suspicious long runs."""
        # Create data with very long run of same value
        df = pd.DataFrame({
            'eigentime': [1] * 50 + [-1] * 50  # Long runs
        })
        
        is_valid, diagnostics = val.check_eigentime_values(df)
        # May or may not be suspicious depending on expected max run
        assert 'max_run_length' in diagnostics
        assert diagnostics['max_run_length'] == 50


class TestEnvironmentReset:
    """Test environment reset validation."""
    
    def test_valid_environment_reset(self):
        """Test valid system state probabilities."""
        df = pd.DataFrame({
            'system_state': [
                'P(0)=0.500, P(1)=0.500',
                'P(0)=0.700, P(1)=0.300',
                'P(0)=0.100, P(1)=0.900'
            ]
        })
        
        is_valid, diagnostics = val.check_environment_reset(df)
        assert is_valid == True
        assert diagnostics['normalization_ok'] == True
        assert diagnostics['valid_probabilities'] == True
        
    def test_unnormalized_probabilities(self):
        """Test detection of unnormalized probabilities."""
        df = pd.DataFrame({
            'system_state': [
                'P(0)=0.600, P(1)=0.500',  # Sum > 1
                'P(0)=0.300, P(1)=0.300'   # Sum < 1
            ]
        })
        
        is_valid, diagnostics = val.check_environment_reset(df)
        assert is_valid == False
        assert diagnostics['normalization_ok'] == False
        
    def test_invalid_probability_values(self):
        """Test detection of invalid probability values."""
        df = pd.DataFrame({
            'system_state': [
                'P(0)=-0.100, P(1)=1.100',  # Negative probability
                'P(0)=0.500, P(1)=0.500'
            ]
        })
        
        is_valid, diagnostics = val.check_environment_reset(df)
        assert is_valid == False
        assert diagnostics['valid_probabilities'] == False
        
    def test_purity_analysis(self):
        """Test purity trend analysis."""
        df = pd.DataFrame({
            'system_state': [
                'P(0)=1.000, P(1)=0.000',  # Pure state (purity=1)
                'P(0)=0.900, P(1)=0.100',  # High purity
                'P(0)=0.500, P(1)=0.500'   # Maximum mixture (purity=0.5)
            ]
        })
        
        is_valid, diagnostics = val.check_environment_reset(df)
        assert diagnostics['mean_purity'] > 0.5
        assert diagnostics['final_purity'] == pytest.approx(0.5)
        assert diagnostics['purity_trend'] < 0  # Decreasing purity


class TestEventSequence:
    """Test event sequence validation."""
    
    def test_valid_sequence(self):
        """Test valid sequential event IDs."""
        df = pd.DataFrame({
            'event_id': [1, 2, 3, 4, 5],
            'delta_t': [1.0, 0.5, 2.0, 1.5, 0.8]
        })
        
        is_valid, diagnostics = val.check_event_sequence(df)
        assert is_valid == True
        assert diagnostics['sequence_valid'] == True
        assert diagnostics['n_events'] == 5
        assert diagnostics['delta_t_positive'] == True
        
    def test_gap_in_sequence(self):
        """Test detection of gaps in event sequence."""
        df = pd.DataFrame({
            'event_id': [1, 2, 4, 5],  # Missing 3
            'delta_t': [1.0, 0.5, 2.0, 1.5]
        })
        
        is_valid, diagnostics = val.check_event_sequence(df)
        assert is_valid == False
        assert diagnostics['sequence_valid'] == False
        assert len(diagnostics['gaps']) > 0
        
    def test_negative_delta_t(self):
        """Test detection of negative time intervals."""
        df = pd.DataFrame({
            'event_id': [1, 2, 3],
            'delta_t': [1.0, -0.5, 2.0]  # Negative delta_t
        })
        
        is_valid, diagnostics = val.check_event_sequence(df)
        assert is_valid == False
        assert diagnostics['delta_t_positive'] == False


class TestIntervalDistribution:
    """Test interval distribution validation."""
    
    def test_exponential_distribution(self):
        """Test detection of exponential distribution."""
        # Generate exponential data
        np.random.seed(42)
        delta_t_values = np.random.exponential(scale=2.0, size=100)
        
        df = pd.DataFrame({
            'event_id': range(1, 101),
            'delta_t': delta_t_values
        })
        
        is_valid, diagnostics = val.check_interval_distribution(df)
        # Should pass for truly exponential data (most of the time)
        assert 'p_value' in diagnostics
        assert diagnostics['cv'] == pytest.approx(1.0, rel=0.3)  # CV ≈ 1 for exponential
        
    def test_non_exponential_distribution(self):
        """Test detection of non-exponential distribution."""
        # Generate uniform data (definitely not exponential)
        delta_t_values = np.random.uniform(0, 4, size=100)
        
        df = pd.DataFrame({
            'event_id': range(1, 101),
            'delta_t': delta_t_values
        })
        
        is_valid, diagnostics = val.check_interval_distribution(df, significance_level=0.01)
        # Uniform distribution should fail exponential test
        assert diagnostics['cv'] < 0.8  # Much less than 1
        
    def test_insufficient_data(self):
        """Test handling of insufficient data."""
        df = pd.DataFrame({
            'event_id': [1, 2, 3],
            'delta_t': [1.0, 2.0, 1.5]
        })
        
        is_valid, diagnostics = val.check_interval_distribution(df)
        assert is_valid == True  # Skip test with insufficient data
        assert diagnostics.get('test_skipped', False) == True


class TestValidateSimulationOutput:
    """Test comprehensive validation function."""
    
    def test_full_validation_pass(self):
        """Test full validation with good data."""
        # Generate good simulation data
        sim = QFTTSimulator(max_events=50)
        df = sim.run(random_seed=42)
        
        results = val.validate_simulation_output(df, sim=sim, verbose=False)
        
        # Check all validations were run
        assert 'initial_energy' in results
        assert 'entropy_monotonic' in results
        assert 'eigentime_distribution' in results
        assert 'environment_reset' in results
        assert 'event_sequence' in results
        assert 'interval_distribution' in results
        
        # Most should pass for valid simulation
        passing_checks = sum(1 for passed, _ in results.values() if passed)
        assert passing_checks >= len(results) - 1  # Allow one failure
        
    def test_validation_without_simulator(self):
        """Test validation without simulator instance."""
        # Create minimal valid data
        df = pd.DataFrame({
            'event_id': [1, 2, 3],
            'eigentime': [1, -1, 1],
            'delta_t': [1.0, 2.0, 1.5],
            'entropy': [0.0, 0.1, 0.2],
            'system_state': ['P(0)=0.500, P(1)=0.500'] * 3
        })
        
        results = val.validate_simulation_output(df, sim=None, verbose=False)
        
        # Should not have initial_energy check
        assert 'initial_energy' not in results
        assert 'entropy_monotonic' in results


class TestFailureModeDetection:
    """Test detection of known failure modes."""
    
    def test_no_collapse_detection(self):
        """Test detection of no collapse scenario."""
        empty_df = pd.DataFrame()
        params = {'threshold': 0.99, 'g': 0.01, 'dt': 0.05}
        
        failures = val.detect_failure_modes(empty_df, params)
        assert failures['no_collapse'] == True
        assert failures['insufficient_events'] == False  # Different from no collapse
        
    def test_insufficient_events(self):
        """Test detection of insufficient events."""
        df = pd.DataFrame({
            'event_id': range(1, 51),  # Only 50 events
            'delta_t': [1.0] * 50,
            'entropy': np.linspace(0, 0.5, 50)
        })
        params = {'threshold': 0.7, 'g': 0.1, 'dt': 0.05}
        
        failures = val.detect_failure_modes(df, params, expected_events=1000)
        assert failures['insufficient_events'] == True
        
    def test_zeno_regime(self):
        """Test detection of quantum Zeno regime."""
        # Very frequent collapses
        df = pd.DataFrame({
            'event_id': range(1, 101),
            'delta_t': [0.06] * 100,  # Just above dt
            'entropy': np.linspace(0, 0.1, 100)
        })
        params = {'threshold': 0.01, 'g': 2.0, 'dt': 0.05}
        
        failures = val.detect_failure_modes(df, params)
        assert failures['zeno_regime'] == True
        
    def test_rapid_thermalization(self):
        """Test detection of rapid thermalization."""
        n_events = 100
        df = pd.DataFrame({
            'event_id': range(1, n_events + 1),
            'delta_t': [1.0] * n_events,
            'entropy': np.concatenate([np.linspace(0, 0.98, 50), 
                                      [0.99] * 50])  # Saturates quickly
        })
        params = {'lambda_se': 1.0}  # Strong coupling
        
        failures = val.detect_failure_modes(df, params)
        assert failures['rapid_thermalization'] == True
        
    def test_no_time_emergence(self):
        """Test detection of no time emergence."""
        # Few events with high threshold
        df = pd.DataFrame({
            'event_id': range(1, 201),
            'delta_t': [5.0] * 200,  # Long intervals
            'entropy': np.linspace(0, 0.2, 200)
        })
        params = {'threshold': 0.95, 'g': 0.05, 'dt': 0.05}
        
        failures = val.detect_failure_modes(df, params, expected_events=1000)
        assert failures['no_time_emergence'] == True


class TestEdgeCases:
    """Test edge cases in validation."""
    
    def test_empty_dataframe(self):
        """Test validation of empty DataFrame."""
        empty_df = pd.DataFrame()
        
        # Should handle gracefully
        results = val.validate_simulation_output(empty_df, verbose=False)
        
        # Most checks should fail or be skipped
        for check_name, (passed, diagnostics) in results.items():
            if check_name != 'initial_energy':  # This needs simulator
                assert passed == False or 'test_skipped' in diagnostics
                
    def test_single_event(self):
        """Test validation with single event."""
        single_event_df = pd.DataFrame({
            'event_id': [1],
            'eigentime': [1],
            'delta_t': [1.0],
            'entropy': [0.5],
            'system_state': ['P(0)=0.500, P(1)=0.500']
        })
        
        results = val.validate_simulation_output(single_event_df, verbose=False)
        
        # Some checks should handle single event gracefully
        sequence_passed, _ = results['event_sequence']
        assert sequence_passed == True  # Single event is valid sequence