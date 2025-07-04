"""
Unit tests for QFTT simulator.

© 2024 - MIT License
"""

import pytest
import numpy as np
import pandas as pd
from qftt_bootstrap_loop.src.qftt_simulator import QFTTSimulator, run_quick_simulation


class TestQFTTSimulator:
    """Test suite for QFTTSimulator class."""
    
    def test_initialization(self):
        """Test simulator initialization with default parameters."""
        sim = QFTTSimulator()
        
        assert sim.Omega == 1.0
        assert sim.g == 0.2
        assert sim.lambda_se == 0.1
        assert sim.dt == 0.05
        assert sim.threshold == 0.5
        assert sim.max_events == 1000
        assert sim.t_max == 200.0
        
    def test_initial_energy_zero(self):
        """Test that initial state has total energy ≈ 0."""
        sim = QFTTSimulator()
        assert abs(sim.initial_energy) < 1e-6
        
    def test_parameter_validation(self):
        """Test parameter validation."""
        # Invalid threshold
        with pytest.raises(ValueError, match="threshold must be between 0 and 1"):
            QFTTSimulator(threshold=1.5)
            
        # Invalid dt
        with pytest.raises(ValueError, match="dt must be positive"):
            QFTTSimulator(dt=-0.1)
            
        # Invalid max_events
        with pytest.raises(ValueError, match="max_events must be positive"):
            QFTTSimulator(max_events=0)
            
    def test_basic_simulation(self):
        """Test running a basic simulation."""
        sim = QFTTSimulator(max_events=10, t_max=50.0)
        df = sim.run(random_seed=42)
        
        # Check output format
        assert isinstance(df, pd.DataFrame)
        assert len(df) > 0
        assert len(df) <= 10
        
        # Check columns
        expected_cols = ['event_id', 'event_type', 'eigentime', 'delta_t',
                        'entropy', 'energy', 'collapse_basis', 'system_state']
        assert all(col in df.columns for col in expected_cols)
        
        # Check event IDs are sequential
        assert list(df['event_id']) == list(range(1, len(df) + 1))
        
        # Check eigentime values are ±1
        assert all(df['eigentime'].isin([1.0, -1.0]))
        
        # Check delta_t is positive
        assert all(df['delta_t'] > 0)
        
    def test_reproducibility(self):
        """Test that same seed gives same results."""
        sim1 = QFTTSimulator(max_events=20)
        df1 = sim1.run(random_seed=123)
        
        sim2 = QFTTSimulator(max_events=20)
        df2 = sim2.run(random_seed=123)
        
        # Check same number of events
        assert len(df1) == len(df2)
        
        # Check same eigentime sequence
        assert np.array_equal(df1['eigentime'].values, df2['eigentime'].values)
        
        # Check same entropy values (within tolerance)
        assert np.allclose(df1['entropy'].values, df2['entropy'].values, atol=1e-10)
        
    def test_no_collapse_high_threshold(self):
        """Test that very high threshold prevents collapses."""
        sim = QFTTSimulator(threshold=1.0, max_events=10, t_max=10.0)
        df = sim.run(random_seed=42)
        
        # Should have no events
        assert len(df) == 0
        
    def test_no_collapse_zero_coupling(self):
        """Test that zero coupling prevents collapses."""
        sim = QFTTSimulator(g=0.0, max_events=10, t_max=10.0)
        df = sim.run(random_seed=42)
        
        # Should have no or very few events
        assert len(df) < 3
        
    def test_frequent_collapse_low_threshold(self):
        """Test frequent collapses with very low threshold."""
        sim = QFTTSimulator(threshold=0.01, max_events=50, dt=0.1)
        df = sim.run(random_seed=42)
        
        # Should reach max events quickly
        assert len(df) == 50
        
        # Check for Zeno-like behavior (very small delta_t)
        mean_dt = df['delta_t'].mean()
        assert mean_dt < 1.0  # Much smaller than normal
        
    def test_entropy_increases(self):
        """Test that entropy generally increases."""
        sim = QFTTSimulator(max_events=100)
        df = sim.run(random_seed=42)
        
        # Entropy should generally increase
        entropy_diff = np.diff(df['entropy'].values)
        fraction_increasing = np.sum(entropy_diff > -1e-6) / len(entropy_diff)
        assert fraction_increasing > 0.8  # Most steps should increase entropy
        
        # Final entropy should be higher than initial
        assert df.iloc[-1]['entropy'] > df.iloc[0]['entropy']
        
    def test_state_info_method(self):
        """Test get_state_info method."""
        sim = QFTTSimulator()
        info = sim.get_state_info()
        
        assert 'time' in info
        assert 'clock_entropy' in info
        assert 'system_entropy' in info
        assert 'env_entropy' in info
        assert 'total_energy' in info
        assert 'clock_time_exp' in info
        assert 'time_uncertainty' in info
        
        # Initial state checks
        assert info['time'] == 0.0
        assert abs(info['total_energy']) < 1e-6
        
    def test_quick_simulation_function(self):
        """Test convenience function for quick runs."""
        df = run_quick_simulation(n_events=10, g=0.3)
        
        assert len(df) <= 10
        assert isinstance(df, pd.DataFrame)
        
    def test_output_files_created(self, tmp_path, monkeypatch):
        """Test that output files are created."""
        # Change to temp directory
        monkeypatch.chdir(tmp_path)
        
        sim = QFTTSimulator(max_events=5)
        df = sim.run()
        
        # Check files exist
        assert (tmp_path / "qftt_events.csv").exists()
        assert (tmp_path / "qftt_events.h5").exists()
        
        # Check CSV content
        df_loaded = pd.read_csv(tmp_path / "qftt_events.csv")
        assert len(df_loaded) == len(df)
        
    def test_system_state_format(self):
        """Test that system_state column has correct format."""
        sim = QFTTSimulator(max_events=5)
        df = sim.run(random_seed=42)
        
        for state_str in df['system_state']:
            # Should match format "P(0)=0.XXX, P(1)=0.XXX"
            assert state_str.startswith("P(0)=")
            assert ", P(1)=" in state_str
            
            # Parse and check probabilities sum to 1
            parts = state_str.split(', ')
            p0 = float(parts[0].split('=')[1])
            p1 = float(parts[1].split('=')[1])
            assert abs(p0 + p1 - 1.0) < 1e-3
            
    def test_environment_reset_effect(self):
        """Test effect of environment reset on entropy."""
        # Run with environment coupling
        sim1 = QFTTSimulator(lambda_se=0.2, max_events=50)
        df1 = sim1.run(random_seed=42)
        
        # Run without environment coupling
        sim2 = QFTTSimulator(lambda_se=0.0, max_events=50)
        df2 = sim2.run(random_seed=42)
        
        # With coupling should have different entropy evolution
        if len(df1) > 10 and len(df2) > 10:
            entropy1_change = df1.iloc[-1]['entropy'] - df1.iloc[0]['entropy']
            entropy2_change = df2.iloc[-1]['entropy'] - df2.iloc[0]['entropy']
            assert abs(entropy1_change - entropy2_change) > 0.1


class TestEdgeCases:
    """Test edge cases and failure modes."""
    
    def test_very_long_simulation(self):
        """Test that simulation stops at t_max."""
        sim = QFTTSimulator(threshold=0.99, max_events=1000, t_max=5.0, dt=0.01)
        df = sim.run(random_seed=42)
        
        # Should stop due to time limit, not event limit
        assert len(df) < 1000
        assert sim.t >= 5.0
        
    def test_extreme_parameters(self):
        """Test simulation with extreme but valid parameters."""
        # Very strong coupling
        sim1 = QFTTSimulator(g=2.0, max_events=10)
        df1 = sim1.run(random_seed=42)
        assert len(df1) > 0
        
        # Very weak coupling
        sim2 = QFTTSimulator(g=0.01, max_events=10, t_max=50.0)
        df2 = sim2.run(random_seed=42)
        # May or may not have events depending on threshold
        
    def test_different_omega_values(self):
        """Test with different energy scales."""
        for omega in [0.1, 1.0, 10.0]:
            sim = QFTTSimulator(Omega=omega, max_events=10)
            df = sim.run(random_seed=42)
            # Should still work with different energy scales
            assert isinstance(df, pd.DataFrame)