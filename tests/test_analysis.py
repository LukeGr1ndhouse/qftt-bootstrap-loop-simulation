"""
Unit tests for QFTT analysis module.

© 2024 - MIT License
"""

import pytest
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import tempfile

from qftt_bootstrap_loop.src import qftt_analysis as qa


class TestDataLoading:
    """Test data loading functions."""
    
    def test_load_csv(self, tmp_path):
        """Test loading event data from CSV."""
        # Create sample data
        data = {
            'event_id': [1, 2, 3],
            'event_type': ['T_collapse'] * 3,
            'eigentime': [1, -1, 1],
            'delta_t': [1.0, 2.0, 1.5],
            'entropy': [0.0, 0.5, 0.7],
            'energy': [0.5, 0.3, 0.2],
            'collapse_basis': ['clock_T_eigenbasis'] * 3,
            'system_state': ['P(0)=1.000, P(1)=0.000'] * 3
        }
        df = pd.DataFrame(data)
        
        # Save to CSV
        csv_path = tmp_path / "test_events.csv"
        df.to_csv(csv_path, index=False)
        
        # Load and verify
        df_loaded = qa.load_events(csv_path)
        assert len(df_loaded) == 3
        assert list(df_loaded.columns) == list(df.columns)
        
    def test_load_hdf5(self, tmp_path):
        """Test loading event data from HDF5."""
        # Create sample data
        data = {
            'event_id': [1, 2, 3],
            'eigentime': [1, -1, 1],
            'delta_t': [1.0, 2.0, 1.5],
            'entropy': [0.0, 0.5, 0.7]
        }
        df = pd.DataFrame(data)
        
        # Save to HDF5
        h5_path = tmp_path / "test_events.h5"
        df.to_hdf(h5_path, key='events', mode='w')
        
        # Load and verify
        df_loaded = qa.load_events(h5_path)
        assert len(df_loaded) == 3
        
    def test_load_invalid_format(self, tmp_path):
        """Test error handling for invalid file format."""
        invalid_path = tmp_path / "test.txt"
        invalid_path.write_text("invalid data")
        
        with pytest.raises(ValueError, match="Unsupported file format"):
            qa.load_events(invalid_path)


class TestPlottingFunctions:
    """Test plotting and visualization functions."""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample event data for testing."""
        n_events = 100
        event_ids = np.arange(1, n_events + 1)
        
        # Simulate increasing entropy
        entropy = np.cumsum(np.random.uniform(0, 0.02, n_events))
        entropy = np.clip(entropy, 0, 1)
        
        # Simulate exponential intervals
        delta_t = np.random.exponential(scale=0.5, size=n_events)
        
        data = {
            'event_id': event_ids,
            'eigentime': np.random.choice([1, -1], size=n_events),
            'delta_t': delta_t,
            'entropy': entropy,
            'energy': 0.5 - entropy * 0.3,  # Decreasing energy
            'event_type': ['T_collapse'] * n_events,
            'collapse_basis': ['clock_T_eigenbasis'] * n_events,
            'system_state': ['P(0)=0.500, P(1)=0.500'] * n_events
        }
        
        return pd.DataFrame(data)
    
    def test_plot_entropy_vs_event(self, sample_data, tmp_path):
        """Test entropy vs event plotting."""
        fig = qa.plot_entropy_vs_event(sample_data)
        
        # Check figure properties
        assert isinstance(fig, plt.Figure)
        assert len(fig.axes) == 1
        
        # Check axes labels
        ax = fig.axes[0]
        assert ax.get_xlabel() == 'Event Number'
        assert ax.get_ylabel() == 'System Entropy (bits)'
        
        # Save and check file creation
        save_path = tmp_path / "entropy_plot.png"
        qa.plot_entropy_vs_event(sample_data, save_path=str(save_path))
        assert save_path.exists()
        
        plt.close(fig)
        
    def test_plot_interval_histogram(self, sample_data, tmp_path):
        """Test interval histogram plotting."""
        fig = qa.plot_interval_histogram(sample_data, n_bins=20, fit_exponential=True)
        
        assert isinstance(fig, plt.Figure)
        ax = fig.axes[0]
        
        # Check that histogram and fit line exist
        assert len(ax.patches) > 0  # Histogram bars
        assert len(ax.lines) > 0    # Exponential fit line
        
        # Check axes labels
        assert ax.get_xlabel() == 'Time Interval Δt'
        assert ax.get_ylabel() == 'Probability Density'
        
        plt.close(fig)
        
    def test_plot_qq_exponential(self, sample_data):
        """Test Q-Q plot against exponential distribution."""
        fig = qa.plot_qq_exponential(sample_data)
        
        assert isinstance(fig, plt.Figure)
        ax = fig.axes[0]
        
        # Check that Q-Q plot has data
        assert len(ax.lines) > 0
        assert ax.get_xlabel() == 'Theoretical Quantiles'
        assert ax.get_ylabel() == 'Sample Quantiles'
        
        plt.close(fig)
        
    def test_plot_time_series(self, sample_data):
        """Test time series plotting."""
        fig = qa.plot_time_series(sample_data, quantities=['entropy', 'energy'])
        
        assert isinstance(fig, plt.Figure)
        # Should have 2 subplots for 2 quantities
        assert len(fig.axes) == 2
        
        # Check subplot labels
        assert fig.axes[0].get_ylabel() == 'Entropy'
        assert fig.axes[1].get_ylabel() == 'Energy'
        assert fig.axes[1].get_xlabel() == 'Event Number'
        
        plt.close(fig)
        
    def test_plot_heatmap(self, tmp_path):
        """Test heatmap plotting for parameter sweeps."""
        # Create mock parameter sweep data
        g_values = np.array([0.1, 0.2, 0.3])
        threshold_values = np.array([0.3, 0.5, 0.7])
        results = np.random.rand(len(threshold_values), len(g_values)) * 1000
        
        fig = qa.plot_heatmap(
            param_grid=None,  # Not used in simplified version
            results=results,
            x_label='Coupling g',
            y_label='Threshold',
            title='Event Count Heatmap'
        )
        
        assert isinstance(fig, plt.Figure)
        assert len(fig.axes) == 2  # Main plot + colorbar
        
        plt.close(fig)


class TestStatisticalAnalysis:
    """Test statistical analysis functions."""
    
    def test_autocorrelation(self):
        """Test autocorrelation computation."""
        # Test with known sequence
        data = np.array([1, -1, 1, -1, 1, -1, 1, -1])
        autocorr = qa.autocorrelation(data, max_lag=4)
        
        assert len(autocorr) == 4
        assert autocorr[0] == pytest.approx(1.0)  # Zero lag = 1
        assert autocorr[1] < 0  # Negative correlation at lag 1
        
        # Test with random data
        random_data = np.random.randn(100)
        autocorr_random = qa.autocorrelation(random_data)
        assert autocorr_random[0] == pytest.approx(1.0)
        
    def test_analyze_eigentime_distribution(self):
        """Test eigentime distribution analysis."""
        # Create balanced data
        balanced_data = pd.DataFrame({
            'eigentime': [1, -1] * 50  # Perfectly balanced
        })
        
        stats = qa.analyze_eigentime_distribution(balanced_data)
        
        assert stats['plus_count'] == 50
        assert stats['minus_count'] == 50
        assert stats['plus_fraction'] == pytest.approx(0.5)
        assert stats['is_fair'] == True
        assert stats['binomial_p_value'] > 0.05
        
        # Create unbalanced data
        unbalanced_data = pd.DataFrame({
            'eigentime': [1] * 90 + [-1] * 10
        })
        
        stats_unbalanced = qa.analyze_eigentime_distribution(unbalanced_data)
        assert stats_unbalanced['is_fair'] == False
        assert stats_unbalanced['binomial_p_value'] < 0.05
        
    def test_get_interval_stats(self):
        """Test interval statistics computation."""
        # Create exponential-like data
        np.random.seed(42)
        exponential_data = pd.DataFrame({
            'event_id': range(1, 101),
            'delta_t': np.random.exponential(scale=2.0, size=100)
        })
        
        stats = qa.get_interval_stats(exponential_data)
        
        assert 'mean' in stats
        assert 'std' in stats
        assert 'lambda_mle' in stats
        assert stats['mean'] == pytest.approx(2.0, rel=0.2)  # Within 20%
        assert stats['lambda_mle'] == pytest.approx(0.5, rel=0.2)
        
        # Check KS test (should pass for exponential data)
        assert 'ks_pvalue' in stats
        assert 'is_exponential' in stats
        
    def test_entropy_rate_analysis(self):
        """Test entropy rate analysis."""
        # Create data with linear entropy growth
        n_events = 100
        data = pd.DataFrame({
            'event_id': range(1, n_events + 1),
            'entropy': np.linspace(0, 0.8, n_events),
            'delta_t': np.ones(n_events) * 0.1  # Constant intervals
        })
        
        analysis = qa.entropy_rate_analysis(data)
        
        assert analysis['total_entropy_gain'] == pytest.approx(0.8, rel=0.01)
        assert analysis['total_time'] == pytest.approx(10.0, rel=0.01)
        assert analysis['avg_entropy_rate'] == pytest.approx(0.08, rel=0.01)
        assert analysis['entropy_per_event'] == pytest.approx(0.008, rel=0.01)
        
        # Test with empty data
        empty_data = pd.DataFrame(columns=['entropy', 'delta_t'])
        analysis_empty = qa.entropy_rate_analysis(empty_data)
        assert analysis_empty['avg_entropy_rate'] == 0


class TestEdgeCases:
    """Test edge cases and error handling."""
    
    def test_empty_dataframe(self):
        """Test handling of empty DataFrames."""
        empty_df = pd.DataFrame()
        
        # These should not crash
        stats = qa.get_interval_stats(empty_df)
        assert 'mean' in stats
        
    def test_single_event(self):
        """Test handling of single event data."""
        single_event = pd.DataFrame({
            'event_id': [1],
            'eigentime': [1],
            'delta_t': [1.0],
            'entropy': [0.5]
        })
        
        # Should handle gracefully
        eigentime_stats = qa.analyze_eigentime_distribution(single_event)
        assert eigentime_stats['plus_count'] == 1
        
    def test_missing_columns(self):
        """Test handling of missing columns."""
        incomplete_data = pd.DataFrame({
            'event_id': [1, 2, 3],
            'entropy': [0.1, 0.2, 0.3]
            # Missing delta_t, eigentime, etc.
        })
        
        # Should handle missing columns gracefully
        with pytest.raises(KeyError):
            qa.plot_interval_histogram(incomplete_data)


class TestIntegration:
    """Integration tests with actual simulation data."""
    
    def test_full_analysis_pipeline(self, tmp_path):
        """Test complete analysis pipeline with simulated data."""
        # Generate realistic simulation data
        from qftt_bootstrap_loop.src.qftt_simulator import run_quick_simulation
        
        # Run quick simulation
        df = run_quick_simulation(n_events=50, random_seed=42)
        
        # Save to file
        csv_path = tmp_path / "test_simulation.csv"
        df.to_csv(csv_path, index=False)
        
        # Load and analyze
        df_loaded = qa.load_events(csv_path)
        
        # Run all analyses
        interval_stats = qa.get_interval_stats(df_loaded)
        eigentime_stats = qa.analyze_eigentime_distribution(df_loaded)
        entropy_stats = qa.entropy_rate_analysis(df_loaded)
        
        # Verify results make sense
        assert interval_stats['mean'] > 0
        assert 0 < eigentime_stats['plus_fraction'] < 1
        assert entropy_stats['avg_entropy_rate'] >= 0
        
        # Generate all plots
        fig1 = qa.plot_entropy_vs_event(df_loaded)
        fig2 = qa.plot_interval_histogram(df_loaded)
        fig3 = qa.plot_qq_exponential(df_loaded)
        
        # Close all figures
        plt.close('all')