"""
QFTT Analysis Module

Analysis and visualization utilities for QFTT simulation outputs.

© 2024 - MIT License
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from scipy.optimize import curve_fit
from typing import Optional, Tuple, Dict, Union
import seaborn as sns
from pathlib import Path


# Set default plot style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")


def load_events(filepath: Union[str, Path]) -> pd.DataFrame:
    """
    Load event data from CSV or HDF5 file.
    
    Parameters
    ----------
    filepath : str or Path
        Path to the event data file
        
    Returns
    -------
    pd.DataFrame
        Event data
    """
    filepath = Path(filepath)
    
    if filepath.suffix == '.csv':
        return pd.read_csv(filepath)
    elif filepath.suffix in ['.h5', '.hdf5']:
        return pd.read_hdf(filepath, key='events')
    else:
        raise ValueError(f"Unsupported file format: {filepath.suffix}")


def plot_entropy_vs_event(df: pd.DataFrame, 
                         save_path: Optional[str] = None,
                         figsize: Tuple[float, float] = (10, 6)) -> plt.Figure:
    """
    Plot system entropy as a function of event number.
    
    Parameters
    ----------
    df : pd.DataFrame
        Event data with 'event_id' and 'entropy' columns
    save_path : str, optional
        Path to save figure
    figsize : tuple
        Figure size (width, height)
        
    Returns
    -------
    matplotlib.figure.Figure
        The figure object
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    ax.plot(df['event_id'], df['entropy'], 'b-', alpha=0.8, linewidth=2)
    ax.scatter(df['event_id'], df['entropy'], c='darkblue', s=20, alpha=0.5)
    
    ax.set_xlabel('Event Number', fontsize=12)
    ax.set_ylabel('System Entropy (bits)', fontsize=12)
    ax.set_title('System Entropy Growth Over Collapse Events', fontsize=14, fontweight='bold')
    
    # Add theoretical maximum line
    ax.axhline(y=1.0, color='red', linestyle='--', alpha=0.5, label='Max entropy (1 bit)')
    
    # Add trend line
    z = np.polyfit(df['event_id'], df['entropy'], 2)
    p = np.poly1d(z)
    ax.plot(df['event_id'], p(df['event_id']), "r--", alpha=0.5, label='Trend')
    
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        
    return fig


def plot_interval_histogram(df: pd.DataFrame, 
                           n_bins: int = 50,
                           fit_exponential: bool = True,
                           save_path: Optional[str] = None,
                           figsize: Tuple[float, float] = (10, 6)) -> plt.Figure:
    """
    Plot histogram of time intervals between collapse events.
    
    Parameters
    ----------
    df : pd.DataFrame
        Event data with 'delta_t' column
    n_bins : int
        Number of histogram bins
    fit_exponential : bool
        Whether to fit and overlay exponential distribution
    save_path : str, optional
        Path to save figure
    figsize : tuple
        Figure size
        
    Returns
    -------
    matplotlib.figure.Figure
        The figure object
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Get delta_t values (excluding first event which has no previous)
    delta_t = df['delta_t'][df['event_id'] > 1].values
    
    # Plot histogram
    counts, bins, patches = ax.hist(delta_t, bins=n_bins, density=True, 
                                   alpha=0.7, color='blue', edgecolor='black')
    
    if fit_exponential:
        # Fit exponential distribution
        mean_dt = np.mean(delta_t)
        lambda_fit = 1.0 / mean_dt
        
        # Overlay exponential PDF
        x = np.linspace(0, np.max(delta_t), 1000)
        y = lambda_fit * np.exp(-lambda_fit * x)
        ax.plot(x, y, 'r--', linewidth=2, 
                label=f'Exponential fit (λ={lambda_fit:.3f})')
        
    ax.set_xlabel('Time Interval Δt', fontsize=12)
    ax.set_ylabel('Probability Density', fontsize=12)
    ax.set_title('Distribution of Time Intervals Between Collapse Events', 
                 fontsize=14, fontweight='bold')
    
    # Add statistics text
    textstr = f'Mean: {np.mean(delta_t):.3f}\nStd: {np.std(delta_t):.3f}'
    ax.text(0.7, 0.9, textstr, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    if fit_exponential:
        ax.legend()
        
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        
    return fig


def autocorrelation(data: np.ndarray, max_lag: Optional[int] = None) -> np.ndarray:
    """
    Compute autocorrelation function for a time series.
    
    Parameters
    ----------
    data : np.ndarray
        Time series data
    max_lag : int, optional
        Maximum lag to compute (default: len(data)//4)
        
    Returns
    -------
    np.ndarray
        Autocorrelation values for each lag
    """
    if max_lag is None:
        max_lag = len(data) // 4
        
    data = data - np.mean(data)
    autocorr = np.correlate(data, data, mode='full')
    autocorr = autocorr[len(autocorr)//2:]
    autocorr = autocorr / autocorr[0]  # Normalize
    
    return autocorr[:max_lag]


def plot_qq_exponential(df: pd.DataFrame, 
                       save_path: Optional[str] = None,
                       figsize: Tuple[float, float] = (8, 8)) -> plt.Figure:
    """
    Create Q-Q plot comparing delta_t distribution to exponential.
    
    Parameters
    ----------
    df : pd.DataFrame
        Event data with 'delta_t' column
    save_path : str, optional
        Path to save figure
    figsize : tuple
        Figure size
        
    Returns
    -------
    matplotlib.figure.Figure
        The figure object
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Get delta_t values
    delta_t = df['delta_t'][df['event_id'] > 1].values
    
    # Theoretical exponential quantiles
    mean_dt = np.mean(delta_t)
    
    # Q-Q plot
    stats.probplot(delta_t, dist=stats.expon(scale=mean_dt), plot=ax)
    
    ax.set_title('Q-Q Plot: Δt vs Exponential Distribution', fontsize=14, fontweight='bold')
    ax.set_xlabel('Theoretical Quantiles', fontsize=12)
    ax.set_ylabel('Sample Quantiles', fontsize=12)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        
    return fig


def plot_heatmap(param_grid: np.ndarray, 
                results: np.ndarray,
                x_label: str,
                y_label: str,
                title: str,
                cmap: str = 'viridis',
                save_path: Optional[str] = None,
                figsize: Tuple[float, float] = (10, 8)) -> plt.Figure:
    """
    Plot 2D heatmap of parameter sweep results.
    
    Parameters
    ----------
    param_grid : np.ndarray
        2D array of parameter combinations
    results : np.ndarray
        2D array of results
    x_label, y_label : str
        Axis labels
    title : str
        Plot title
    cmap : str
        Colormap name
    save_path : str, optional
        Path to save figure
    figsize : tuple
        Figure size
        
    Returns
    -------
    matplotlib.figure.Figure
        The figure object
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    im = ax.imshow(results, cmap=cmap, aspect='auto', origin='lower')
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Value', fontsize=12)
    
    ax.set_xlabel(x_label, fontsize=12)
    ax.set_ylabel(y_label, fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        
    return fig


def analyze_eigentime_distribution(df: pd.DataFrame) -> Dict:
    """
    Analyze the distribution of eigentime values (+1/-1).
    
    Parameters
    ----------
    df : pd.DataFrame
        Event data with 'eigentime' column
        
    Returns
    -------
    dict
        Statistics about eigentime distribution
    """
    plus_count = (df['eigentime'] == 1).sum()
    minus_count = (df['eigentime'] == -1).sum()
    total = len(df)
    
    # Binomial test for fairness
    p_value = stats.binom_test(plus_count, total, p=0.5, alternative='two-sided')
    
    # Run lengths
    eigentime_seq = df['eigentime'].values
    run_lengths = []
    current_run = 1
    
    for i in range(1, len(eigentime_seq)):
        if eigentime_seq[i] == eigentime_seq[i-1]:
            current_run += 1
        else:
            run_lengths.append(current_run)
            current_run = 1
    run_lengths.append(current_run)
    
    return {
        'plus_count': plus_count,
        'minus_count': minus_count,
        'plus_fraction': plus_count / total,
        'minus_fraction': minus_count / total,
        'binomial_p_value': p_value,
        'is_fair': p_value > 0.05,
        'mean_run_length': np.mean(run_lengths),
        'max_run_length': np.max(run_lengths),
        'num_runs': len(run_lengths)
    }


def get_interval_stats(df: pd.DataFrame) -> Dict:
    """
    Compute statistics for collapse intervals.
    
    Parameters
    ----------
    df : pd.DataFrame
        Event data with 'delta_t' column
        
    Returns
    -------
    dict
        Interval statistics
    """
    delta_t = df['delta_t'][df['event_id'] > 1].values
    
    # Fit exponential
    mean_dt = np.mean(delta_t)
    lambda_mle = 1.0 / mean_dt
    
    # Kolmogorov-Smirnov test
    ks_stat, ks_pvalue = stats.kstest(delta_t, 
                                      lambda x: stats.expon.cdf(x, scale=mean_dt))
    
    return {
        'mean': mean_dt,
        'std': np.std(delta_t),
        'min': np.min(delta_t),
        'max': np.max(delta_t),
        'median': np.median(delta_t),
        'lambda_mle': lambda_mle,
        'ks_statistic': ks_stat,
        'ks_pvalue': ks_pvalue,
        'is_exponential': ks_pvalue > 0.05
    }


def plot_time_series(df: pd.DataFrame, 
                    quantities: list = ['entropy', 'energy'],
                    save_path: Optional[str] = None,
                    figsize: Tuple[float, float] = (12, 8)) -> plt.Figure:
    """
    Plot time series of multiple quantities.
    
    Parameters
    ----------
    df : pd.DataFrame
        Event data
    quantities : list
        Column names to plot
    save_path : str, optional
        Path to save figure
    figsize : tuple
        Figure size
        
    Returns
    -------
    matplotlib.figure.Figure
        The figure object
    """
    n_plots = len(quantities)
    fig, axes = plt.subplots(n_plots, 1, figsize=figsize, sharex=True)
    
    if n_plots == 1:
        axes = [axes]
    
    for i, qty in enumerate(quantities):
        ax = axes[i]
        ax.plot(df['event_id'], df[qty], 'b-', alpha=0.8)
        ax.set_ylabel(qty.capitalize(), fontsize=12)
        ax.grid(True, alpha=0.3)
        
    axes[-1].set_xlabel('Event Number', fontsize=12)
    fig.suptitle('Time Series of System Quantities', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        
    return fig


def entropy_rate_analysis(df: pd.DataFrame) -> Dict:
    """
    Analyze the rate of entropy increase.
    
    Parameters
    ----------
    df : pd.DataFrame
        Event data
        
    Returns
    -------
    dict
        Entropy rate statistics
    """
    # Entropy differences
    entropy_diff = np.diff(df['entropy'].values)
    
    # Cumulative time
    cumulative_time = np.cumsum(df['delta_t'].values)
    
    # Average entropy rate (bits per unit time)
    total_entropy_gain = df['entropy'].iloc[-1] - df['entropy'].iloc[0]
    total_time = cumulative_time[-1]
    avg_rate = total_entropy_gain / total_time if total_time > 0 else 0
    
    # Entropy per event
    entropy_per_event = total_entropy_gain / len(df) if len(df) > 0 else 0
    
    return {
        'total_entropy_gain': total_entropy_gain,
        'total_time': total_time,
        'avg_entropy_rate': avg_rate,
        'entropy_per_event': entropy_per_event,
        'mean_entropy_step': np.mean(entropy_diff[entropy_diff > 0]),
        'fraction_increasing': np.sum(entropy_diff > 0) / len(entropy_diff)
    }