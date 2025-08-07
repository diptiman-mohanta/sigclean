"""
Plotting functions for biomedical signal visualization.

This module contains functions for visualizing biomedical signals,
frequency spectra, and before/after filtering comparisons.
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as signal
from matplotlib.gridspec import GridSpec
import warnings

def plot_signal(data, fs=None, time_axis=None, title="Biomedical Signal", 
                xlabel="Time", ylabel="Amplitude", figsize=(12, 6), 
                color='blue', alpha=1.0, linewidth=1.0):
    """
    Plot biomedical signal with time axis.
    
    Parameters:
    -----------
    data : array-like
        Signal data to plot
    fs : float, optional
        Sampling frequency in Hz (used to create time axis)
    time_axis : array-like, optional
        Custom time axis (overrides fs)
    title : str
        Plot title
    xlabel : str
        X-axis label
    ylabel : str
        Y-axis label
    figsize : tuple
        Figure size (width, height)
    color : str
        Line color
    alpha : float
        Line transparency
    linewidth : float
        Line width
    
    Returns:
    --------
    fig : matplotlib.figure.Figure
        Figure object
    ax : matplotlib.axes.Axes
        Axes object
    """
    data = np.asarray(data)
    
    # Create time axis
    if time_axis is not None:
        time_axis = np.asarray(time_axis)
        if len(time_axis) != len(data):
            raise ValueError("Time axis length must match data length")
    elif fs is not None:
        time_axis = np.arange(len(data)) / fs
        xlabel = "Time (s)" if xlabel == "Time" else xlabel
    else:
        time_axis = np.arange(len(data))
        xlabel = "Samples" if xlabel == "Time" else xlabel
    
    # Create plot
    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(time_axis, data, color=color, alpha=alpha, linewidth=linewidth)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig, ax


def plot_frequency_spectrum(data, fs, method='welch', nperseg=None, 
                          title="Frequency Spectrum", figsize=(10, 6),
                          freq_range=None, log_scale=False):
    """
    Plot frequency spectrum of biomedical signal.
    
    Parameters:
    -----------
    data : array-like
        Signal data
    fs : float
        Sampling frequency in Hz
    method : str
        Method for spectrum calculation ('welch', 'periodogram', 'fft')
    nperseg : int, optional
        Length of each segment for Welch's method
    title : str
        Plot title
    figsize : tuple
        Figure size (width, height)
    freq_range : tuple, optional
        Frequency range to display (min_freq, max_freq)
    log_scale : bool
        Use logarithmic scale for magnitude
    
    Returns:
    --------
    fig : matplotlib.figure.Figure
        Figure object
    ax : matplotlib.axes.Axes
        Axes object
    freqs : ndarray
        Frequency array
    psd : ndarray
        Power spectral density
    """
    data = np.asarray(data)
    
    if method == 'welch':
        if nperseg is None:
            nperseg = min(len(data) // 8, 1024)
        freqs, psd = signal.welch(data, fs, nperseg=nperseg)
        
    elif method == 'periodogram':
        freqs, psd = signal.periodogram(data, fs)
        
    elif method == 'fft':
        fft = np.fft.fft(data)
        freqs = np.fft.fftfreq(len(data), 1/fs)
        # Take positive frequencies only
        pos_freqs = freqs[:len(freqs)//2]
        psd = np.abs(fft[:len(fft)//2])**2
        freqs = pos_freqs
        
    else:
        raise ValueError("method must be 'welch', 'periodogram', or 'fft'")
    
    # Apply frequency range filter
    if freq_range is not None:
        mask = (freqs >= freq_range[0]) & (freqs <= freq_range[1])
        freqs = freqs[mask]
        psd = psd[mask]
    
    # Create plot
    fig, ax = plt.subplots(figsize=figsize)
    
    if log_scale:
        ax.semilogy(freqs, psd)
        ax.set_ylabel("Power Spectral Density (log scale)")
    else:
        ax.plot(freqs, psd)
        ax.set_ylabel("Power Spectral Density")
    
    ax.set_xlabel("Frequency (Hz)")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig, ax, freqs, psd


def plot_before_after(original_data, processed_data, fs=None, 
                     titles=("Original Signal", "Processed Signal"),
                     figsize=(14, 8), colors=('blue', 'red')):
    """
    Plot before and after comparison of signal processing.
    
    Parameters:
    -----------
    original_data : array-like
        Original signal data
    processed_data : array-like
        Processed signal data
    fs : float, optional
        Sampling frequency in Hz
    titles : tuple
        Titles for original and processed signals
    figsize : tuple
        Figure size (width, height)
    colors : tuple
        Colors for original and processed signals
    
    Returns:
    --------
    fig : matplotlib.figure.Figure
        Figure object
    axes : tuple
        Tuple of axes objects (ax1, ax2)
    """
    original_data = np.asarray(original_data)
    processed_data = np.asarray(processed_data)
    
    if len(original_data) != len(processed_data):
        warnings.warn("Original and processed data have different lengths")
    
    # Create time axis
    if fs is not None:
        time_axis_orig = np.arange(len(original_data)) / fs
        time_axis_proc = np.arange(len(processed_data)) / fs
        xlabel = "Time (s)"
    else:
        time_axis_orig = np.arange(len(original_data))
        time_axis_proc = np.arange(len(processed_data))
        xlabel = "Samples"
    
    # Create subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize, sharex=True)
    
    # Plot original signal
    ax1.plot(time_axis_orig, original_data, color=colors[0], linewidth=1.0)
    ax1.set_ylabel("Amplitude")
    ax1.set_title(titles[0])
    ax1.grid(True, alpha=0.3)
    
    # Plot processed signal
    ax2.plot(time_axis_proc, processed_data, color=colors[1], linewidth=1.0)
    ax2.set_xlabel(xlabel)
    ax2.set_ylabel("Amplitude")
    ax2.set_title(titles[1])
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig, (ax1, ax2)


def plot_multiple_signals(signals, fs=None, labels=None, title="Multiple Signals",
                         figsize=(14, 10), colors=None, alpha=0.8):
    """
    Plot multiple signals on the same or separate axes.
    
    Parameters:
    -----------
    signals : list or ndarray
        List of signals or 2D array where each row is a signal
    fs : float, optional
        Sampling frequency in Hz
    labels : list, optional
        Labels for each signal
    title : str
        Plot title
    figsize : tuple
        Figure size (width, height)
    colors : list, optional
        Colors for each signal
    alpha : float
        Line transparency
    
    Returns:
    --------
    fig : matplotlib.figure.Figure
        Figure object
    ax : matplotlib.axes.Axes
        Axes object
    """
    if isinstance(signals, np.ndarray) and signals.ndim == 2:
        signals = [signals[i] for i in range(signals.shape[0])]
    
    n_signals = len(signals)
    
    if labels is None:
        labels = [f"Signal {i+1}" for i in range(n_signals)]
    
    if colors is None:
        colors = plt.cm.tab10(np.linspace(0, 1, n_signals))
    
    # Create time axis for the longest signal
    max_length = max(len(sig) for sig in signals)
    if fs is not None:
        time_axis = np.arange(max_length) / fs
        xlabel = "Time (s)"
    else:
        time_axis = np.arange(max_length)
        xlabel = "Samples"
    
    # Create plot
    fig, ax = plt.subplots(figsize=figsize)
    
    for i, (signal_data, label, color) in enumerate(zip(signals, labels, colors)):
        signal_data = np.asarray(signal_data)
        signal_time = time_axis[:len(signal_data)]
        ax.plot(signal_time, signal_data, label=label, color=color, 
                alpha=alpha, linewidth=1.0)
    
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Amplitude")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig, ax


def plot_spectrogram(data, fs, nperseg=256, noverlap=None, 
                    title="Spectrogram", figsize=(12, 8), cmap='viridis'):
    """
    Plot spectrogram of biomedical signal.
    
    Parameters:
    -----------
    data : array-like
        Signal data
    fs : float
        Sampling frequency in Hz
    nperseg : int
        Length of each segment
    noverlap : int, optional
        Number of points to overlap between segments
    title : str
        Plot title
    figsize : tuple
        Figure size (width, height)
    cmap : str
        Colormap for spectrogram
    
    Returns:
    --------
    fig : matplotlib.figure.Figure
        Figure object
    ax : matplotlib.axes.Axes
        Axes object
    """
    data = np.asarray(data)
    
    if noverlap is None:
        noverlap = nperseg // 8
    
    # Compute spectrogram
    freqs, times, Sxx = signal.spectrogram(data, fs, nperseg=nperseg, 
                                          noverlap=noverlap)
    
    # Create plot
    fig, ax = plt.subplots(figsize=figsize)
    
    im = ax.pcolormesh(times, freqs, 10 * np.log10(Sxx), cmap=cmap)
    ax.set_ylabel('Frequency (Hz)')
    ax.set_xlabel('Time (s)')
    ax.set_title(title)
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Power Spectral Density (dB/Hz)')
    
    plt.tight_layout()
    return fig, ax


def plot_signal_quality_assessment(data, fs, window_length=5.0, 
                                 title="Signal Quality Assessment",
                                 figsize=(15, 10)):
    """
    Create comprehensive signal quality assessment plot.
    
    Parameters:
    -----------
    data : array-like
        Signal data
    fs : float
        Sampling frequency in Hz
    window_length : float
        Length of analysis window in seconds
    title : str
        Plot title
    figsize : tuple
        Figure size (width, height)
    
    Returns:
    --------
    fig : matplotlib.figure.Figure
        Figure object
    axes : dict
        Dictionary of axes objects
    """
    data = np.asarray(data)
    time_axis = np.arange(len(data)) / fs
    
    # Create grid layout
    fig = plt.figure(figsize=figsize)
    gs = GridSpec(3, 2, figure=fig)
    
    axes = {}
    
    # 1. Original signal
    axes['signal'] = fig.add_subplot(gs[0, :])
    axes['signal'].plot(time_axis, data, 'b-', linewidth=0.8)
    axes['signal'].set_ylabel('Amplitude')
    axes['signal'].set_title('Original Signal')
    axes['signal'].grid(True, alpha=0.3)
    
    # 2. Frequency spectrum
    axes['spectrum'] = fig.add_subplot(gs[1, 0])
    freqs, psd = signal.welch(data, fs, nperseg=min(len(data)//8, 1024))
    axes['spectrum'].semilogy(freqs, psd)
    axes['spectrum'].set_xlabel('Frequency (Hz)')
    axes['spectrum'].set_ylabel('PSD (log scale)')
    axes['spectrum'].set_title('Power Spectral Density')
    axes['spectrum'].grid(True, alpha=0.3)
    
    # 3. Histogram
    axes['histogram'] = fig.add_subplot(gs[1, 1])
    axes['histogram'].hist(data, bins=50, alpha=0.7, edgecolor='black')
    axes['histogram'].set_xlabel('Amplitude')
    axes['histogram'].set_ylabel('Count')
    axes['histogram'].set_title('Amplitude Distribution')
    axes['histogram'].grid(True, alpha=0.3)
    
    # 4. Moving statistics
    axes['statistics'] = fig.add_subplot(gs[2, :])
    window_samples = int(window_length * fs)
    
    # Calculate moving mean and std
    moving_mean = []
    moving_std = []
    
    for i in range(0, len(data), window_samples//4):
        window_data = data[i:i+window_samples]
        if len(window_data) > 0:
            moving_mean.append(np.mean(window_data))
            moving_std.append(np.std(window_data))
    
    window_times = np.arange(0, len(moving_mean)) * (window_samples//4) / fs
    
    ax_twin = axes['statistics'].twinx()
    
    line1 = axes['statistics'].plot(window_times, moving_mean, 'g-', 
                                   label='Moving Mean', linewidth=2)
    line2 = ax_twin.plot(window_times, moving_std, 'r-', 
                        label='Moving Std', linewidth=2)
    
    axes['statistics'].set_xlabel('Time (s)')
    axes['statistics'].set_ylabel('Moving Mean', color='g')
    ax_twin.set_ylabel('Moving Std', color='r')
    axes['statistics'].set_title(f'Moving Statistics (Window: {window_length}s)')
    
    # Combine legends
    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    axes['statistics'].legend(lines, labels, loc='upper right')
    
    axes['statistics'].grid(True, alpha=0.3)
    
    plt.suptitle(title, fontsize=16, y=0.98)
    plt.tight_layout()
    
    return fig, axes


def plot_filter_response(b, a, fs, title="Filter Response", figsize=(12, 6)):
    """
    Plot frequency response of digital filter.
    
    Parameters:
    -----------
    b : array-like
        Numerator coefficients
    a : array-like
        Denominator coefficients
    fs : float
        Sampling frequency in Hz
    title : str
        Plot title
    figsize : tuple
        Figure size (width, height)
    
    Returns:
    --------
    fig : matplotlib.figure.Figure
        Figure object
    axes : tuple
        Tuple of axes objects (magnitude, phase)
    """
    # Calculate frequency response
    w, h = signal.freqz(b, a, fs=fs)
    
    # Create subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize)
    
    # Magnitude response
    ax1.plot(w, 20 * np.log10(abs(h)))
    ax1.set_ylabel('Magnitude (dB)')
    ax1.set_title(f'{title} - Magnitude Response')
    ax1.grid(True, alpha=0.3)
    
    # Phase response
    ax2.plot(w, np.unwrap(np.angle(h)) * 180 / np.pi)
    ax2.set_xlabel('Frequency (Hz)')
    ax2.set_ylabel('Phase (degrees)')
    ax2.set_title('Phase Response')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig, (ax1, ax2)