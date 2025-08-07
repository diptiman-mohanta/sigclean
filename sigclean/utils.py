"""
Utility functions for biomedical signal processing.

This module contains utility functions for signal preprocessing,
normalization, resampling, segmentation, and quality assessment.
"""

import numpy as np
import scipy.signal as signal
from scipy.interpolate import interp1d
from scipy.stats import zscore
from scipy.signal import find_peaks
import warnings

def normalize_signal(data, method='zscore', feature_range=(0, 1)):
    """
    Normalize biomedical signal using various methods.
    
    Parameters:
    -----------
    data : array-like
        Input signal data
    method : str
        Normalization method ('zscore', 'minmax', 'robust', 'unit_vector')
    feature_range : tuple
        Target range for minmax normalization (default: (0, 1))
    
    Returns:
    --------
    normalized_data : ndarray
        Normalized signal
    norm_params : dict
        Parameters used for normalization (for inverse transform)
    """
    data = np.asarray(data)
    
    if method == 'zscore':
        mean_val = np.mean(data)
        std_val = np.std(data)
        if std_val == 0:
            warnings.warn("Standard deviation is zero, returning zero-centered data")
            normalized_data = data - mean_val
        else:
            normalized_data = (data - mean_val) / std_val
        norm_params = {'mean': mean_val, 'std': std_val}
        
    elif method == 'minmax':
        min_val = np.min(data)
        max_val = np.max(data)
        if min_val == max_val:
            warnings.warn("Data has no variance, returning constant array")
            normalized_data = np.full_like(data, feature_range[0])
        else:
            normalized_data = (data - min_val) / (max_val - min_val)
            normalized_data = normalized_data * (feature_range[1] - feature_range[0]) + feature_range[0]
        norm_params = {'min': min_val, 'max': max_val, 'feature_range': feature_range}
        
    elif method == 'robust':
        median_val = np.median(data)
        mad = np.median(np.abs(data - median_val))  # Median Absolute Deviation
        if mad == 0:
            warnings.warn("MAD is zero, using standard deviation instead")
            mad = np.std(data)
        normalized_data = (data - median_val) / mad
        norm_params = {'median': median_val, 'mad': mad}
        
    elif method == 'unit_vector':
        norm = np.linalg.norm(data)
        if norm == 0:
            warnings.warn("Vector norm is zero, returning original data")
            normalized_data = data
        else:
            normalized_data = data / norm
        norm_params = {'norm': norm}
        
    else:
        raise ValueError("method must be 'zscore', 'minmax', 'robust', or 'unit_vector'")
    
    return normalized_data, norm_params


def resample_signal(data, original_fs, target_fs, method='scipy'):
    """
    Resample biomedical signal to different sampling frequency.
    
    Parameters:
    -----------
    data : array-like
        Input signal data
    original_fs : float
        Original sampling frequency in Hz
    target_fs : float
        Target sampling frequency in Hz
    method : str
        Resampling method ('scipy', 'interpolation')
    
    Returns:
    --------
    resampled_data : ndarray
        Resampled signal
    new_time_axis : ndarray
        New time axis for resampled signal
    """
    data = np.asarray(data)
    
    if original_fs == target_fs:
        time_axis = np.arange(len(data)) / original_fs
        return data, time_axis
    
    original_time = np.arange(len(data)) / original_fs
    
    if method == 'scipy':
        # Calculate number of samples in resampled signal
        num_samples = int(len(data) * target_fs / original_fs)
        
        # Use scipy's resample function
        resampled_data = signal.resample(data, num_samples)
        new_time_axis = np.linspace(0, original_time[-1], num_samples)
        
    elif method == 'interpolation':
        # Create new time axis
        new_time_axis = np.arange(0, original_time[-1], 1/target_fs)
        
        # Interpolate
        f = interp1d(original_time, data, kind='linear', 
                    bounds_error=False, fill_value='extrapolate')
        resampled_data = f(new_time_axis)
        
    else:
        raise ValueError("method must be 'scipy' or 'interpolation'")
    
    return resampled_data, new_time_axis


def segment_signal(data, fs, segment_length, overlap=0.0, window='hann'):
    """
    Segment signal into overlapping windows.
    
    Parameters:
    -----------
    data : array-like
        Input signal data
    fs : float
        Sampling frequency in Hz
    segment_length : float
        Length of each segment in seconds
    overlap : float
        Overlap between segments (0.0 to 1.0)
    window : str
        Window function to apply ('hann', 'hamming', 'blackman', 'rect')
    
    Returns:
    --------
    segments : ndarray
        2D array where each row is a segment
    segment_times : ndarray
        Start times for each segment
    """
    data = np.asarray(data)
    segment_samples = int(segment_length * fs)
    overlap_samples = int(segment_samples * overlap)
    step_size = segment_samples - overlap_samples
    
    if segment_samples > len(data):
        raise ValueError("Segment length is longer than signal")
    
    # Calculate number of segments
    num_segments = (len(data) - segment_samples) // step_size + 1
    
    # Initialize segments array
    segments = np.zeros((num_segments, segment_samples))
    segment_times = np.zeros(num_segments)
    
    # Create window function
    if window == 'hann':
        win = np.hanning(segment_samples)
    elif window == 'hamming':
        win = np.hamming(segment_samples)
    elif window == 'blackman':
        win = np.blackman(segment_samples)
    elif window == 'rect':
        win = np.ones(segment_samples)
    else:
        raise ValueError("window must be 'hann', 'hamming', 'blackman', or 'rect'")
    
    # Extract segments
    for i in range(num_segments):
        start_idx = i * step_size
        end_idx = start_idx + segment_samples
        segments[i] = data[start_idx:end_idx] * win
        segment_times[i] = start_idx / fs
    
    return segments, segment_times


def calculate_snr(signal_data, noise_data=None, method='power'):
    """
    Calculate Signal-to-Noise Ratio (SNR).
    
    Parameters:
    -----------
    signal_data : array-like
        Clean signal or signal+noise
    noise_data : array-like, optional
        Noise signal (if None, estimated from signal)
    method : str
        Method for SNR calculation ('power', 'amplitude')
    
    Returns:
    --------
    snr_db : float
        Signal-to-Noise Ratio in decibels
    """
    signal_data = np.asarray(signal_data)
    
    if noise_data is None:
        # Estimate noise as high-frequency components
        noise_data = signal_data - signal.medfilt(signal_data, kernel_size=5)
    else:
        noise_data = np.asarray(noise_data)
    
    if method == 'power':
        signal_power = np.mean(signal_data ** 2)
        noise_power = np.mean(noise_data ** 2)
        
        if noise_power == 0:
            return float('inf')
        
        snr_linear = signal_power / noise_power
        
    elif method == 'amplitude':
        signal_rms = np.sqrt(np.mean(signal_data ** 2))
        noise_rms = np.sqrt(np.mean(noise_data ** 2))
        
        if noise_rms == 0:
            return float('inf')
        
        snr_linear = signal_rms / noise_rms
        
    else:
        raise ValueError("method must be 'power' or 'amplitude'")
    
    snr_db = 10 * np.log10(snr_linear)
    
    return snr_db


def detect_peaks(data, fs, min_distance=None, height=None, prominence=None):
    """
    Detect peaks in biomedical signals (e.g., R-peaks in ECG).
    
    Parameters:
    -----------
    data : array-like
        Input signal data
    fs : float
        Sampling frequency in Hz
    min_distance : float
        Minimum distance between peaks in seconds
    height : float
        Minimum height of peaks
    prominence : float
        Minimum prominence of peaks
    
    Returns:
    --------
    peak_indices : ndarray
        Indices of detected peaks
    peak_properties : dict
        Properties of detected peaks
    """
    data = np.asarray(data)
    
    # Convert min_distance from seconds to samples
    if min_distance is not None:
        min_distance_samples = int(min_distance * fs)
    else:
        min_distance_samples = None
    
    # Find peaks
    peak_indices, properties = find_peaks(
        data, 
        distance=min_distance_samples,
        height=height,
        prominence=prominence
    )
    
    # Convert peak times to seconds
    peak_times = peak_indices / fs
    properties['peak_times'] = peak_times
    
    return peak_indices, properties


def remove_outliers(data, method='iqr', threshold=1.5):
    """
    Remove outliers from signal data.
    
    Parameters:
    -----------
    data : array-like
        Input signal data
    method : str
        Method for outlier detection ('iqr', 'zscore', 'mad')
    threshold : float
        Threshold for outlier detection
    
    Returns:
    --------
    cleaned_data : ndarray
        Data with outliers removed/replaced
    outlier_mask : ndarray
        Boolean mask indicating outlier locations
    """
    data = np.asarray(data)
    outlier_mask = np.zeros(len(data), dtype=bool)
    
    if method == 'iqr':
        Q1 = np.percentile(data, 25)
        Q3 = np.percentile(data, 75)
        IQR = Q3 - Q1
        lower_bound = Q1 - threshold * IQR
        upper_bound = Q3 + threshold * IQR
        outlier_mask = (data < lower_bound) | (data > upper_bound)
        
    elif method == 'zscore':
        z_scores = np.abs(zscore(data))
        outlier_mask = z_scores > threshold
        
    elif method == 'mad':
        median_val = np.median(data)
        mad = np.median(np.abs(data - median_val))
        modified_z_scores = 0.6745 * (data - median_val) / mad
        outlier_mask = np.abs(modified_z_scores) > threshold
        
    else:
        raise ValueError("method must be 'iqr', 'zscore', or 'mad'")
    
    # Replace outliers with interpolated values
    cleaned_data = data.copy()
    if np.any(outlier_mask):
        outlier_indices = np.where(outlier_mask)[0]
        clean_indices = np.where(~outlier_mask)[0]
        
        if len(clean_indices) > 1:
            cleaned_data[outlier_indices] = np.interp(
                outlier_indices, clean_indices, data[clean_indices]
            )
    
    return cleaned_data, outlier_mask


def calculate_signal_quality(data, fs, window_length=5.0):
    """
    Assess signal quality using multiple metrics.
    
    Parameters:
    -----------
    data : array-like
        Input signal data
    fs : float
        Sampling frequency in Hz
    window_length : float
        Length of analysis window in seconds
    
    Returns:
    --------
    quality_metrics : dict
        Dictionary containing quality assessment metrics
    """
    data = np.asarray(data)
    window_samples = int(window_length * fs)
    
    # Initialize metrics
    quality_metrics = {}
    
    # 1. Signal-to-Noise Ratio
    try:
        snr = calculate_snr(data)
        quality_metrics['snr_db'] = snr
    except:
        quality_metrics['snr_db'] = None
    
    # 2. Baseline stability (standard deviation of low-frequency component)
    try:
        baseline = signal.medfilt(data, kernel_size=min(window_samples, len(data)//10))
        quality_metrics['baseline_stability'] = np.std(baseline)
    except:
        quality_metrics['baseline_stability'] = None
    
    # 3. Artifact percentage
    try:
        _, artifact_mask = remove_outliers(data, method='iqr', threshold=2.0)
        quality_metrics['artifact_percentage'] = np.mean(artifact_mask) * 100
    except:
        quality_metrics['artifact_percentage'] = None
    
    # 4. Frequency domain metrics
    try:
        freqs, psd = signal.welch(data, fs, nperseg=min(window_samples, len(data)//4))
        quality_metrics['peak_frequency'] = freqs[np.argmax(psd)]
        quality_metrics['spectral_centroid'] = np.sum(freqs * psd) / np.sum(psd)
    except:
        quality_metrics['peak_frequency'] = None
        quality_metrics['spectral_centroid'] = None
    
    # 5. Statistical metrics
    quality_metrics['mean'] = np.mean(data)
    quality_metrics['std'] = np.std(data)
    quality_metrics['skewness'] = np.mean(((data - np.mean(data)) / np.std(data)) ** 3)
    quality_metrics['kurtosis'] = np.mean(((data - np.mean(data)) / np.std(data)) ** 4) - 3
    
    return quality_metrics


def apply_window(data, window_type='hann'):
    """
    Apply window function to signal data.
    
    Parameters:
    -----------
    data : array-like
        Input signal data
    window_type : str
        Type of window ('hann', 'hamming', 'blackman', 'tukey', 'rect')
    
    Returns:
    --------
    windowed_data : ndarray
        Signal with window function applied
    """
    data = np.asarray(data)
    
    if window_type == 'hann':
        window = np.hanning(len(data))
    elif window_type == 'hamming':
        window = np.hamming(len(data))
    elif window_type == 'blackman':
        window = np.blackman(len(data))
    elif window_type == 'tukey':
        window = signal.windows.tukey(len(data), alpha=0.5)
    elif window_type == 'rect':
        window = np.ones(len(data))
    else:
        raise ValueError("window_type must be 'hann', 'hamming', 'blackman', 'tukey', or 'rect'")
    
    windowed_data = data * window
    
    return windowed_data
