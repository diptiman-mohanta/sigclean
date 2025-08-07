"""
Signal filtering functions for biomedical signal processing.

This module contains various filtering techniques commonly used in
biomedical signal processing including bandpass filters, notch filters,
baseline removal, and artifact removal.
"""

import numpy as np
import scipy.signal as signal
from scipy.signal import butter, filtfilt, iirnotch, find_peaks
from scipy.ndimage import median_filter
import warnings

def butterworth_filter(data, fs, filter_type='bandpass', low_freq=None, 
                      high_freq=None, order=4):
    """
    Apply Butterworth filter to biomedical signal.
    
    Parameters:
    -----------
    data : array-like
        Input signal data
    fs : float
        Sampling frequency in Hz
    filter_type : str
        Type of filter ('lowpass', 'highpass', 'bandpass', 'bandstop')
    low_freq : float
        Low cutoff frequency in Hz (for highpass and bandpass)
    high_freq : float
        High cutoff frequency in Hz (for lowpass and bandpass)
    order : int
        Filter order (default: 4)
    
    Returns:
    --------
    filtered_data : ndarray
        Filtered signal
    """
    nyquist = fs / 2
    
    if filter_type == 'lowpass':
        if high_freq is None:
            raise ValueError("high_freq must be specified for lowpass filter")
        if high_freq >= nyquist:
            warnings.warn(f"Cutoff frequency {high_freq} Hz is too high for sampling rate {fs} Hz")
            high_freq = nyquist * 0.9
        
        b, a = butter(order, high_freq / nyquist, btype='low')
        
    elif filter_type == 'highpass':
        if low_freq is None:
            raise ValueError("low_freq must be specified for highpass filter")
        if low_freq >= nyquist:
            raise ValueError("Cutoff frequency cannot exceed Nyquist frequency")
        
        b, a = butter(order, low_freq / nyquist, btype='high')
        
    elif filter_type == 'bandpass':
        if low_freq is None or high_freq is None:
            raise ValueError("Both low_freq and high_freq must be specified for bandpass filter")
        if low_freq >= high_freq:
            raise ValueError("low_freq must be less than high_freq")
        if high_freq >= nyquist:
            warnings.warn(f"High cutoff frequency {high_freq} Hz is too high for sampling rate {fs} Hz")
            high_freq = nyquist * 0.9
            
        b, a = butter(order, [low_freq / nyquist, high_freq / nyquist], btype='band')
        
    elif filter_type == 'bandstop':
        if low_freq is None or high_freq is None:
            raise ValueError("Both low_freq and high_freq must be specified for bandstop filter")
        if low_freq >= high_freq:
            raise ValueError("low_freq must be less than high_freq")
            
        b, a = butter(order, [low_freq / nyquist, high_freq / nyquist], btype='bandstop')
    
    else:
        raise ValueError("filter_type must be 'lowpass', 'highpass', 'bandpass', or 'bandstop'")
    
    # Apply zero-phase filtering
    filtered_data = filtfilt(b, a, data)
    
    return filtered_data


def notch_filter(data, fs, freq=50.0, quality_factor=30):
    """
    Apply notch filter to remove power line interference.
    
    Parameters:
    -----------
    data : array-like
        Input signal data
    fs : float
        Sampling frequency in Hz
    freq : float
        Frequency to remove (default: 50 Hz for EU, use 60 Hz for US)
    quality_factor : float
        Quality factor (higher values = narrower notch)
    
    Returns:
    --------
    filtered_data : ndarray
        Filtered signal with power line interference removed
    """
    nyquist = fs / 2
    
    if freq >= nyquist:
        warnings.warn(f"Notch frequency {freq} Hz is too high for sampling rate {fs} Hz")
        return data
    
    # Design notch filter
    b, a = iirnotch(freq, quality_factor, fs)
    
    # Apply filter
    filtered_data = filtfilt(b, a, data)
    
    return filtered_data


def baseline_removal(data, method='detrend', window_size=None, polynomial_order=3):
    """
    Remove baseline drift from biomedical signals.
    
    Parameters:
    -----------
    data : array-like
        Input signal data
    method : str
        Method for baseline removal ('detrend', 'median', 'polynomial')
    window_size : int
        Window size for median filtering (only for 'median' method)
    polynomial_order : int
        Order of polynomial for detrending (only for 'polynomial' method)
    
    Returns:
    --------
    corrected_data : ndarray
        Signal with baseline removed
    """
    data = np.asarray(data)
    
    if method == 'detrend':
        # Simple linear detrending
        corrected_data = signal.detrend(data, type='linear')
        
    elif method == 'median':
        if window_size is None:
            window_size = len(data) // 10  # Default to 10% of signal length
        
        # Median filtering for baseline estimation
        baseline = median_filter(data, size=window_size)
        corrected_data = data - baseline
        
    elif method == 'polynomial':
        # Polynomial detrending
        x = np.arange(len(data))
        coeffs = np.polyfit(x, data, polynomial_order)
        baseline = np.polyval(coeffs, x)
        corrected_data = data - baseline
        
    else:
        raise ValueError("method must be 'detrend', 'median', or 'polynomial'")
    
    return corrected_data


def artifact_removal(data, fs, threshold_std=3.0, window_size=None):
    """
    Remove artifacts from biomedical signals using statistical thresholding.
    
    Parameters:
    -----------
    data : array-like
        Input signal data
    fs : float
        Sampling frequency in Hz
    threshold_std : float
        Number of standard deviations for artifact detection threshold
    window_size : int
        Window size for local artifact detection (default: 1 second)
    
    Returns:
    --------
    cleaned_data : ndarray
        Signal with artifacts removed/interpolated
    artifact_mask : ndarray
        Boolean mask indicating artifact locations
    """
    data = np.asarray(data)
    
    if window_size is None:
        window_size = int(fs)  # 1 second window
    
    cleaned_data = data.copy()
    artifact_mask = np.zeros(len(data), dtype=bool)
    
    # Calculate local statistics
    half_window = window_size // 2
    
    for i in range(len(data)):
        start_idx = max(0, i - half_window)
        end_idx = min(len(data), i + half_window + 1)
        local_window = data[start_idx:end_idx]
        
        local_mean = np.mean(local_window)
        local_std = np.std(local_window)
        
        # Check if current sample is an artifact
        if abs(data[i] - local_mean) > threshold_std * local_std:
            artifact_mask[i] = True
    
    # Interpolate artifacts
    if np.any(artifact_mask):
        artifact_indices = np.where(artifact_mask)[0]
        clean_indices = np.where(~artifact_mask)[0]
        
        if len(clean_indices) > 1:
            cleaned_data[artifact_indices] = np.interp(
                artifact_indices, clean_indices, data[clean_indices]
            )
    
    return cleaned_data, artifact_mask


def adaptive_filter(data, reference_signal=None, filter_length=32, mu=0.01):
    """
    Apply adaptive filtering for noise reduction.
    
    Parameters:
    -----------
    data : array-like
        Primary input signal (signal + noise)
    reference_signal : array-like
        Reference noise signal (optional)
    filter_length : int
        Length of adaptive filter
    mu : float
        Step size parameter (learning rate)
    
    Returns:
    --------
    filtered_signal : ndarray
        Adaptively filtered signal
    error_signal : ndarray
        Error signal (filtered output)
    """
    data = np.asarray(data)
    
    if reference_signal is None:
        # Create delayed version as reference for self-adaptive filtering
        reference_signal = np.roll(data, 1)
        reference_signal[0] = 0
    else:
        reference_signal = np.asarray(reference_signal)
        if len(reference_signal) != len(data):
            raise ValueError("Reference signal must have same length as input data")
    
    # Initialize filter weights
    weights = np.zeros(filter_length)
    filtered_signal = np.zeros(len(data))
    error_signal = np.zeros(len(data))
    
    # LMS adaptive filtering
    for n in range(filter_length, len(data)):
        # Extract reference signal segment
        x = reference_signal[n-filter_length:n][::-1]  # Reverse for convolution
        
        # Filter output
        y = np.dot(weights, x)
        filtered_signal[n] = y
        
        # Error signal
        error = data[n] - y
        error_signal[n] = error
        
        # Update weights
        weights += mu * error * x
    
    return filtered_signal, error_signal


def moving_average_filter(data, window_size):
    """
    Apply moving average filter for smoothing.
    
    Parameters:
    -----------
    data : array-like
        Input signal data
    window_size : int
        Size of moving average window
    
    Returns:
    --------
    smoothed_data : ndarray
        Smoothed signal
    """
    if window_size < 1:
        raise ValueError("Window size must be at least 1")
    
    if window_size >= len(data):
        return np.full_like(data, np.mean(data))
    
    # Create convolution kernel
    kernel = np.ones(window_size) / window_size
    
    # Apply convolution with same mode to preserve length
    smoothed_data = np.convolve(data, kernel, mode='same')
    
    return smoothed_data


def savitzky_golay_filter(data, window_length, polyorder=3):
    """
    Apply Savitzky-Golay filter for signal smoothing while preserving features.
    
    Parameters:
    -----------
    data : array-like
        Input signal data
    window_length : int
        Length of filter window (must be odd)
    polyorder : int
        Order of polynomial used for fitting
    
    Returns:
    --------
    filtered_data : ndarray
        Filtered signal
    """
    if window_length % 2 == 0:
        window_length += 1  # Ensure odd window length
    
    if window_length < polyorder + 1:
        raise ValueError("window_length must be greater than polyorder")
    
    filtered_data = signal.savgol_filter(data, window_length, polyorder)
    
    return filtered_data