"""
Utility functions for biomedical signal processing.

This module contains utility functions for signal preprocessing,
normalization, resampling, segmentation, and quality assessment.
"""

from __future__ import annotations

import warnings
from typing import Any, Dict, Optional, Tuple

import numpy as np
import scipy.signal as signal
from numpy.typing import ArrayLike, NDArray
from scipy.interpolate import interp1d
from scipy.signal import find_peaks
from scipy.stats import zscore


def normalize_signal(
    data: ArrayLike,
    method: str = "zscore",
    feature_range: Tuple[float, float] = (0, 1),
) -> Tuple[NDArray[np.float64], Dict[str, Any]]:
    """
    Normalize biomedical signal using various methods.

    Parameters
    -----------
    data : array-like
        Input signal data
    method : str
        Normalization method ('zscore', 'minmax', 'robust', 'unit_vector')
    feature_range : tuple
        Target range for minmax normalization (default: (0, 1))

    Returns
    --------
    normalized_data : ndarray
        Normalized signal
    norm_params : dict
        Parameters used for normalization (for inverse transform)
    """
    data = np.asarray(data, dtype=np.float64)

    if method == "zscore":
        mean_val = np.mean(data)
        std_val = np.std(data)
        if std_val == 0:
            warnings.warn("Standard deviation is zero, returning zero-centered data")
            normalized_data = data - mean_val
        else:
            normalized_data = (data - mean_val) / std_val
        norm_params = {"mean": mean_val, "std": std_val}

    elif method == "minmax":
        min_val = np.min(data)
        max_val = np.max(data)
        if min_val == max_val:
            warnings.warn("Data has no variance, returning constant array")
            normalized_data = np.full_like(data, feature_range[0])
        else:
            normalized_data = (data - min_val) / (max_val - min_val)
            normalized_data = (
                normalized_data * (feature_range[1] - feature_range[0])
                + feature_range[0]
            )
        norm_params = {"min": min_val, "max": max_val, "feature_range": feature_range}

    elif method == "robust":
        median_val = np.median(data)
        mad = np.median(np.abs(data - median_val))  # Median Absolute Deviation
        if mad == 0:
            warnings.warn("MAD is zero, using standard deviation instead")
            mad = np.std(data)
        # Scale by the consistency constant (1/0.6745) so the result is on
        # the same footing as a standard z-score for normally-distributed
        # data, and consistent with the MAD scaling used in remove_outliers.
        scaled_mad = mad / 0.6745
        normalized_data = (data - median_val) / scaled_mad
        norm_params = {"median": median_val, "mad": mad, "scaled_mad": scaled_mad}

    elif method == "unit_vector":
        norm = np.linalg.norm(data)
        if norm == 0:
            warnings.warn("Vector norm is zero, returning original data")
            normalized_data = data
        else:
            normalized_data = data / norm
        norm_params = {"norm": norm}

    else:
        raise ValueError(
            "method must be 'zscore', 'minmax', 'robust', or 'unit_vector'"
        )

    return normalized_data, norm_params


def resample_signal(
    data: ArrayLike,
    original_fs: float,
    target_fs: float,
    method: str = "scipy",
) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
    """
    Resample biomedical signal to different sampling frequency.

    Parameters
    -----------
    data : array-like
        Input signal data
    original_fs : float
        Original sampling frequency in Hz
    target_fs : float
        Target sampling frequency in Hz
    method : str
        Resampling method ('scipy', 'interpolation'). 'scipy' uses an
        FFT-based resample, which assumes the signal is periodic and can
        introduce edge (Gibbs) artifacts on non-periodic real-world signals;
        prefer 'interpolation', or pre-window the signal, if edge artifacts
        matter for your use case.

    Returns
    --------
    resampled_data : ndarray
        Resampled signal
    new_time_axis : ndarray
        New time axis for resampled signal, spaced at exactly 1/target_fs.
    """
    data = np.asarray(data, dtype=np.float64)

    if original_fs == target_fs:
        time_axis = np.arange(len(data)) / original_fs
        return data, time_axis

    original_duration = len(data) / original_fs

    if method == "scipy":
        # Calculate number of samples in resampled signal
        num_samples = int(round(len(data) * target_fs / original_fs))

        # Use scipy's resample function
        resampled_data = signal.resample(data, num_samples)
        # Samples must be spaced at exactly 1/target_fs, not
        # original_duration / (num_samples - 1) (what linspace(..., endpoint=True)
        # would give), otherwise the returned axis silently disagrees with
        # the sampling frequency the caller asked for.
        new_time_axis = np.arange(num_samples) / target_fs

    elif method == "interpolation":
        original_time = np.arange(len(data)) / original_fs
        # Compute the sample count directly instead of stepping an
        # np.arange by a float (1/target_fs), which is prone to
        # floating-point drift producing an off-by-one sample count.
        num_samples = int(round(original_duration * target_fs))
        new_time_axis = np.arange(num_samples) / target_fs

        # Interpolate
        f = interp1d(
            original_time,
            data,
            kind="linear",
            bounds_error=False,
            fill_value="extrapolate",
        )
        resampled_data = f(new_time_axis)

    else:
        raise ValueError("method must be 'scipy' or 'interpolation'")

    return resampled_data, new_time_axis


def segment_signal(
    data: ArrayLike,
    fs: float,
    segment_length: float,
    overlap: float = 0.0,
    window: str = "hann",
) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
    """
    Segment signal into overlapping windows.

    Parameters
    -----------
    data : array-like
        Input signal data
    fs : float
        Sampling frequency in Hz
    segment_length : float
        Length of each segment in seconds
    overlap : float
        Overlap between segments, in [0.0, 1.0)
    window : str
        Window function to apply ('hann', 'hamming', 'blackman', 'rect')

    Returns
    --------
    segments : ndarray
        2D array where each row is a segment
    segment_times : ndarray
        Start times for each segment
    """
    data = np.asarray(data, dtype=np.float64)

    if segment_length <= 0:
        raise ValueError("segment_length must be positive")
    if not (0.0 <= overlap < 1.0):
        raise ValueError("overlap must satisfy 0.0 <= overlap < 1.0")

    segment_samples = int(segment_length * fs)
    if segment_samples < 1:
        raise ValueError("segment_length is too short for the given sampling rate")
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
    if window == "hann":
        win = np.hanning(segment_samples)
    elif window == "hamming":
        win = np.hamming(segment_samples)
    elif window == "blackman":
        win = np.blackman(segment_samples)
    elif window == "rect":
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


def calculate_snr(
    signal_data: ArrayLike,
    noise_data: Optional[ArrayLike] = None,
    method: str = "power",
) -> float:
    """
    Calculate Signal-to-Noise Ratio (SNR).

    Parameters
    -----------
    signal_data : array-like
        Clean signal or signal+noise
    noise_data : array-like, optional
        Noise signal (if None, estimated from signal)
    method : str
        Method for SNR calculation ('power', 'amplitude'). Both give the
        same physical quantity (power ratio, in dB); 'amplitude' is provided
        for convenience when you only have RMS values on hand.

    Returns
    --------
    snr_db : float
        Signal-to-Noise Ratio in decibels
    """
    signal_arr: NDArray[np.float64] = np.asarray(signal_data, dtype=np.float64)

    noise_arr: NDArray[np.float64]
    if noise_data is None:
        # Estimate noise as high-frequency components
        noise_arr = signal_arr - signal.medfilt(signal_arr, kernel_size=5)
    else:
        noise_arr = np.asarray(noise_data, dtype=np.float64)

    if method == "power":
        signal_power = np.mean(signal_arr**2)
        noise_power = np.mean(noise_arr**2)

        if noise_power == 0:
            return float("inf")

        snr_linear = signal_power / noise_power
        snr_db = 10 * np.log10(snr_linear)

    elif method == "amplitude":
        signal_rms = np.sqrt(np.mean(signal_arr**2))
        noise_rms = np.sqrt(np.mean(noise_arr**2))

        if noise_rms == 0:
            return float("inf")

        snr_linear = signal_rms / noise_rms
        # Power is proportional to amplitude squared, so an amplitude
        # (RMS) ratio needs 20*log10, not 10*log10, to express the same
        # dB quantity as the power method above. Using 10*log10 here
        # (the previous behavior) understated every amplitude-mode SNR
        # by exactly a factor of 2 in dB.
        snr_db = 20 * np.log10(snr_linear)

    else:
        raise ValueError("method must be 'power' or 'amplitude'")

    return float(snr_db)


def detect_peaks(
    data: ArrayLike,
    fs: float,
    min_distance: Optional[float] = None,
    height: Optional[float] = None,
    prominence: Optional[float] = None,
) -> Tuple[NDArray[np.intp], Dict[str, Any]]:
    """
    Detect peaks in biomedical signals (e.g., R-peaks in ECG).

    Parameters
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

    Returns
    --------
    peak_indices : ndarray
        Indices of detected peaks
    peak_properties : dict
        Properties of detected peaks
    """
    data = np.asarray(data, dtype=np.float64)

    # Convert min_distance from seconds to samples
    if min_distance is not None:
        min_distance_samples = int(min_distance * fs)
    else:
        min_distance_samples = None

    # Find peaks
    peak_indices, properties = find_peaks(
        data, distance=min_distance_samples, height=height, prominence=prominence
    )

    # Convert peak times to seconds
    peak_times = peak_indices / fs
    properties["peak_times"] = peak_times

    return peak_indices, properties


def remove_outliers(
    data: ArrayLike,
    method: str = "iqr",
    threshold: float = 1.5,
) -> Tuple[NDArray[np.float64], NDArray[np.bool_]]:
    """
    Remove outliers from signal data.

    Parameters
    -----------
    data : array-like
        Input signal data
    method : str
        Method for outlier detection ('iqr', 'zscore', 'mad')
    threshold : float
        Threshold for outlier detection

    Returns
    --------
    cleaned_data : ndarray
        Data with outliers removed/replaced
    outlier_mask : ndarray
        Boolean mask indicating outlier locations
    """
    data = np.asarray(data, dtype=np.float64)
    outlier_mask = np.zeros(len(data), dtype=bool)

    if method == "iqr":
        Q1 = np.percentile(data, 25)
        Q3 = np.percentile(data, 75)
        IQR = Q3 - Q1
        lower_bound = Q1 - threshold * IQR
        upper_bound = Q3 + threshold * IQR
        outlier_mask = (data < lower_bound) | (data > upper_bound)

    elif method == "zscore":
        z_scores = np.abs(zscore(data))
        outlier_mask = z_scores > threshold

    elif method == "mad":
        median_val = np.median(data)
        mad = np.median(np.abs(data - median_val))
        if mad == 0:
            warnings.warn("MAD is zero, no samples flagged as outliers")
            mad = np.inf
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
        else:
            warnings.warn(
                "Fewer than 2 non-outlier samples remain; outliers were "
                "left unmodified because there is nothing to interpolate from."
            )

    return cleaned_data, outlier_mask


def calculate_signal_quality(
    data: ArrayLike,
    fs: float,
    window_length: float = 5.0,
) -> Dict[str, Optional[float]]:
    """
    Assess signal quality using multiple metrics.

    Parameters
    -----------
    data : array-like
        Input signal data
    fs : float
        Sampling frequency in Hz
    window_length : float
        Length of analysis window in seconds

    Returns
    --------
    quality_metrics : dict
        Dictionary containing quality assessment metrics
    """
    data = np.asarray(data, dtype=np.float64)
    window_samples = int(window_length * fs)

    # Initialize metrics
    quality_metrics: Dict[str, Optional[float]] = {}

    # 1. Signal-to-Noise Ratio
    try:
        quality_metrics["snr_db"] = calculate_snr(data)
    except Exception:
        quality_metrics["snr_db"] = None

    # 2. Baseline stability (standard deviation of low-frequency component)
    try:
        kernel_size = min(window_samples, len(data) // 10)
        if kernel_size % 2 == 0:
            kernel_size += 1  # medfilt requires an odd kernel size
        kernel_size = max(kernel_size, 1)
        baseline = signal.medfilt(data, kernel_size=kernel_size)
        quality_metrics["baseline_stability"] = float(np.std(baseline))
    except Exception:
        quality_metrics["baseline_stability"] = None

    # 3. Artifact percentage
    try:
        _, artifact_mask = remove_outliers(data, method="iqr", threshold=2.0)
        quality_metrics["artifact_percentage"] = float(np.mean(artifact_mask) * 100)
    except Exception:
        quality_metrics["artifact_percentage"] = None

    # 4. Frequency domain metrics
    try:
        nperseg = min(window_samples, len(data) // 4)
        freqs, psd = signal.welch(data, fs, nperseg=nperseg)
        if np.sum(psd) > 0:
            quality_metrics["peak_frequency"] = float(freqs[np.argmax(psd)])
            quality_metrics["spectral_centroid"] = float(
                np.sum(freqs * psd) / np.sum(psd)
            )
        else:
            quality_metrics["peak_frequency"] = None
            quality_metrics["spectral_centroid"] = None
    except Exception:
        quality_metrics["peak_frequency"] = None
        quality_metrics["spectral_centroid"] = None

    # 5. Statistical metrics
    data_std = float(np.std(data))
    quality_metrics["mean"] = float(np.mean(data))
    quality_metrics["std"] = data_std
    if data_std > 0:
        quality_metrics["skewness"] = float(
            np.mean(((data - np.mean(data)) / data_std) ** 3)
        )
        quality_metrics["kurtosis"] = float(
            np.mean(((data - np.mean(data)) / data_std) ** 4) - 3
        )
    else:
        quality_metrics["skewness"] = None
        quality_metrics["kurtosis"] = None

    return quality_metrics


def apply_window(data: ArrayLike, window_type: str = "hann") -> NDArray[np.float64]:
    """
    Apply window function to signal data.

    Parameters
    -----------
    data : array-like
        Input signal data
    window_type : str
        Type of window ('hann', 'hamming', 'blackman', 'tukey', 'rect')

    Returns
    --------
    windowed_data : ndarray
        Signal with window function applied
    """
    data = np.asarray(data, dtype=np.float64)

    if window_type == "hann":
        window = np.hanning(len(data))
    elif window_type == "hamming":
        window = np.hamming(len(data))
    elif window_type == "blackman":
        window = np.blackman(len(data))
    elif window_type == "tukey":
        window = signal.windows.tukey(len(data), alpha=0.5)
    elif window_type == "rect":
        window = np.ones(len(data))
    else:
        raise ValueError(
            "window_type must be 'hann', 'hamming', 'blackman', 'tukey', or 'rect'"
        )

    windowed_data: NDArray[np.float64] = data * window

    return windowed_data
