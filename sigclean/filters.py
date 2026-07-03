"""
Signal filtering functions for biomedical signal processing.

This module contains various filtering techniques commonly used in
biomedical signal processing including bandpass filters, notch filters,
baseline removal, and artifact removal.
"""

from __future__ import annotations

import warnings
from typing import Optional, Tuple

import numpy as np
import scipy.signal as signal
from numpy.typing import ArrayLike, NDArray
from scipy.ndimage import median_filter, uniform_filter1d
from scipy.signal import butter, filtfilt, iirnotch, sosfiltfilt


def _validate_cutoff(name: str, freq: float, nyquist: float) -> None:
    if freq <= 0:
        raise ValueError(f"{name} must be positive, got {freq} Hz")
    if freq >= nyquist:
        raise ValueError(
            f"{name} ({freq} Hz) must be below the Nyquist frequency "
            f"({nyquist} Hz, i.e. fs/2). Increase fs or lower {name}."
        )


def butterworth_filter(
    data: ArrayLike,
    fs: float,
    filter_type: str = "bandpass",
    low_freq: Optional[float] = None,
    high_freq: Optional[float] = None,
    order: int = 4,
) -> NDArray[np.float64]:
    """
    Apply Butterworth filter to biomedical signal.

    Parameters
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

    Returns
    --------
    filtered_data : ndarray
        Filtered signal

    Raises
    -------
    ValueError
        If cutoff frequencies are missing, out of order, or not strictly
        below the Nyquist frequency (fs / 2). Frequencies are never silently
        clamped, since doing so changes the filter the caller asked for
        without their knowledge.
    """
    nyquist = fs / 2

    if filter_type == "lowpass":
        if high_freq is None:
            raise ValueError("high_freq must be specified for lowpass filter")
        _validate_cutoff("high_freq", high_freq, nyquist)

        sos = butter(order, high_freq / nyquist, btype="low", output="sos")

    elif filter_type == "highpass":
        if low_freq is None:
            raise ValueError("low_freq must be specified for highpass filter")
        _validate_cutoff("low_freq", low_freq, nyquist)

        sos = butter(order, low_freq / nyquist, btype="high", output="sos")

    elif filter_type == "bandpass":
        if low_freq is None or high_freq is None:
            raise ValueError(
                "Both low_freq and high_freq must be specified for bandpass filter"
            )
        if low_freq >= high_freq:
            raise ValueError("low_freq must be less than high_freq")
        _validate_cutoff("low_freq", low_freq, nyquist)
        _validate_cutoff("high_freq", high_freq, nyquist)

        sos = butter(
            order, [low_freq / nyquist, high_freq / nyquist], btype="band", output="sos"
        )

    elif filter_type == "bandstop":
        if low_freq is None or high_freq is None:
            raise ValueError(
                "Both low_freq and high_freq must be specified for bandstop filter"
            )
        if low_freq >= high_freq:
            raise ValueError("low_freq must be less than high_freq")
        _validate_cutoff("low_freq", low_freq, nyquist)
        _validate_cutoff("high_freq", high_freq, nyquist)

        sos = butter(
            order,
            [low_freq / nyquist, high_freq / nyquist],
            btype="bandstop",
            output="sos",
        )

    else:
        raise ValueError(
            "filter_type must be 'lowpass', 'highpass', 'bandpass', or 'bandstop'"
        )

    # Second-order-sections + zero-phase filtering: numerically stable even
    # for high order / narrow-band designs, unlike the (b, a) transfer-function
    # form which can lose precision catastrophically (producing NaN output)
    # under the exact same conditions.
    filtered_data: NDArray[np.float64] = sosfiltfilt(sos, data)

    return filtered_data


def notch_filter(
    data: ArrayLike,
    fs: float,
    freq: float = 50.0,
    quality_factor: float = 30,
) -> NDArray[np.float64]:
    """
    Apply notch filter to remove power line interference.

    Parameters
    -----------
    data : array-like
        Input signal data
    fs : float
        Sampling frequency in Hz
    freq : float
        Frequency to remove (default: 50 Hz for EU, use 60 Hz for US)
    quality_factor : float
        Quality factor (higher values = narrower notch)

    Returns
    --------
    filtered_data : ndarray
        Filtered signal with power line interference removed

    Raises
    -------
    ValueError
        If the notch frequency is not strictly below the Nyquist frequency.
    """
    nyquist = fs / 2
    _validate_cutoff("freq", freq, nyquist)

    # Design notch filter
    b, a = iirnotch(freq, quality_factor, fs)

    # Apply filter
    filtered_data: NDArray[np.float64] = filtfilt(b, a, data)

    return filtered_data


def baseline_removal(
    data: ArrayLike,
    method: str = "detrend",
    window_size: Optional[int] = None,
    polynomial_order: int = 3,
) -> NDArray[np.float64]:
    """
    Remove baseline drift from biomedical signals.

    Parameters
    -----------
    data : array-like
        Input signal data
    method : str
        Method for baseline removal ('detrend', 'median', 'polynomial')
    window_size : int
        Window size for median filtering (only for 'median' method)
    polynomial_order : int
        Order of polynomial for detrending (only for 'polynomial' method)

    Returns
    --------
    corrected_data : ndarray
        Signal with baseline removed
    """
    data = np.asarray(data, dtype=np.float64)

    if method == "detrend":
        # Simple linear detrending
        corrected_data = signal.detrend(data, type="linear")

    elif method == "median":
        if window_size is None:
            window_size = len(data) // 10  # Default to 10% of signal length
        if window_size % 2 == 0:
            window_size += 1  # Odd size keeps the filter symmetric about each sample

        # Median filtering for baseline estimation
        baseline = median_filter(data, size=window_size, mode="reflect")
        corrected_data = data - baseline

    elif method == "polynomial":
        # Polynomial detrending. Fitting raw sample indices (0..N) directly
        # produces a severely ill-conditioned Vandermonde matrix for long
        # recordings, so center and scale x before fitting (numpy's own
        # documentation recommends this for numerical stability) and undo
        # the transform when evaluating the baseline.
        n = len(data)
        x = np.arange(n, dtype=np.float64)
        x_mean = x.mean()
        x_scale = max(x.std(), 1.0)
        x_norm = (x - x_mean) / x_scale
        coeffs = np.polyfit(x_norm, data, polynomial_order)
        baseline = np.polyval(coeffs, x_norm)
        corrected_data = data - baseline

    else:
        raise ValueError("method must be 'detrend', 'median', or 'polynomial'")

    return np.asarray(corrected_data, dtype=np.float64)


def artifact_removal(
    data: ArrayLike,
    fs: float,
    threshold_std: float = 3.0,
    window_size: Optional[int] = None,
) -> Tuple[NDArray[np.float64], NDArray[np.bool_]]:
    """
    Remove artifacts from biomedical signals using robust local thresholding.

    Parameters
    -----------
    data : array-like
        Input signal data
    fs : float
        Sampling frequency in Hz
    threshold_std : float
        Detection threshold, in robust (MAD-scaled) standard-deviation-
        equivalent units. Named threshold_std for backwards compatibility.
    window_size : int
        Window size for local artifact detection (default: 1 second)

    Returns
    --------
    cleaned_data : ndarray
        Signal with artifacts removed/interpolated
    artifact_mask : ndarray
        Boolean mask indicating artifact locations

    Notes
    ------
    Local statistics are computed with a rolling median and median absolute
    deviation (MAD) rather than a rolling mean/std. A rolling mean/std is
    itself dragged toward the artifact by the very sample it is testing,
    which makes large spikes systematically under-detected; the median-based
    estimator is robust to that contamination. Both passes are vectorized
    (via scipy.ndimage.median_filter), which is orders of magnitude faster
    than a per-sample Python loop on real-length recordings.
    """
    data = np.asarray(data, dtype=np.float64)

    if window_size is None:
        window_size = int(fs)  # 1 second window
    if window_size % 2 == 0:
        window_size += 1  # Odd size keeps the filter symmetric about each sample
    window_size = max(window_size, 3)

    cleaned_data = data.copy()

    local_median = median_filter(data, size=window_size, mode="reflect")
    abs_deviation = np.abs(data - local_median)
    local_mad = median_filter(abs_deviation, size=window_size, mode="reflect")

    # 0.6745 is the standard consistency constant that makes MAD comparable
    # to a standard deviation for normally-distributed data.
    safe_mad = np.where(local_mad > 0, local_mad, np.finfo(np.float64).eps)
    modified_z = 0.6745 * (data - local_median) / safe_mad
    artifact_mask = np.abs(modified_z) > threshold_std

    # Interpolate artifacts
    if np.any(artifact_mask):
        artifact_indices = np.where(artifact_mask)[0]
        clean_indices = np.where(~artifact_mask)[0]

        if len(clean_indices) > 1:
            cleaned_data[artifact_indices] = np.interp(
                artifact_indices, clean_indices, data[clean_indices]
            )

    return cleaned_data, artifact_mask


def adaptive_filter(
    data: ArrayLike,
    reference_signal: Optional[ArrayLike] = None,
    filter_length: int = 32,
    mu: float = 0.01,
) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
    """
    Apply LMS adaptive filtering for noise reduction.

    Parameters
    -----------
    data : array-like
        Primary input signal (signal + noise)
    reference_signal : array-like
        Reference signal, correlated with the noise component of `data` but
        not with the signal of interest (e.g. a separately-measured noise
        source). If omitted, a 1-sample-delayed copy of `data` is used as a
        crude self-reference; this does not implement true adaptive noise
        cancellation and will also attenuate genuine signal content that is
        correlated sample-to-sample (as most biomedical signals are). Pass
        an actual reference signal whenever one is available.
    filter_length : int
        Length of adaptive filter
    mu : float
        Step size parameter (learning rate). Too large diverges the LMS
        update; too small converges slowly. Scale relative to the input
        signal power (normalized LMS) if unsure.

    Returns
    --------
    filtered_signal : ndarray
        Adaptively filtered signal
    error_signal : ndarray
        Error signal (filtered output)
    """
    data = np.asarray(data, dtype=np.float64)

    if reference_signal is None:
        warnings.warn(
            "No reference_signal provided: falling back to a delayed copy "
            "of the input as a self-reference. This is not true adaptive "
            "noise cancellation and may attenuate real signal content. "
            "Pass an independent reference signal for meaningful results."
        )
        # Create delayed version as reference for self-adaptive filtering
        reference_signal = np.roll(data, 1)
        reference_signal[0] = 0
    else:
        reference_signal = np.asarray(reference_signal, dtype=np.float64)
        if len(reference_signal) != len(data):
            raise ValueError("Reference signal must have same length as input data")

    # Initialize filter weights
    weights = np.zeros(filter_length)
    filtered_signal = np.zeros(len(data))
    error_signal = np.zeros(len(data))

    # LMS adaptive filtering
    for n in range(filter_length, len(data)):
        # Extract reference signal segment
        x = reference_signal[n - filter_length : n][::-1]  # Reverse for convolution

        # Filter output
        y = np.dot(weights, x)
        filtered_signal[n] = y

        # Error signal
        error = data[n] - y
        error_signal[n] = error

        # Update weights
        weights += mu * error * x

    return filtered_signal, error_signal


def moving_average_filter(data: ArrayLike, window_size: int) -> NDArray[np.float64]:
    """
    Apply moving average filter for smoothing.

    Parameters
    -----------
    data : array-like
        Input signal data
    window_size : int
        Size of moving average window

    Returns
    --------
    smoothed_data : ndarray
        Smoothed signal
    """
    data = np.asarray(data, dtype=np.float64)

    if window_size < 1:
        raise ValueError("Window size must be at least 1")

    if window_size >= len(data):
        return np.full_like(data, np.mean(data))

    # uniform_filter1d with mode='nearest' reflects the edge value outward
    # instead of implicitly zero-padding (as np.convolve(..., mode='same')
    # does), which otherwise biases every sample within window_size/2 of
    # either edge toward zero. It is also an O(n) running-sum implementation
    # rather than an O(n * window_size) direct convolution.
    smoothed_data: NDArray[np.float64] = uniform_filter1d(
        data, size=window_size, mode="nearest"
    )

    return smoothed_data


def savitzky_golay_filter(
    data: ArrayLike,
    window_length: int,
    polyorder: int = 3,
) -> NDArray[np.float64]:
    """
    Apply Savitzky-Golay filter for signal smoothing while preserving features.

    Parameters
    -----------
    data : array-like
        Input signal data
    window_length : int
        Length of filter window (must be odd)
    polyorder : int
        Order of polynomial used for fitting

    Returns
    --------
    filtered_data : ndarray
        Filtered signal
    """
    if window_length % 2 == 0:
        window_length += 1  # Ensure odd window length

    if window_length < polyorder + 1:
        raise ValueError("window_length must be greater than polyorder")

    filtered_data: NDArray[np.float64] = signal.savgol_filter(
        data, window_length, polyorder
    )

    return filtered_data
