"""
Heart-rate-variability (HRV) metrics for ECG-like signals.

Built on top of `sigclean.utils.detect_peaks`: detect R-peaks, convert
consecutive peak-to-peak distances into RR intervals (in milliseconds), and
compute the standard time-domain HRV metrics from them.
"""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .utils import detect_peaks

# Physiologically plausible RR interval range, in milliseconds
# (30-200 bpm). RR intervals outside this range are almost always peak
# detection errors (missed or extra beats) rather than real heartbeats, and
# left in, a single missed beat can double an RR interval and dominate SDNN.
DEFAULT_MIN_RR_MS = 300.0
DEFAULT_MAX_RR_MS = 2000.0


def rr_intervals(
    peak_indices: ArrayLike,
    fs: float,
    min_rr_ms: Optional[float] = DEFAULT_MIN_RR_MS,
    max_rr_ms: Optional[float] = DEFAULT_MAX_RR_MS,
) -> NDArray[np.float64]:
    """
    Convert R-peak sample indices into RR (beat-to-beat) intervals.

    Parameters
    -----------
    peak_indices : array-like
        Sample indices of detected R-peaks, as returned by `detect_peaks`.
    fs : float
        Sampling frequency in Hz.
    min_rr_ms, max_rr_ms : float, optional
        RR intervals outside [min_rr_ms, max_rr_ms] are dropped as likely
        peak-detection errors rather than genuine beats. Pass None for
        either bound to disable that side of the filter.

    Returns
    --------
    rr_ms : ndarray
        RR intervals in milliseconds.
    """
    peak_indices = np.asarray(peak_indices, dtype=np.float64)
    if len(peak_indices) < 2:
        return np.array([], dtype=np.float64)

    rr_ms = np.diff(peak_indices) / fs * 1000.0

    if min_rr_ms is not None:
        rr_ms = rr_ms[rr_ms >= min_rr_ms]
    if max_rr_ms is not None:
        rr_ms = rr_ms[rr_ms <= max_rr_ms]

    return rr_ms


def sdnn(rr_ms: ArrayLike) -> float:
    """Standard deviation of RR intervals (ms) -- overall HRV magnitude."""
    rr_ms = np.asarray(rr_ms, dtype=np.float64)
    if len(rr_ms) < 2:
        return float("nan")
    return float(np.std(rr_ms, ddof=1))


def rmssd(rr_ms: ArrayLike) -> float:
    """Root mean square of successive RR interval differences (ms) --
    short-term, parasympathetically-mediated HRV."""
    rr_ms = np.asarray(rr_ms, dtype=np.float64)
    if len(rr_ms) < 2:
        return float("nan")
    successive_diffs = np.diff(rr_ms)
    return float(np.sqrt(np.mean(successive_diffs**2)))


def pnn50(rr_ms: ArrayLike) -> float:
    """Percentage of successive RR interval pairs differing by > 50 ms."""
    rr_ms = np.asarray(rr_ms, dtype=np.float64)
    if len(rr_ms) < 2:
        return float("nan")
    successive_diffs = np.abs(np.diff(rr_ms))
    return float(np.mean(successive_diffs > 50.0) * 100.0)


def mean_heart_rate(rr_ms: ArrayLike) -> float:
    """Mean heart rate in beats per minute, from RR intervals (ms)."""
    rr_ms = np.asarray(rr_ms, dtype=np.float64)
    if len(rr_ms) == 0:
        return float("nan")
    return float(60000.0 / np.mean(rr_ms))


def hrv_metrics(
    data: ArrayLike,
    fs: float,
    min_distance: float = 0.3,
    height: Optional[float] = None,
    prominence: Optional[float] = None,
    min_rr_ms: Optional[float] = DEFAULT_MIN_RR_MS,
    max_rr_ms: Optional[float] = DEFAULT_MAX_RR_MS,
) -> Dict[str, float]:
    """
    Detect R-peaks in an ECG-like signal and compute standard time-domain
    HRV metrics in one call.

    Parameters
    -----------
    data : array-like
        ECG-like signal, ideally already bandpass-filtered/cleaned.
    fs : float
        Sampling frequency in Hz.
    min_distance : float
        Minimum distance between peaks in seconds, passed to `detect_peaks`
        (default 0.3s, i.e. caps heart rate at 200 bpm).
    height, prominence : float, optional
        Passed through to `detect_peaks` to help discriminate R-peaks from
        noise; tune per signal amplitude if peak detection misses/over-detects.
    min_rr_ms, max_rr_ms : float, optional
        Passed through to `rr_intervals` for outlier rejection.

    Returns
    --------
    metrics : dict
        'num_peaks', 'num_rr_intervals', 'mean_hr_bpm', 'sdnn_ms',
        'rmssd_ms', 'pnn50_percent'. Metrics that need at least 2 valid RR
        intervals are `nan` if too few peaks were detected.
    """
    data = np.asarray(data, dtype=np.float64)
    peak_indices, _ = detect_peaks(
        data,
        fs,
        min_distance=min_distance,
        height=height,
        prominence=prominence,
    )
    rr_ms = rr_intervals(peak_indices, fs, min_rr_ms=min_rr_ms, max_rr_ms=max_rr_ms)

    return {
        "num_peaks": float(len(peak_indices)),
        "num_rr_intervals": float(len(rr_ms)),
        "mean_hr_bpm": mean_heart_rate(rr_ms),
        "sdnn_ms": sdnn(rr_ms),
        "rmssd_ms": rmssd(rr_ms),
        "pnn50_percent": pnn50(rr_ms),
    }
