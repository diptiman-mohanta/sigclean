"""
SigClean - A library for cleaning biomedical signals

A comprehensive Python library for preprocessing and cleaning biomedical signals
including ECG, EMG, EEG, and other physiological signals.
"""

__version__ = "1.1.0"
__author__ = "Diptiman Mohanta"
__email__ = "diptimanmohanta7@gmail.com"

# Import main modules
from . import filters, hrv, io, pipeline, plot, utils

# Import commonly used functions for convenience
from .filters import (
    adaptive_filter,
    artifact_removal,
    baseline_removal,
    butterworth_filter,
    notch_filter,
)
from .hrv import hrv_metrics, rr_intervals
from .io import read_csv_column
from .pipeline import (
    ArtifactRemoval,
    BandpassFilter,
    BandstopFilter,
    BaselineRemoval,
    HighpassFilter,
    LowpassFilter,
    MovingAverage,
    Normalize,
    NotchFilter,
    Pipeline,
    SavitzkyGolay,
)
from .plot import (
    plot_before_after,
    plot_filter_response,
    plot_frequency_spectrum,
    plot_multiple_signals,
    plot_signal,
    plot_signal_quality_assessment,
    plot_spectrogram,
)
from .utils import (
    calculate_snr,
    detect_peaks,
    normalize_signal,
    remove_outliers,
    resample_signal,
    segment_signal,
)

__all__ = [
    "filters",
    "utils",
    "plot",
    "pipeline",
    "hrv",
    "io",
    "butterworth_filter",
    "notch_filter",
    "baseline_removal",
    "artifact_removal",
    "adaptive_filter",
    "normalize_signal",
    "resample_signal",
    "segment_signal",
    "calculate_snr",
    "detect_peaks",
    "remove_outliers",
    "plot_signal",
    "plot_frequency_spectrum",
    "plot_before_after",
    "plot_multiple_signals",
    "plot_spectrogram",
    "plot_signal_quality_assessment",
    "plot_filter_response",
    "Pipeline",
    "NotchFilter",
    "BandpassFilter",
    "LowpassFilter",
    "HighpassFilter",
    "BandstopFilter",
    "BaselineRemoval",
    "ArtifactRemoval",
    "MovingAverage",
    "SavitzkyGolay",
    "Normalize",
    "hrv_metrics",
    "rr_intervals",
    "read_csv_column",
]
