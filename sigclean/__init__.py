"""
SigClean - A library for cleaning biomedical signals

A comprehensive Python library for preprocessing and cleaning biomedical signals
including ECG, EMG, EEG, and other physiological signals.
"""

__version__ = "1.0.0"
__author__ = "Diptiman Mohanta"
__email__ = "diptimanmohanta7@gmail.com"

# Import main modules
from . import filters
from . import utils
from . import plot

# Import commonly used functions for convenience
from .filters import (
    butterworth_filter,
    notch_filter,
    baseline_removal,
    artifact_removal,
    adaptive_filter
)

from .utils import (
    normalize_signal,
    resample_signal,
    segment_signal,
    calculate_snr,
    detect_peaks,
    remove_outliers
)

from .plot import (
    plot_signal,
    plot_frequency_spectrum,
    plot_before_after,
    plot_multiple_signals
)

__all__ = [
    'filters',
    'utils', 
    'plot',
    'butterworth_filter',
    'notch_filter',
    'baseline_removal',
    'artifact_removal',
    'adaptive_filter',
    'normalize_signal',
    'resample_signal',
    'segment_signal',
    'calculate_snr',
    'detect_peaks',
    'remove_outliers',
    'plot_signal',
    'plot_frequency_spectrum',
    'plot_before_after',
    'plot_multiple_signals'
]