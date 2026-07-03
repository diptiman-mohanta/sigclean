"""
Shared pytest fixtures for synthetic biomedical-like test signals.

`tests/test_filters.py` and `tests/test_utils.py` predate this file and use
unittest-style `setUp` methods instead (pytest runs both styles fine); newer
tests for `pipeline`, `hrv`, `io`, and `cli` use these fixtures to avoid
re-deriving the same synthetic signals in every test module.
"""

import numpy as np
import pytest
from scipy.ndimage import gaussian_filter1d

FS = 1000


@pytest.fixture
def fs():
    return FS


@pytest.fixture
def clean_sine_signal(fs):
    """A clean 10 Hz sine wave, 2 seconds long."""
    duration = 2
    t = np.arange(duration * fs) / fs
    return np.sin(2 * np.pi * 10 * t)


@pytest.fixture
def noisy_signal(fs, clean_sine_signal):
    """`clean_sine_signal` plus Gaussian noise and 50 Hz powerline interference."""
    rng = np.random.RandomState(42)
    noise = 0.1 * rng.randn(len(clean_sine_signal))
    t = np.arange(len(clean_sine_signal)) / fs
    powerline = 0.2 * np.sin(2 * np.pi * 50 * t)
    return clean_sine_signal + noise + powerline


@pytest.fixture
def synthetic_ecg_known_hr(fs):
    """A synthetic ECG-like signal with sharp, evenly-spaced peaks at a known
    heart rate, for testing R-peak detection and HRV metrics against ground
    truth. Returns (signal, heart_rate_bpm)."""
    duration = 30
    hr_bpm = 72.0
    beat_interval = 60.0 / hr_bpm

    t = np.arange(duration * fs) / fs
    signal = np.zeros(len(t))
    peak_times = np.arange(0, duration, beat_interval)
    for peak_time in peak_times:
        idx = int(peak_time * fs)
        if idx < len(signal):
            signal[idx] = 1.0
    signal = gaussian_filter1d(signal, sigma=2)

    return signal, hr_bpm


@pytest.fixture
def multichannel_signal(noisy_signal):
    """A 3-channel version of `noisy_signal`, shape (3, n_samples)."""
    return np.stack([noisy_signal, noisy_signal * 0.5 + 0.1, -noisy_signal], axis=0)
