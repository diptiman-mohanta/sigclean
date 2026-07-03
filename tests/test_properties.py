"""
Property-based tests (hypothesis) for shape/finiteness invariants that
should hold across the *entire* input space, not just the handful of
synthetic signals the example-based tests happen to construct.

These target exactly the kind of numerical edge case that caused real bugs
in this codebase before (e.g. filters silently returning NaN under some
parameter combinations): rather than enumerate cases by hand, let hypothesis
search for a counterexample.
"""

import numpy as np
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays

from sigclean.filters import (
    baseline_removal,
    butterworth_filter,
    moving_average_filter,
    notch_filter,
)
from sigclean.utils import normalize_signal, remove_outliers

# Signals bounded away from +/-inf/NaN and from absurd magnitudes, long
# enough for filtfilt's default padding requirements.
finite_signals = arrays(
    dtype=np.float64,
    shape=st.integers(min_value=200, max_value=500),
    elements=st.floats(
        min_value=-1e3, max_value=1e3, allow_nan=False, allow_infinity=False
    ),
)

short_finite_signals = arrays(
    dtype=np.float64,
    shape=st.integers(min_value=10, max_value=200),
    elements=st.floats(
        min_value=-1e3, max_value=1e3, allow_nan=False, allow_infinity=False
    ),
)

settings.register_profile(
    "sigclean", max_examples=30, suppress_health_check=[HealthCheck.too_slow]
)
settings.load_profile("sigclean")


@given(data=finite_signals, order=st.integers(min_value=1, max_value=4))
def test_butterworth_bandpass_preserves_length_and_finiteness(data, order):
    fs = 1000.0
    filtered = butterworth_filter(
        data, fs, filter_type="bandpass", low_freq=1.0, high_freq=40.0, order=order
    )
    assert len(filtered) == len(data)
    assert np.all(np.isfinite(filtered))


@given(data=finite_signals)
def test_notch_filter_preserves_length_and_finiteness(data):
    fs = 1000.0
    filtered = notch_filter(data, fs, freq=50.0)
    assert len(filtered) == len(data)
    assert np.all(np.isfinite(filtered))


@given(data=short_finite_signals, window_size=st.integers(min_value=1, max_value=50))
def test_moving_average_preserves_length_and_finiteness(data, window_size):
    smoothed = moving_average_filter(data, window_size)
    assert len(smoothed) == len(data)
    assert np.all(np.isfinite(smoothed))


@given(data=short_finite_signals)
def test_moving_average_of_constant_signal_is_identity(data):
    # Regardless of window size or signal length, averaging a constant
    # signal must return that same constant everywhere (the bug this
    # regression test targets: edge samples used to be biased toward zero).
    constant_value = float(data[0])
    constant_signal = np.full(len(data), constant_value)
    window_size = max(1, len(data) // 3)
    smoothed = moving_average_filter(constant_signal, window_size)
    np.testing.assert_allclose(smoothed, constant_value)


@given(data=short_finite_signals)
def test_baseline_removal_detrend_preserves_length_and_finiteness(data):
    corrected = baseline_removal(data, method="detrend")
    assert len(corrected) == len(data)
    assert np.all(np.isfinite(corrected))


@given(data=short_finite_signals)
def test_normalize_zscore_preserves_length(data):
    normalized, params = normalize_signal(data, method="zscore")
    assert len(normalized) == len(data)
    assert np.all(np.isfinite(normalized))


@given(data=short_finite_signals, threshold=st.floats(min_value=0.5, max_value=5.0))
def test_remove_outliers_mask_matches_length(data, threshold):
    cleaned, mask = remove_outliers(data, method="iqr", threshold=threshold)
    assert len(cleaned) == len(data)
    assert len(mask) == len(data)
    assert mask.dtype == bool
    assert np.all(np.isfinite(cleaned))
