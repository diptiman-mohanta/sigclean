"""Tests for sigclean.hrv."""

import numpy as np
import pytest

from sigclean.hrv import (
    hrv_metrics,
    mean_heart_rate,
    pnn50,
    rmssd,
    rr_intervals,
    sdnn,
)


def test_rr_intervals_from_regular_peaks():
    fs = 1000
    peak_indices = np.array([0, 1000, 2000, 3000])  # 1 Hz -> 1000 ms RR intervals
    rr_ms = rr_intervals(peak_indices, fs)
    np.testing.assert_allclose(rr_ms, [1000.0, 1000.0, 1000.0])


def test_rr_intervals_filters_implausible_values():
    fs = 1000
    # 100 ms RR (600 bpm, implausible) and 3000 ms RR (20 bpm, implausible)
    # mixed in with a normal 800 ms RR interval.
    peak_indices = np.array([0, 100, 900, 3900])
    rr_ms = rr_intervals(peak_indices, fs)
    np.testing.assert_allclose(rr_ms, [800.0])


def test_rr_intervals_too_few_peaks_returns_empty():
    assert len(rr_intervals(np.array([0]), fs=1000)) == 0
    assert len(rr_intervals(np.array([]), fs=1000)) == 0


@pytest.mark.parametrize("metric_fn", [sdnn, rmssd, pnn50])
def test_metrics_are_nan_for_insufficient_data(metric_fn):
    assert np.isnan(metric_fn(np.array([])))
    assert np.isnan(metric_fn(np.array([800.0])))


def test_mean_heart_rate_matches_known_interval():
    rr_ms = np.full(10, 1000.0)  # 1000 ms RR -> 60 bpm
    assert mean_heart_rate(rr_ms) == pytest.approx(60.0)


def test_sdnn_zero_for_perfectly_regular_rr():
    rr_ms = np.full(10, 800.0)
    assert sdnn(rr_ms) == pytest.approx(0.0, abs=1e-9)


def test_rmssd_zero_for_perfectly_regular_rr():
    rr_ms = np.full(10, 800.0)
    assert rmssd(rr_ms) == pytest.approx(0.0, abs=1e-9)


def test_hrv_metrics_recovers_known_heart_rate(synthetic_ecg_known_hr, fs):
    signal, true_hr_bpm = synthetic_ecg_known_hr
    metrics = hrv_metrics(signal, fs, min_distance=0.3, height=0.1)
    assert metrics["mean_hr_bpm"] == pytest.approx(true_hr_bpm, abs=1.0)
    assert metrics["num_peaks"] > 0
    assert metrics["sdnn_ms"] >= 0


def test_hrv_metrics_handles_flat_signal(fs):
    # No peaks at all should degrade gracefully, not raise.
    flat_signal = np.zeros(fs * 5)
    metrics = hrv_metrics(flat_signal, fs)
    assert metrics["num_peaks"] == 0
    assert np.isnan(metrics["sdnn_ms"])
