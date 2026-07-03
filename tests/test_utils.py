"""
Unit tests for sigclean.utils module.

Unlike the shape/dtype-only assertions in test_filters.py, these tests check
actual numerical correctness against known ground truth, since that is where
several real bugs previously hid undetected.
"""

import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sigclean.utils import (
    apply_window,
    calculate_signal_quality,
    calculate_snr,
    detect_peaks,
    normalize_signal,
    remove_outliers,
    resample_signal,
    segment_signal,
)


class TestNormalizeSignal(unittest.TestCase):
    def setUp(self):
        np.random.seed(0)
        self.data = np.random.randn(100_000)

    def test_zscore_normalization(self):
        normed, params = normalize_signal(self.data, method="zscore")
        self.assertAlmostEqual(np.mean(normed), 0.0, places=6)
        self.assertAlmostEqual(np.std(normed), 1.0, places=6)

    def test_minmax_normalization_range(self):
        normed, params = normalize_signal(
            self.data, method="minmax", feature_range=(0, 1)
        )
        self.assertAlmostEqual(np.min(normed), 0.0, places=6)
        self.assertAlmostEqual(np.max(normed), 1.0, places=6)

    def test_robust_normalization_is_scaled_like_zscore(self):
        # For normally-distributed data, a correctly-scaled robust (MAD-based)
        # normalization should have unit standard deviation, just like zscore.
        # An unscaled MAD (dividing by raw median absolute deviation) instead
        # gives std ~1.4826, which was the bug.
        normed, params = normalize_signal(self.data, method="robust")
        self.assertAlmostEqual(np.std(normed), 1.0, delta=0.05)

    def test_robust_normalization_consistent_with_remove_outliers_mad(self):
        # normalize_signal's 'robust' method and remove_outliers's 'mad'
        # method both compute a modified z-score; they must use the same
        # scaling convention (the 0.6745 consistency constant).
        normed, params = normalize_signal(self.data, method="robust")
        median_val = np.median(self.data)
        mad = np.median(np.abs(self.data - median_val))
        expected = (self.data - median_val) / (mad / 0.6745)
        np.testing.assert_allclose(normed, expected)

    def test_unit_vector_normalization(self):
        normed, params = normalize_signal(self.data, method="unit_vector")
        self.assertAlmostEqual(np.linalg.norm(normed), 1.0, places=6)

    def test_invalid_method_raises(self):
        with self.assertRaises(ValueError):
            normalize_signal(self.data, method="invalid")


class TestResampleSignal(unittest.TestCase):
    def setUp(self):
        self.original_fs = 1000
        self.target_fs = 500
        t = np.arange(2000) / self.original_fs
        self.data = np.sin(2 * np.pi * 5 * t)

    def test_scipy_time_axis_matches_target_fs(self):
        resampled, t_new = resample_signal(
            self.data, self.original_fs, self.target_fs, method="scipy"
        )
        dt = t_new[1] - t_new[0]
        self.assertAlmostEqual(dt, 1 / self.target_fs, places=9)
        self.assertAlmostEqual(
            t_new[-1], (len(resampled) - 1) / self.target_fs, places=9
        )

    def test_interpolation_time_axis_matches_target_fs(self):
        resampled, t_new = resample_signal(
            self.data, self.original_fs, self.target_fs, method="interpolation"
        )
        dt = t_new[1] - t_new[0]
        self.assertAlmostEqual(dt, 1 / self.target_fs, places=9)
        self.assertEqual(len(resampled), len(t_new))

    def test_same_fs_is_noop(self):
        resampled, t_new = resample_signal(
            self.data, self.original_fs, self.original_fs
        )
        np.testing.assert_array_equal(resampled, self.data)

    def test_upsampling_increases_length(self):
        resampled, t_new = resample_signal(self.data, self.original_fs, 2000)
        self.assertGreater(len(resampled), len(self.data))

    def test_invalid_method_raises(self):
        with self.assertRaises(ValueError):
            resample_signal(self.data, self.original_fs, self.target_fs, method="bogus")


class TestSegmentSignal(unittest.TestCase):
    def setUp(self):
        self.fs = 1000
        self.data = np.random.randn(10_000)

    def test_no_overlap_segment_count(self):
        segments, times = segment_signal(
            self.data, self.fs, segment_length=1.0, overlap=0.0, window="rect"
        )
        self.assertEqual(segments.shape[1], self.fs)
        self.assertEqual(segments.shape[0], len(self.data) // self.fs)

    def test_overlap_of_one_raises(self):
        with self.assertRaises(ValueError):
            segment_signal(self.data, self.fs, segment_length=1.0, overlap=1.0)

    def test_negative_overlap_raises(self):
        with self.assertRaises(ValueError):
            segment_signal(self.data, self.fs, segment_length=1.0, overlap=-0.1)

    def test_nonpositive_segment_length_raises(self):
        with self.assertRaises(ValueError):
            segment_signal(self.data, self.fs, segment_length=0.0)

    def test_segment_longer_than_signal_raises(self):
        with self.assertRaises(ValueError):
            segment_signal(self.data, self.fs, segment_length=1000.0)

    def test_invalid_window_raises(self):
        with self.assertRaises(ValueError):
            segment_signal(self.data, self.fs, segment_length=1.0, window="bogus")


class TestCalculateSnr(unittest.TestCase):
    def setUp(self):
        np.random.seed(1)
        t = np.linspace(0, 1, 1000)
        self.clean = np.sin(2 * np.pi * 5 * t)
        self.noise = 0.1 * np.random.randn(len(t))
        self.noisy = self.clean + self.noise

    def test_power_and_amplitude_methods_agree(self):
        # Both methods describe the same physical power ratio in dB, so
        # they must produce the same result given the same signal/noise.
        snr_power = calculate_snr(self.noisy, self.noise, method="power")
        snr_amp = calculate_snr(self.noisy, self.noise, method="amplitude")
        self.assertAlmostEqual(snr_power, snr_amp, places=9)

    def test_zero_noise_is_infinite(self):
        snr = calculate_snr(self.clean, np.zeros_like(self.clean), method="power")
        self.assertEqual(snr, float("inf"))

    def test_invalid_method_raises(self):
        with self.assertRaises(ValueError):
            calculate_snr(self.noisy, self.noise, method="bogus")


class TestDetectPeaks(unittest.TestCase):
    def test_detects_known_peaks(self):
        fs = 1000
        t = np.arange(3 * fs) / fs
        data = np.sin(2 * np.pi * 1 * t)  # 1 Hz -> 3 peaks in 3 seconds
        peak_indices, props = detect_peaks(data, fs, min_distance=0.5, height=0.9)
        self.assertEqual(len(peak_indices), 3)
        np.testing.assert_allclose(props["peak_times"], peak_indices / fs)


class TestRemoveOutliers(unittest.TestCase):
    def setUp(self):
        np.random.seed(2)
        self.data = np.random.randn(1000)
        self.data[500] = 100.0  # obvious outlier

    def test_iqr_detects_outlier(self):
        cleaned, mask = remove_outliers(self.data, method="iqr")
        self.assertTrue(mask[500])
        self.assertLess(abs(cleaned[500]), 10)

    def test_zscore_detects_outlier(self):
        cleaned, mask = remove_outliers(self.data, method="zscore", threshold=3.0)
        self.assertTrue(mask[500])

    def test_mad_detects_outlier(self):
        cleaned, mask = remove_outliers(self.data, method="mad", threshold=3.5)
        self.assertTrue(mask[500])

    def test_invalid_method_raises(self):
        with self.assertRaises(ValueError):
            remove_outliers(self.data, method="bogus")


class TestCalculateSignalQuality(unittest.TestCase):
    def test_returns_finite_metrics_for_clean_signal(self):
        fs = 1000
        t = np.arange(5 * fs) / fs
        data = np.sin(2 * np.pi * 5 * t) + 0.05 * np.random.randn(len(t))
        metrics = calculate_signal_quality(data, fs, window_length=1.0)
        for key in (
            "snr_db",
            "baseline_stability",
            "artifact_percentage",
            "peak_frequency",
            "spectral_centroid",
            "mean",
            "std",
            "skewness",
            "kurtosis",
        ):
            self.assertIn(key, metrics)
            self.assertIsNotNone(metrics[key])
            self.assertTrue(np.isfinite(metrics[key]))

    def test_constant_signal_does_not_crash(self):
        # std == 0 previously caused a silent NaN in skewness/kurtosis via
        # division by zero; it should now degrade gracefully to None.
        data = np.ones(2000) * 3.0
        metrics = calculate_signal_quality(data, fs=1000, window_length=1.0)
        self.assertIsNone(metrics["skewness"])
        self.assertIsNone(metrics["kurtosis"])


class TestApplyWindow(unittest.TestCase):
    def test_rect_window_is_identity(self):
        data = np.random.randn(100)
        windowed = apply_window(data, window_type="rect")
        np.testing.assert_array_equal(windowed, data)

    def test_hann_window_tapers_edges(self):
        data = np.ones(100)
        windowed = apply_window(data, window_type="hann")
        self.assertAlmostEqual(windowed[0], 0.0, places=6)

    def test_invalid_window_type_raises(self):
        with self.assertRaises(ValueError):
            apply_window(np.random.randn(10), window_type="bogus")


if __name__ == "__main__":
    unittest.main(verbosity=2)
