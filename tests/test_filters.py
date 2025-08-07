"""
Unit tests for sigclean.filters module.

This module contains comprehensive tests for all filtering functions
in the sigclean library.
"""

import unittest
import numpy as np
import sys
import os

# Add parent directory to path to import sigclean
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sigclean.filters import (
    butterworth_filter,
    notch_filter,
    baseline_removal,
    artifact_removal,
    adaptive_filter,
    moving_average_filter,
    savitzky_golay_filter
)


class TestFilters(unittest.TestCase):
    """Test cases for signal filtering functions."""
    
    def setUp(self):
        """Set up test data."""
        # Create synthetic test signals
        self.fs = 1000  # Sampling frequency
        self.duration = 2  # seconds
        self.t = np.linspace(0, self.duration, self.fs * self.duration)
        
        # Clean sine wave
        self.clean_signal = np.sin(2 * np.pi * 10 * self.t)  # 10 Hz sine wave
        
        # Noisy signal (clean signal + noise + powerline interference)
        np.random.seed(42)  # For reproducible tests
        noise = 0.1 * np.random.randn(len(self.t))
        powerline = 0.2 * np.sin(2 * np.pi * 50 * self.t)  # 50 Hz interference
        self.noisy_signal = self.clean_signal + noise + powerline
        
        # Signal with baseline drift
        baseline_drift = 0.5 * self.t  # Linear drift
        self.drifted_signal = self.clean_signal + baseline_drift
        
        # Signal with artifacts
        self.artifact_signal = self.clean_signal.copy()
        artifact_indices = [100, 500, 1000, 1500]  # Add spike artifacts
        self.artifact_signal[artifact_indices] += 5.0
    
    def test_butterworth_lowpass(self):
        """Test Butterworth lowpass filter."""
        filtered = butterworth_filter(
            self.noisy_signal, self.fs, 
            filter_type='lowpass', high_freq=30
        )
        
        # Check that output length matches input
        self.assertEqual(len(filtered), len(self.noisy_signal))
        
        # Check that filter reduces high-frequency noise
        # (This is a basic check - more sophisticated validation could be added)
        self.assertIsInstance(filtered, np.ndarray)
    
    def test_butterworth_highpass(self):
        """Test Butterworth highpass filter."""
        filtered = butterworth_filter(
            self.drifted_signal, self.fs,
            filter_type='highpass', low_freq=1
        )
        
        self.assertEqual(len(filtered), len(self.drifted_signal))
        self.assertIsInstance(filtered, np.ndarray)
    
    def test_butterworth_bandpass(self):
        """Test Butterworth bandpass filter."""
        filtered = butterworth_filter(
            self.noisy_signal, self.fs,
            filter_type='bandpass', low_freq=5, high_freq=15
        )
        
        self.assertEqual(len(filtered), len(self.noisy_signal))
        self.assertIsInstance(filtered, np.ndarray)
    
    def test_butterworth_bandstop(self):
        """Test Butterworth bandstop filter."""
        filtered = butterworth_filter(
            self.noisy_signal, self.fs,
            filter_type='bandstop', low_freq=45, high_freq=55
        )
        
        self.assertEqual(len(filtered), len(self.noisy_signal))
        self.assertIsInstance(filtered, np.ndarray)
    
    def test_butterworth_invalid_params(self):
        """Test Butterworth filter with invalid parameters."""
        # Test invalid filter type
        with self.assertRaises(ValueError):
            butterworth_filter(self.clean_signal, self.fs, filter_type='invalid')
        
        # Test missing frequency parameters
        with self.assertRaises(ValueError):
            butterworth_filter(self.clean_signal, self.fs, filter_type='lowpass')
        
        # Test invalid frequency order
        with self.assertRaises(ValueError):
            butterworth_filter(
                self.clean_signal, self.fs,
                filter_type='bandpass', low_freq=50, high_freq=10
            )
    
    def test_notch_filter(self):
        """Test notch filter for powerline interference removal."""
        filtered = notch_filter(self.noisy_signal, self.fs, freq=50.0)
        
        self.assertEqual(len(filtered), len(self.noisy_signal))
        self.assertIsInstance(filtered, np.ndarray)
        
        # Test with different quality factors
        filtered_high_q = notch_filter(self.noisy_signal, self.fs, freq=50.0, quality_factor=60)
        self.assertEqual(len(filtered_high_q), len(self.noisy_signal))
    
    def test_baseline_removal_detrend(self):
        """Test baseline removal using detrend method."""
        corrected = baseline_removal(self.drifted_signal, method='detrend')
        
        self.assertEqual(len(corrected), len(self.drifted_signal))
        self.assertIsInstance(corrected, np.ndarray)
        
        # Check that mean is approximately zero after detrending
        self.assertLess(abs(np.mean(corrected)), 0.1)
    
    def test_baseline_removal_median(self):
        """Test baseline removal using median method."""
        corrected = baseline_removal(self.drifted_signal, method='median', window_size=100)
        
        self.assertEqual(len(corrected), len(self.drifted_signal))
        self.assertIsInstance(corrected, np.ndarray)
    
    def test_baseline_removal_polynomial(self):
        """Test baseline removal using polynomial method."""
        corrected = baseline_removal(self.drifted_signal, method='polynomial', polynomial_order=2)
        
        self.assertEqual(len(corrected), len(self.drifted_signal))
        self.assertIsInstance(corrected, np.ndarray)
    
    def test_baseline_removal_invalid_method(self):
        """Test baseline removal with invalid method."""
        with self.assertRaises(ValueError):
            baseline_removal(self.drifted_signal, method='invalid')
    
    def test_artifact_removal(self):
        """Test artifact removal function."""
        cleaned, artifact_mask = artifact_removal(self.artifact_signal, self.fs)
        
        self.assertEqual(len(cleaned), len(self.artifact_signal))
        self.assertEqual(len(artifact_mask), len(self.artifact_signal))
        self.assertIsInstance(cleaned, np.ndarray)
        self.assertIsInstance(artifact_mask, np.ndarray)
        self.assertEqual(artifact_mask.dtype, bool)
        
        # Check that some artifacts were detected
        self.assertTrue(np.any(artifact_mask))
    
    def test_adaptive_filter(self):
        """Test adaptive filter."""
        filtered, error = adaptive_filter(self.noisy_signal)
        
        self.assertEqual(len(filtered), len(self.noisy_signal))
        self.assertEqual(len(error), len(self.noisy_signal))
        self.assertIsInstance(filtered, np.ndarray)
        self.assertIsInstance(error, np.ndarray)
    
    def test_adaptive_filter_with_reference(self):
        """Test adaptive filter with reference signal."""
        # Create a reference signal (delayed version of noise)
        reference = np.roll(self.noisy_signal, 5)
        filtered, error = adaptive_filter(self.noisy_signal, reference)
        
        self.assertEqual(len(filtered), len(self.noisy_signal))
        self.assertEqual(len(error), len(self.noisy_signal))
    
    def test_adaptive_filter_mismatched_length(self):
        """Test adaptive filter with mismatched signal lengths."""
        short_reference = self.noisy_signal[:100]
        
        with self.assertRaises(ValueError):
            adaptive_filter(self.noisy_signal, short_reference)
    
    def test_moving_average_filter(self):
        """Test moving average filter."""
        window_size = 10
        smoothed = moving_average_filter(self.noisy_signal, window_size)
        
        self.assertEqual(len(smoothed), len(self.noisy_signal))
        self.assertIsInstance(smoothed, np.ndarray)
        
        # Test with invalid window size
        with self.assertRaises(ValueError):
            moving_average_filter(self.noisy_signal, 0)
    
    def test_moving_average_large_window(self):
        """Test moving average with large window."""
        large_window = len(self.noisy_signal) + 10
        smoothed = moving_average_filter(self.noisy_signal, large_window)
        
        # Should return constant array with mean value
        expected_mean = np.mean(self.noisy_signal)
        np.testing.assert_array_almost_equal(smoothed, expected_mean, decimal=5)
    
    def test_savitzky_golay_filter(self):
        """Test Savitzky-Golay filter."""
        window_length = 21
        polyorder = 3
        filtered = savitzky_golay_filter(self.noisy_signal, window_length, polyorder)
        
        self.assertEqual(len(filtered), len(self.noisy_signal))
        self.assertIsInstance(filtered, np.ndarray)
    
    def test_savitzky_golay_even_window(self):
        """Test Savitzky-Golay filter with even window length."""
        window_length = 20  # Even number
        filtered = savitzky_golay_filter(self.noisy_signal, window_length)
        
        # Should automatically adjust to odd number
        self.assertEqual(len(filtered), len(self.noisy_signal))
    
    def test_savitzky_golay_invalid_params(self):
        """Test Savitzky-Golay filter with invalid parameters."""
        with self.assertRaises(ValueError):
            savitzky_golay_filter(self.noisy_signal, window_length=5, polyorder=5)
    
    def test_filter_with_empty_input(self):
        """Test filters with empty input."""
        empty_signal = np.array([])
        
        # Most filters should handle empty input gracefully
        with self.assertRaises((ValueError, IndexError)):
            butterworth_filter(empty_signal, self.fs, 'lowpass', high_freq=10)
    
    def test_filter_with_single_sample(self):
        """Test filters with single sample input."""
        single_sample = np.array([1.0])
        
        # Some filters might not work with single sample
        with self.assertRaises((ValueError, IndexError)):
            butterworth_filter(single_sample, self.fs, 'lowpass', high_freq=10)
    
    def test_filter_output_types(self):
        """Test that all filters return numpy arrays."""
        # Test various filters
        result1 = butterworth_filter(self.clean_signal, self.fs, 'lowpass', high_freq=50)
        result2 = notch_filter(self.clean_signal, self.fs)
        result3 = baseline_removal(self.clean_signal)
        result4, _ = artifact_removal(self.clean_signal, self.fs)
        result5, _ = adaptive_filter(self.clean_signal)
        result6 = moving_average_filter(self.clean_signal, 5)
        result7 = savitzky_golay_filter(self.clean_signal, 11)
        
        for result in [result1, result2, result3, result4, result5, result6, result7]:
            self.assertIsInstance(result, np.ndarray)
    
    def tearDown(self):
        """Clean up after tests."""
        pass


class TestFilterIntegration(unittest.TestCase):
    """Integration tests for combining multiple filters."""
    
    def setUp(self):
        """Set up test data for integration tests."""
        self.fs = 1000
        self.duration = 1
        self.t = np.linspace(0, self.duration, self.fs * self.duration)
        
        # Complex test signal with multiple issues
        signal_10hz = np.sin(2 * np.pi * 10 * self.t)
        noise = 0.2 * np.random.randn(len(self.t))
        powerline = 0.3 * np.sin(2 * np.pi * 50 * self.t)
        baseline_drift = 0.1 * self.t
        
        self.complex_signal = signal_10hz + noise + powerline + baseline_drift
    
    def test_complete_preprocessing_pipeline(self):
        """Test complete preprocessing pipeline."""
        # Step 1: Remove baseline drift
        step1 = baseline_removal(self.complex_signal, method='detrend')
        
        # Step 2: Remove powerline interference
        step2 = notch_filter(step1, self.fs, freq=50.0)
        
        # Step 3: Apply bandpass filter
        step3 = butterworth_filter(step2, self.fs, 'bandpass', low_freq=1, high_freq=30)
        
        # Step 4: Remove artifacts
        step4, artifact_mask = artifact_removal(step3, self.fs)
        
        # Step 5: Smooth the signal
        final_signal = moving_average_filter(step4, 5)
        
        # Verify each step produces valid output
        for step in [step1, step2, step3, step4, final_signal]:
            self.assertEqual(len(step), len(self.complex_signal))
            self.assertIsInstance(step, np.ndarray)
            self.assertFalse(np.any(np.isnan(step)))
            self.assertFalse(np.any(np.isinf(step)))


if __name__ == '__main__':
    # Run the tests
    unittest.main(verbosity=2)
