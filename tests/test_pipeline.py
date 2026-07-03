"""Tests for sigclean.pipeline."""

import numpy as np
import pytest

from sigclean.pipeline import (
    ArtifactRemoval,
    BandpassFilter,
    BaselineRemoval,
    Normalize,
    NotchFilter,
    Pipeline,
)


@pytest.fixture
def standard_pipeline():
    return Pipeline(
        [
            NotchFilter(freq=50.0),
            BandpassFilter(low_freq=0.5, high_freq=40),
            BaselineRemoval(method="detrend"),
            ArtifactRemoval(),
            Normalize(method="zscore"),
        ]
    )


def test_empty_steps_raises():
    with pytest.raises(ValueError):
        Pipeline([])


def test_call_returns_array_matching_input_length(standard_pipeline, noisy_signal, fs):
    cleaned = standard_pipeline(noisy_signal, fs=fs)
    assert isinstance(cleaned, np.ndarray)
    assert len(cleaned) == len(noisy_signal)
    assert np.all(np.isfinite(cleaned))


def test_run_returns_metadata_for_stateful_steps(standard_pipeline, noisy_signal, fs):
    result = standard_pipeline.run(noisy_signal, fs=fs)
    assert "artifact_removal" in result.metadata
    assert "normalize" in result.metadata
    assert result.metadata["artifact_removal"].dtype == bool
    assert len(result.metadata["artifact_removal"]) == len(noisy_signal)
    assert "mean" in result.metadata["normalize"]


def test_run_omits_metadata_for_stateless_steps(fs, clean_sine_signal):
    # NotchFilter/BandpassFilter/BaselineRemoval return no metadata, so a
    # pipeline made only of those should have an empty metadata dict.
    pipeline = Pipeline(
        [
            NotchFilter(freq=50.0),
            BandpassFilter(low_freq=0.5, high_freq=40),
        ]
    )
    result = pipeline.run(clean_sine_signal, fs=fs)
    assert result.metadata == {}


def test_multichannel_input_preserves_shape(standard_pipeline, multichannel_signal, fs):
    result = standard_pipeline.run(multichannel_signal, fs=fs)
    assert result.data.shape == multichannel_signal.shape


def test_multichannel_metadata_is_list_per_channel(
    standard_pipeline, multichannel_signal, fs
):
    result = standard_pipeline.run(multichannel_signal, fs=fs)
    n_channels = multichannel_signal.shape[0]
    assert len(result.metadata["artifact_removal"]) == n_channels
    assert len(result.metadata["normalize"]) == n_channels


def test_multichannel_matches_looping_1d_manually(
    standard_pipeline, multichannel_signal, fs
):
    # Running the pipeline on a 2D array must give the same result as
    # running it on each row independently -- multi-channel support should
    # not change the per-channel math.
    result_2d = standard_pipeline.run(multichannel_signal, fs=fs)
    for i in range(multichannel_signal.shape[0]):
        expected = standard_pipeline(multichannel_signal[i], fs=fs)
        np.testing.assert_allclose(result_2d.data[i], expected)


def test_invalid_ndim_raises(standard_pipeline, fs):
    with pytest.raises(ValueError):
        standard_pipeline(np.zeros((2, 3, 4)), fs=fs)


def test_plain_callable_step_supported(fs):
    def double(data, fs):
        return data * 2.0

    pipeline = Pipeline([double])
    out = pipeline(np.array([1.0, 2.0, 3.0]), fs=fs)
    np.testing.assert_array_equal(out, [2.0, 4.0, 6.0])
