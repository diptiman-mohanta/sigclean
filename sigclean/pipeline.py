"""
Composable processing pipelines for biomedical signals.

Chaining the functions in `sigclean.filters` and `sigclean.utils` by hand
(as in `examples/demo_csv_pipeline.py`) works, but is repetitive and doesn't
compose well across scripts. `Pipeline` wraps that same chain into a single,
reusable, inspectable object, and transparently applies it per-channel when
given a 2D `(n_channels, n_samples)` array (e.g. multi-lead ECG, multi-channel
EEG/EMG) instead of requiring callers to loop by hand.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Protocol, Sequence, Tuple, Union

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .filters import (
    adaptive_filter,
    artifact_removal,
    baseline_removal,
    butterworth_filter,
    moving_average_filter,
    notch_filter,
    savitzky_golay_filter,
)
from .utils import normalize_signal


class PipelineStep(Protocol):
    """Interface a pipeline step must satisfy.

    `name` identifies the step in `PipelineResult.metadata`. `__call__` takes
    the current signal and sampling frequency and returns the transformed
    signal plus optional metadata (e.g. an artifact mask, normalization
    parameters) that isn't itself part of the signal.
    """

    name: str

    def __call__(
        self, data: NDArray[np.float64], fs: float
    ) -> Tuple[NDArray[np.float64], Optional[Any]]: ...


def _as_step(step: Union[PipelineStep, Callable[..., Any]]) -> PipelineStep:
    """Wrap a plain callable ``fn(data, fs) -> data`` (no metadata) as a step."""
    if hasattr(step, "name") and callable(step):
        return step

    fn = step

    @dataclass
    class _CallableStep:
        name: str = getattr(fn, "__name__", "step")

        def __call__(
            self, data: NDArray[np.float64], fs: float
        ) -> Tuple[NDArray[np.float64], Optional[Any]]:
            return np.asarray(fn(data, fs), dtype=np.float64), None

    return _CallableStep()


@dataclass
class NotchFilter:
    """Pipeline step: remove power-line interference. See `notch_filter`."""

    freq: float = 50.0
    quality_factor: float = 30
    name: str = "notch_filter"

    def __call__(
        self, data: NDArray[np.float64], fs: float
    ) -> Tuple[NDArray[np.float64], None]:
        return (
            notch_filter(data, fs, freq=self.freq, quality_factor=self.quality_factor),
            None,
        )


@dataclass
class BandpassFilter:
    """Pipeline step: Butterworth bandpass. See `butterworth_filter`."""

    low_freq: float
    high_freq: float
    order: int = 4
    name: str = "bandpass_filter"

    def __call__(
        self, data: NDArray[np.float64], fs: float
    ) -> Tuple[NDArray[np.float64], None]:
        filtered = butterworth_filter(
            data,
            fs,
            filter_type="bandpass",
            low_freq=self.low_freq,
            high_freq=self.high_freq,
            order=self.order,
        )
        return filtered, None


@dataclass
class LowpassFilter:
    """Pipeline step: Butterworth lowpass. See `butterworth_filter`."""

    high_freq: float
    order: int = 4
    name: str = "lowpass_filter"

    def __call__(
        self, data: NDArray[np.float64], fs: float
    ) -> Tuple[NDArray[np.float64], None]:
        filtered = butterworth_filter(
            data,
            fs,
            filter_type="lowpass",
            high_freq=self.high_freq,
            order=self.order,
        )
        return filtered, None


@dataclass
class HighpassFilter:
    """Pipeline step: Butterworth highpass. See `butterworth_filter`."""

    low_freq: float
    order: int = 4
    name: str = "highpass_filter"

    def __call__(
        self, data: NDArray[np.float64], fs: float
    ) -> Tuple[NDArray[np.float64], None]:
        filtered = butterworth_filter(
            data,
            fs,
            filter_type="highpass",
            low_freq=self.low_freq,
            order=self.order,
        )
        return filtered, None


@dataclass
class BandstopFilter:
    """Pipeline step: Butterworth bandstop. See `butterworth_filter`."""

    low_freq: float
    high_freq: float
    order: int = 4
    name: str = "bandstop_filter"

    def __call__(
        self, data: NDArray[np.float64], fs: float
    ) -> Tuple[NDArray[np.float64], None]:
        filtered = butterworth_filter(
            data,
            fs,
            filter_type="bandstop",
            low_freq=self.low_freq,
            high_freq=self.high_freq,
            order=self.order,
        )
        return filtered, None


@dataclass
class BaselineRemoval:
    """Pipeline step: baseline drift removal. See `baseline_removal`."""

    method: str = "detrend"
    window_size: Optional[int] = None
    polynomial_order: int = 3
    name: str = "baseline_removal"

    def __call__(
        self, data: NDArray[np.float64], fs: float
    ) -> Tuple[NDArray[np.float64], None]:
        corrected = baseline_removal(
            data,
            method=self.method,
            window_size=self.window_size,
            polynomial_order=self.polynomial_order,
        )
        return corrected, None


@dataclass
class ArtifactRemoval:
    """Pipeline step: robust artifact detection/interpolation.

    Metadata is the boolean artifact mask returned by `artifact_removal`.
    """

    threshold_std: float = 3.0
    window_size: Optional[int] = None
    name: str = "artifact_removal"

    def __call__(
        self, data: NDArray[np.float64], fs: float
    ) -> Tuple[NDArray[np.float64], NDArray[np.bool_]]:
        cleaned, mask = artifact_removal(
            data,
            fs,
            threshold_std=self.threshold_std,
            window_size=self.window_size,
        )
        return cleaned, mask


@dataclass
class MovingAverage:
    """Pipeline step: moving-average smoothing. See `moving_average_filter`."""

    window_size: int
    name: str = "moving_average"

    def __call__(
        self, data: NDArray[np.float64], fs: float
    ) -> Tuple[NDArray[np.float64], None]:
        return moving_average_filter(data, self.window_size), None


@dataclass
class SavitzkyGolay:
    """Pipeline step: Savitzky-Golay smoothing. See `savitzky_golay_filter`."""

    window_length: int
    polyorder: int = 3
    name: str = "savitzky_golay"

    def __call__(
        self, data: NDArray[np.float64], fs: float
    ) -> Tuple[NDArray[np.float64], None]:
        return savitzky_golay_filter(data, self.window_length, self.polyorder), None


@dataclass
class AdaptiveFilter:
    """Pipeline step: LMS adaptive filtering. See `adaptive_filter`."""

    reference_signal: Optional[ArrayLike] = None
    filter_length: int = 32
    mu: float = 0.01
    name: str = "adaptive_filter"

    def __call__(
        self, data: NDArray[np.float64], fs: float
    ) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
        filtered, error = adaptive_filter(
            data,
            reference_signal=self.reference_signal,
            filter_length=self.filter_length,
            mu=self.mu,
        )
        return filtered, error


@dataclass
class Normalize:
    """Pipeline step: signal normalization. See `normalize_signal`.

    Metadata is the `norm_params` dict returned by `normalize_signal`.
    """

    method: str = "zscore"
    feature_range: Tuple[float, float] = (0, 1)
    name: str = "normalize"

    def __call__(
        self, data: NDArray[np.float64], fs: float
    ) -> Tuple[NDArray[np.float64], Dict[str, Any]]:
        normalized, params = normalize_signal(
            data,
            method=self.method,
            feature_range=self.feature_range,
        )
        return normalized, params


StepLike = Union[PipelineStep, Callable[[NDArray[np.float64], float], ArrayLike]]


@dataclass
class PipelineResult:
    """Output of `Pipeline.run`.

    `data` is the final signal: 1D if the input was 1D, or 2D
    `(n_channels, n_samples)` if the input was 2D. `metadata[step_name]` holds
    whatever that step returned besides the signal itself (e.g. an artifact
    mask, or normalization parameters) -- a list of per-channel values if the
    input was multi-channel, or the bare value for single-channel input.
    """

    data: NDArray[np.float64]
    metadata: Dict[str, Any] = field(default_factory=dict)


class Pipeline:
    """A reusable, ordered sequence of signal-processing steps.

    Examples
    --------
    >>> pipeline = Pipeline([
    ...     NotchFilter(freq=50.0),
    ...     BandpassFilter(low_freq=0.5, high_freq=40),
    ...     BaselineRemoval(method='detrend'),
    ...     ArtifactRemoval(),
    ...     Normalize(method='zscore'),
    ... ])
    >>> cleaned = pipeline(raw_ecg, fs=1000)               # just the signal
    >>> result = pipeline.run(raw_ecg, fs=1000)            # signal + metadata
    >>> result.metadata['artifact_removal']                # boolean mask

    Multi-channel input (shape `(n_channels, n_samples)`) is run through the
    same step sequence independently per channel and stacked back together;
    per-step metadata becomes a list with one entry per channel.
    """

    def __init__(self, steps: Sequence[StepLike]) -> None:
        if not steps:
            raise ValueError("Pipeline requires at least one step")
        self.steps: List[PipelineStep] = [_as_step(s) for s in steps]

    def _run_1d(
        self, data: NDArray[np.float64], fs: float
    ) -> Tuple[NDArray[np.float64], Dict[str, Any]]:
        current = data
        metadata: Dict[str, Any] = {}
        for step in self.steps:
            current, meta = step(current, fs)
            current = np.asarray(current, dtype=np.float64)
            if meta is not None:
                metadata[step.name] = meta
        return current, metadata

    def run(self, data: ArrayLike, fs: float) -> PipelineResult:
        """Run every step in order and return the signal plus step metadata."""
        arr = np.asarray(data, dtype=np.float64)

        if arr.ndim == 1:
            final_data, metadata = self._run_1d(arr, fs)
            return PipelineResult(data=final_data, metadata=metadata)

        if arr.ndim == 2:
            per_channel_data = []
            per_channel_metadata: List[Dict[str, Any]] = []
            for channel in arr:
                channel_data, channel_meta = self._run_1d(channel, fs)
                per_channel_data.append(channel_data)
                per_channel_metadata.append(channel_meta)

            stacked = np.stack(per_channel_data, axis=0)
            combined_metadata: Dict[str, Any] = {
                step.name: [meta.get(step.name) for meta in per_channel_metadata]
                for step in self.steps
                if any(step.name in meta for meta in per_channel_metadata)
            }
            return PipelineResult(data=stacked, metadata=combined_metadata)

        raise ValueError(
            f"Pipeline expects 1D (samples,) or 2D (channels, samples) input, got shape {arr.shape}"
        )

    def __call__(self, data: ArrayLike, fs: float) -> NDArray[np.float64]:
        """Shortcut for `run(data, fs).data`, for callers who don't need metadata."""
        return self.run(data, fs).data
