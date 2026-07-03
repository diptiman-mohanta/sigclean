# SigClean

SigClean is a Python library for cleaning and preprocessing biomedical
signals -- ECG, EMG, EEG, and other physiological signals. It provides
signal filtering, artifact removal, noise reduction, signal quality
assessment, heart-rate-variability metrics, a composable processing
pipeline with built-in multi-channel support, and a command-line interface.

## Installation

```bash
pip install -e .              # from a checkout of this repo
pip install -e ".[io]"        # + WFDB/EDF file format support
pip install -e ".[dev]"       # + testing/linting/type-checking tools
```

## Quick example

```python
from sigclean.pipeline import (
    Pipeline, NotchFilter, BandpassFilter, BaselineRemoval,
    ArtifactRemoval, Normalize,
)

pipeline = Pipeline([
    NotchFilter(freq=50.0),
    BandpassFilter(low_freq=0.5, high_freq=40),
    BaselineRemoval(method='detrend'),
    ArtifactRemoval(),
    Normalize(method='zscore'),
])

cleaned = pipeline(raw_signal, fs=1000)
```

`Pipeline` transparently handles multi-channel input too -- pass a 2D
`(n_channels, n_samples)` array and it's processed per-channel and stacked
back together.

## Command line

```bash
sigclean clean ecg.csv --fs 1000 --column 2 --output cleaned.csv
sigclean hrv ecg.csv --fs 1000 --column 2
```

## Package layout

- `sigclean.filters` -- Butterworth/notch filtering, baseline removal,
  artifact removal, adaptive (LMS) filtering, moving average, Savitzky-Golay.
- `sigclean.utils` -- normalization, resampling, segmentation, SNR, peak
  detection, outlier removal, signal quality metrics.
- `sigclean.pipeline` -- `Pipeline` and composable step classes wrapping the
  above, with automatic multi-channel support.
- `sigclean.hrv` -- RR intervals and time-domain HRV metrics (SDNN, RMSSD, pNN50).
- `sigclean.io` -- CSV column reader, plus WFDB/EDF readers (`pip install sigclean[io]`).
- `sigclean.plot` -- signal, spectrum, before/after, spectrogram, quality,
  and filter-response plots.
- `sigclean.cli` -- the `sigclean` command-line tool.

See [docs/quickstart.md](docs/quickstart.md) for a full walkthrough,
[docs/api.md](docs/api.md) for the API reference, and
[examples/demo_csv_pipeline.py](examples/demo_csv_pipeline.py) for a
complete worked example against real CSV data.

## Development

```bash
pip install -e ".[dev]"
pytest                    # run the test suite
mypy sigclean/            # type-check
pre-commit install        # auto-format/lint on commit
```

See [CHANGELOG.md](CHANGELOG.md) for release history.
