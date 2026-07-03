# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/).

## [1.1.0] - 2026-07-04

### Added
- `sigclean.pipeline.Pipeline`: a composable pipeline for chaining processing
  steps, with automatic multi-channel support (2D `(n_channels, n_samples)` input).
- `sigclean.hrv`: heart-rate-variability metrics (RR intervals, SDNN, RMSSD,
  pNN50) built on top of `detect_peaks`.
- `sigclean.io`: readers for WFDB and EDF biomedical signal formats
  (optional extras: `pip install sigclean[io]`), plus a `read_csv_column` helper.
- `sigclean` CLI (`sigclean clean ...`) for running the standard cleaning
  pipeline from the command line without writing Python.
- `py.typed` marker so downstream type checkers pick up this package's type hints.
- `LICENSE` file and this `CHANGELOG.md`.
- GitHub Actions CI workflow running tests and mypy on every push/PR.
- `.pre-commit-config.yaml` wiring up the black/isort/flake8/mypy hooks already
  declared (but previously unused) in `pyproject.toml`.
- Sphinx documentation scaffold under `docs/`.
- `tests/conftest.py` shared fixtures and property-based tests (hypothesis)
  covering shape/finiteness invariants across the filtering functions.
- `examples/` directory with the CSV demo script and sample data, moved out
  of the package root.
- `.gitignore` (none existed before).

### Fixed
- `calculate_snr(method='amplitude')` used `10*log10` instead of `20*log10`,
  understating every amplitude-mode SNR by exactly a factor of 2 in dB.
- `butterworth_filter` now builds filters as second-order sections
  (`sosfiltfilt`) instead of transfer-function `(b, a)` + `filtfilt`; the old
  form produced silent all-NaN output for higher-order/narrow-band designs.
- `moving_average_filter` switched from `np.convolve(mode='same')` (which
  implicitly zero-pads and biases edge samples toward zero) to
  `uniform_filter1d(mode='nearest')`.
- `resample_signal`'s returned time axis is now spaced at exactly
  `1/target_fs` instead of drifting from the requested rate.
- `segment_signal(overlap=1.0)` now raises `ValueError` instead of crashing
  with `ZeroDivisionError`; `overlap` is validated to `[0.0, 1.0)`.
- `normalize_signal(method='robust')` now applies the standard 1.4826/0.6745
  consistency constant, matching the MAD convention already used in
  `remove_outliers`.
- Nyquist-frequency validation unified across `filters.py`: out-of-range
  cutoffs always raise `ValueError` instead of sometimes silently clamping
  (with a warning) or, for `notch_filter`, silently passing data through
  unfiltered.
- `baseline_removal(method='polynomial')` centers/scales the sample-index
  axis before `np.polyfit`, avoiding an ill-conditioned Vandermonde matrix on
  long recordings.
- `calculate_signal_quality` no longer swallows real bugs behind bare
  `except:` clauses, and guards against zero-variance input producing NaN
  skewness/kurtosis.
- `artifact_removal` rewritten to use vectorized rolling median/MAD instead
  of a per-sample Python loop over rolling mean/std: ~70x faster on
  realistic signal lengths, and more robust (a rolling mean/std is
  contaminated by the very spike it's testing; the median-based version isn't).
- All docstrings used non-standard `Parameters:`/`Returns:`/`Raises:` section
  headers (with a trailing colon); numpydoc/Napoleon only recognizes the
  colon-less form, so every docstring in the project rendered as broken,
  incorrectly-indented text in the new Sphinx docs build. Fixed across
  `filters.py`, `utils.py`, `plot.py`, `hrv.py`, and `io.py`.
- `flake8`'s `[tool.flake8]` section in `pyproject.toml` was never actually
  read by flake8 (it requires the `Flake8-pyproject` plugin, which wasn't
  installed) -- so the configured 88-char line length and ignore list had
  silently never been enforced. Added the plugin and reformatted the whole
  codebase with `black`/`isort` so `flake8` actually passes now.

### Changed
- Removed the duplicate, drifted `setup.py` in favor of `pyproject.toml`
  (PEP 621) as the single source of packaging metadata.
- Removed the unused `setuptools_scm` build dependency (versioning is
  attribute-based via `sigclean.__version__`, not git-tag-based).
- Added full type hints across `filters.py`, `utils.py`, and `plot.py`;
  `mypy --strict`-style checks (already configured in `pyproject.toml`) now
  pass.
- `adaptive_filter` now warns when no `reference_signal` is supplied,
  since the fallback (a delayed copy of the input) is not true adaptive
  noise cancellation.

## [1.0.0] - 2025-07-24

### Added
- Initial release: `filters` (Butterworth/notch filtering, baseline removal,
  artifact removal, adaptive filtering, moving average, Savitzky-Golay),
  `utils` (normalization, resampling, segmentation, SNR, peak detection,
  outlier removal, signal quality metrics), and `plot` (signal, spectrum,
  before/after, spectrogram, quality-assessment, filter-response plots).
