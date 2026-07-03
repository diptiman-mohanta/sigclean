# Quickstart

## Cleaning a single-channel signal

```python
import numpy as np
from sigclean.io import read_csv_column
from sigclean.pipeline import (
    Pipeline, NotchFilter, BandpassFilter, BaselineRemoval,
    ArtifactRemoval, Normalize,
)

data, fs = read_csv_column("ecg.csv", column=2, fs=1000.0)

pipeline = Pipeline([
    NotchFilter(freq=50.0),                          # remove power-line hum
    BandpassFilter(low_freq=0.5, high_freq=40),        # keep the ECG band
    BaselineRemoval(method='detrend'),                 # remove drift
    ArtifactRemoval(),                                  # interpolate spikes
    Normalize(method='zscore'),                         # standardize scale
])

cleaned = pipeline(data, fs=fs)
```

Need the artifact mask or normalization parameters too? Use `.run()` instead
of calling the pipeline directly:

```python
result = pipeline.run(data, fs=fs)
result.data                              # the cleaned signal
result.metadata['artifact_removal']      # boolean mask of detected artifacts
result.metadata['normalize']             # {'mean': ..., 'std': ...}
```

## Multi-channel signals

Pass a 2D array shaped `(n_channels, n_samples)` and the same pipeline runs
independently per channel, then stacks the results back together:

```python
multi_channel_data = np.stack([lead_i, lead_ii, lead_iii], axis=0)
result = pipeline.run(multi_channel_data, fs=fs)
result.data.shape                       # (3, n_samples)
result.metadata['artifact_removal']     # list of 3 masks, one per channel
```

## Heart-rate variability

```python
from sigclean.hrv import hrv_metrics

metrics = hrv_metrics(cleaned, fs=fs)
metrics['mean_hr_bpm']
metrics['sdnn_ms']
metrics['rmssd_ms']
```

## Command line

```bash
sigclean clean ecg.csv --fs 1000 --column 2 --output cleaned.csv
sigclean hrv ecg.csv --fs 1000 --column 2
```

## Reading other file formats

```python
from sigclean.io import read_wfdb, read_edf  # requires: pip install sigclean[io]

data, fs = read_wfdb("physionet_records/100")
data, fs = read_edf("recording.edf")
```
