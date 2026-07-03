# SigClean

SigClean is a Python library for cleaning and preprocessing biomedical
signals -- ECG, EMG, EEG, and other physiological signals. It provides
filtering, artifact removal, noise reduction, signal quality assessment,
heart-rate-variability metrics, and a composable pipeline API with built-in
multi-channel support.

```{toctree}
:maxdepth: 2
:caption: Contents

quickstart
api
```

## Installation

```bash
pip install sigclean

# with WFDB/EDF file format support
pip install sigclean[io]
```

## Quick example

```python
import numpy as np
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

See {doc}`quickstart` for a full walkthrough and {doc}`api` for the complete
function/class reference.
