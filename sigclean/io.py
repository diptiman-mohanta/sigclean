"""
Readers for common biomedical signal file formats.

WFDB (PhysioNet) and EDF support require optional extra dependencies
(``pip install sigclean[io]``) and are imported lazily so that importing
`sigclean` itself never requires them.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np
import pandas as pd
from numpy.typing import NDArray


def read_csv_column(
    path: str,
    column: int = 2,
    fs: float = 1000.0,
    drop_invalid: bool = True,
) -> Tuple[NDArray[np.float64], float]:
    """
    Read a single signal channel from a CSV file.

    Parameters
    -----------
    path : str
        Path to the CSV file.
    column : int
        0-based column index to read (default: 2, the 3rd column).
    fs : float
        Sampling frequency of the data in Hz. CSV files don't generally
        self-describe their sampling rate, so this must be supplied by the
        caller based on how the data was acquired.
    drop_invalid : bool
        If True (default), drop NaN/infinite samples from the column.

    Returns
    --------
    data : ndarray
        The requested column's values.
    fs : float
        The sampling frequency passed in, returned for convenience so this
        function's output can be unpacked directly into pipeline calls.
    """
    df = pd.read_csv(path)

    if column >= df.shape[1]:
        raise ValueError(
            f"Column index {column} doesn't exist; CSV has {df.shape[1]} columns (0-indexed)."
        )

    data = df.iloc[:, column].to_numpy(dtype=np.float64)

    if drop_invalid:
        data = data[np.isfinite(data)]

    if len(data) == 0:
        raise ValueError(f"No valid data found in column {column} of '{path}'.")

    return data, fs


def read_wfdb(record_name: str, channel: int = 0) -> Tuple[NDArray[np.float64], float]:
    """
    Read a single channel from a PhysioNet WFDB record.

    Parameters
    -----------
    record_name : str
        Path to the WFDB record, without file extension (e.g. a path to
        `100.dat`/`100.hea` should be passed as `.../100`).
    channel : int
        0-based index of the channel to read.

    Returns
    --------
    data : ndarray
        The requested channel's samples.
    fs : float
        Sampling frequency in Hz, from the record header.

    Raises
    -------
    ImportError
        If the optional `wfdb` dependency is not installed
        (`pip install sigclean[io]`).
    """
    try:
        import wfdb
    except ImportError as exc:
        raise ImportError(
            "Reading WFDB records requires the 'wfdb' package. "
            "Install it with: pip install sigclean[io]"
        ) from exc

    record = wfdb.rdrecord(record_name)
    if channel >= record.p_signal.shape[1]:
        raise ValueError(
            f"Channel index {channel} doesn't exist; record has "
            f"{record.p_signal.shape[1]} channels (0-indexed)."
        )

    data = np.asarray(record.p_signal[:, channel], dtype=np.float64)
    return data, float(record.fs)


def read_edf(path: str, channel: int = 0) -> Tuple[NDArray[np.float64], float]:
    """
    Read a single channel from an EDF (European Data Format) file, commonly
    used for EEG/polysomnography recordings.

    Parameters
    -----------
    path : str
        Path to the .edf file.
    channel : int
        0-based index of the channel to read.

    Returns
    --------
    data : ndarray
        The requested channel's samples.
    fs : float
        Sampling frequency in Hz, from the file header.

    Raises
    -------
    ImportError
        If the optional `pyedflib` dependency is not installed
        (`pip install sigclean[io]`).
    """
    try:
        import pyedflib
    except ImportError as exc:
        raise ImportError(
            "Reading EDF files requires the 'pyedflib' package. "
            "Install it with: pip install sigclean[io]"
        ) from exc

    reader = pyedflib.EdfReader(path)
    try:
        if channel >= reader.signals_in_file:
            raise ValueError(
                f"Channel index {channel} doesn't exist; file has "
                f"{reader.signals_in_file} channels (0-indexed)."
            )
        data = np.asarray(reader.readSignal(channel), dtype=np.float64)
        fs = float(reader.getSampleFrequency(channel))
    finally:
        reader.close()

    return data, fs
