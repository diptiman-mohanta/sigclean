"""Tests for sigclean.io."""

import numpy as np
import pandas as pd
import pytest

from sigclean.io import read_csv_column, read_edf, read_wfdb


@pytest.fixture
def sample_csv(tmp_path, fs):
    path = tmp_path / "sample.csv"
    df = pd.DataFrame(
        {
            "Time": np.arange(100) / fs,
            "Other": np.random.randn(100),
            "Signal": np.concatenate([np.random.randn(98), [np.nan, np.inf]]),
        }
    )
    df.to_csv(path, index=False)
    return str(path)


def test_read_csv_column_reads_requested_column(sample_csv, fs):
    data, returned_fs = read_csv_column(sample_csv, column=2, fs=fs)
    assert returned_fs == fs
    # NaN and inf should have been dropped
    assert len(data) == 98
    assert np.all(np.isfinite(data))


def test_read_csv_column_invalid_column_raises(sample_csv, fs):
    with pytest.raises(ValueError):
        read_csv_column(sample_csv, column=99, fs=fs)


def test_read_csv_column_all_invalid_raises(tmp_path, fs):
    path = tmp_path / "allnan.csv"
    pd.DataFrame({"A": [1, 2], "B": [1, 2], "C": [np.nan, np.nan]}).to_csv(
        path, index=False
    )
    with pytest.raises(ValueError):
        read_csv_column(str(path), column=2, fs=fs)


def test_read_wfdb_raises_clear_import_error_when_missing():
    try:
        import wfdb  # noqa: F401

        pytest.skip("wfdb is installed; import-error path not exercised")
    except ImportError:
        pass
    with pytest.raises(ImportError, match="sigclean\\[io\\]"):
        read_wfdb("some/record")


def test_read_edf_raises_clear_import_error_when_missing():
    try:
        import pyedflib  # noqa: F401

        pytest.skip("pyedflib is installed; import-error path not exercised")
    except ImportError:
        pass
    with pytest.raises(ImportError, match="sigclean\\[io\\]"):
        read_edf("some/file.edf")
