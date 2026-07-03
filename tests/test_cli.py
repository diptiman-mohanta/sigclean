"""Tests for sigclean.cli."""

import json

import numpy as np
import pandas as pd
import pytest

from sigclean.cli import main


@pytest.fixture
def sample_csv(tmp_path, fs):
    path = tmp_path / "sample.csv"
    t = np.arange(5 * fs) / fs
    signal = np.sin(2 * np.pi * 5 * t) + 0.05 * np.random.RandomState(0).randn(len(t))
    df = pd.DataFrame({"Time": t, "Other": signal, "Signal": signal})
    df.to_csv(path, index=False)
    return str(path)


def test_clean_command_writes_output(sample_csv, fs, tmp_path):
    output_path = str(tmp_path / "out.csv")
    exit_code = main(
        ["clean", sample_csv, "--column", "2", "--fs", str(fs), "--output", output_path]
    )
    assert exit_code == 0

    result_df = pd.read_csv(output_path)
    assert "Cleaned_Signal" in result_df.columns
    assert np.all(np.isfinite(result_df["Cleaned_Signal"]))


def test_hrv_command_prints_json(sample_csv, fs, capsys):
    exit_code = main(["hrv", sample_csv, "--column", "2", "--fs", str(fs)])
    assert exit_code == 0
    captured = capsys.readouterr()
    metrics = json.loads(captured.out)
    assert "mean_hr_bpm" in metrics


def test_missing_file_reports_error_not_traceback(capsys):
    exit_code = main(["clean", "does_not_exist.csv", "--fs", "1000"])
    assert exit_code == 1
    captured = capsys.readouterr()
    assert "Error:" in captured.err


def test_no_command_shows_usage():
    with pytest.raises(SystemExit):
        main([])
