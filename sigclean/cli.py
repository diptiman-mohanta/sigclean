"""
Command-line interface for SigClean.

Installed as the `sigclean` console script (see `pyproject.toml`
`[project.scripts]`). Run `sigclean --help` for usage.
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import List, Optional, Sequence

import numpy as np

from .hrv import hrv_metrics
from .io import read_csv_column
from .pipeline import (
    ArtifactRemoval,
    BandpassFilter,
    BaselineRemoval,
    Normalize,
    NotchFilter,
    Pipeline,
    StepLike,
)


def _build_clean_pipeline(args: argparse.Namespace) -> Pipeline:
    steps: List[StepLike] = [NotchFilter(freq=freq) for freq in args.notch_freq]
    steps.append(
        BandpassFilter(low_freq=args.bandpass_low, high_freq=args.bandpass_high)
    )
    steps.append(BaselineRemoval(method=args.baseline_method))
    steps.append(ArtifactRemoval(threshold_std=args.artifact_threshold))
    if args.normalize_method != "none":
        steps.append(Normalize(method=args.normalize_method))
    return Pipeline(steps)


def _cmd_clean(args: argparse.Namespace) -> int:
    data, fs = read_csv_column(args.input, column=args.column, fs=args.fs)
    pipeline = _build_clean_pipeline(args)
    result = pipeline.run(data, fs)

    output_path = args.output or f"cleaned_{args.input.rsplit('/', 1)[-1]}"
    import pandas as pd

    output_df = pd.DataFrame(
        {
            "Time": np.arange(len(result.data)) / fs,
            "Original_Signal": data[: len(result.data)],
            "Cleaned_Signal": result.data,
        }
    )
    if "artifact_removal" in result.metadata:
        output_df["Artifact"] = result.metadata["artifact_removal"].astype(int)
    output_df.to_csv(output_path, index=False)

    print(f"Cleaned signal written to: {output_path}")
    if "artifact_removal" in result.metadata:
        mask = result.metadata["artifact_removal"]
        print(
            f"Artifacts detected: {int(mask.sum())} samples ({mask.mean() * 100:.2f}%)"
        )
    return 0


def _cmd_hrv(args: argparse.Namespace) -> int:
    data, fs = read_csv_column(args.input, column=args.column, fs=args.fs)
    metrics = hrv_metrics(data, fs, min_distance=args.min_distance, height=args.height)
    print(json.dumps(metrics, indent=2))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="sigclean",
        description="Clean and analyze biomedical signals from the command line.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    clean_parser = subparsers.add_parser(
        "clean",
        help="Run the standard cleaning pipeline on a CSV column and save the result.",
    )
    clean_parser.add_argument("input", help="Path to the input CSV file")
    clean_parser.add_argument(
        "--column", type=int, default=2, help="0-based column index (default: 2)"
    )
    clean_parser.add_argument(
        "--fs", type=float, required=True, help="Sampling frequency in Hz"
    )
    clean_parser.add_argument(
        "--output", type=str, default=None, help="Output CSV path"
    )
    clean_parser.add_argument(
        "--notch-freq",
        type=float,
        nargs="*",
        default=[50.0],
        help="Power-line frequencies to remove, in Hz (default: 50.0)",
    )
    clean_parser.add_argument(
        "--bandpass-low", type=float, default=0.5, help="Bandpass low cutoff in Hz"
    )
    clean_parser.add_argument(
        "--bandpass-high", type=float, default=40.0, help="Bandpass high cutoff in Hz"
    )
    clean_parser.add_argument(
        "--baseline-method",
        type=str,
        default="detrend",
        choices=["detrend", "median", "polynomial"],
        help="Baseline removal method",
    )
    clean_parser.add_argument(
        "--artifact-threshold",
        type=float,
        default=3.0,
        help="Artifact detection threshold",
    )
    clean_parser.add_argument(
        "--normalize-method",
        type=str,
        default="zscore",
        choices=["zscore", "minmax", "robust", "unit_vector", "none"],
        help="Normalization method",
    )
    clean_parser.set_defaults(func=_cmd_clean)

    hrv_parser = subparsers.add_parser(
        "hrv",
        help="Detect R-peaks in a CSV column and report heart-rate-variability metrics.",
    )
    hrv_parser.add_argument("input", help="Path to the input CSV file")
    hrv_parser.add_argument(
        "--column", type=int, default=2, help="0-based column index (default: 2)"
    )
    hrv_parser.add_argument(
        "--fs", type=float, required=True, help="Sampling frequency in Hz"
    )
    hrv_parser.add_argument(
        "--min-distance",
        type=float,
        default=0.3,
        help="Minimum peak distance in seconds",
    )
    hrv_parser.add_argument(
        "--height", type=float, default=None, help="Minimum peak height"
    )
    hrv_parser.set_defaults(func=_cmd_hrv)

    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.func(args))
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
