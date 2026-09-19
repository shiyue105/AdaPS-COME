#!/usr/bin/env python3
"""Aggregate per-corruption ImageNet-C CSV results."""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


KEY_METRICS = [
    "accuracy",
    "top5_accuracy",
    "mean_uncertainty",
    "wrong_uncertainty",
    "correct_uncertainty",
    "uncertainty_gap_wrong_minus_correct",
    "mean_confidence",
    "wrong_confidence",
    "correct_confidence",
    "confidence_gap_correct_minus_wrong",
    "error_auroc_using_uncertainty",
    "error_auprc_using_uncertainty",
    "error_fpr95_using_uncertainty",
    "ece_15bins_percent",
    "mean_js_divergence",
    "mean_reliability",
    "mean_prediction_set_size",
    "mean_sample_weight",
    "high_group_ratio",
    "high_group_accuracy",
    "middle_group_ratio",
    "middle_group_accuracy",
    "low_group_ratio",
    "low_group_accuracy",
]


def load_combined_csvs(results_dir: Path):
    files = sorted(results_dir.rglob("combined_*_seed*.csv"))
    frames = []
    for f in files:
        try:
            df = pd.read_csv(f)
            df["source_file"] = str(f)
            frames.append(df)
        except Exception as e:
            print(f"Skip {f}: {e}")
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True, sort=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", type=str, default="/data/yuehan/COME_optimized_imagenet-c/Results")
    parser.add_argument("--out_dir", type=str, default="/data/yuehan/COME_optimized_imagenet-c/Results/Aggregated")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = load_combined_csvs(results_dir)
    if df.empty:
        raise RuntimeError(f"No combined CSV files found under {results_dir}")

    df.to_csv(out_dir / "aggregate_summary_all_combos.csv", index=False)

    metrics = [m for m in KEY_METRICS if m in df.columns]
    keys1 = ["backbone", "method", "severity"]
    keys2 = ["backbone", "method"]
    for col in ["severity"]:
        if col in df.columns:
            df[col] = df[col].astype(str)

    by_sev = df.groupby(keys1, dropna=False)[metrics].mean(numeric_only=True).reset_index()
    by_method = df.groupby(keys2, dropna=False)[metrics].mean(numeric_only=True).reset_index()

    by_sev.to_csv(out_dir / "mean_by_backbone_method_severity.csv", index=False)
    by_method.to_csv(out_dir / "mean_by_backbone_method.csv", index=False)

    print("Saved:")
    print(out_dir / "aggregate_summary_all_combos.csv")
    print(out_dir / "mean_by_backbone_method_severity.csv")
    print(out_dir / "mean_by_backbone_method.csv")
    print("\nMean by backbone/method:")
    show_cols = ["backbone", "method"] + [c for c in ["accuracy", "mean_uncertainty", "error_auroc_using_uncertainty", "mean_confidence"] if c in by_method.columns]
    print(by_method[show_cols].to_string(index=False))


if __name__ == "__main__":
    main()
