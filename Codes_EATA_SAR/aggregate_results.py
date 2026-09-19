#!/usr/bin/env python3
"""Aggregate ImageNet-C method metrics by severity and method."""
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
    "mean_prediction_set_size",
    "mean_sample_weight",
]


def load_metric_csvs(results_dir: Path) -> pd.DataFrame:
    files = sorted((results_dir / "MethodMetrics" / "vit_base_patch16_224").glob("*_imagenet_c_metrics_seed*.csv"))
    files = [f for f in files if not f.name.startswith("all_methods_")]
    frames = []
    for f in files:
        try:
            df = pd.read_csv(f)
            df["source_file"] = str(f)
            frames.append(df)
        except Exception as e:
            print(f"Skip {f}: {e}")
    return pd.concat(frames, ignore_index=True, sort=False) if frames else pd.DataFrame()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", type=str, default="/data/yuehan/COME_EATA_SAR_imagenet-c/Results")
    parser.add_argument("--out_dir", type=str, default="/data/yuehan/COME_EATA_SAR_imagenet-c/Results/Aggregated")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = load_metric_csvs(results_dir)
    if df.empty:
        raise RuntimeError(f"No method metric CSV files found under {results_dir / 'MethodMetrics'}")

    df.to_csv(out_dir / "aggregate_summary_all_combos.csv", index=False)
    metrics = [m for m in KEY_METRICS if m in df.columns]
    if "severity" in df.columns:
        df["severity"] = pd.to_numeric(df["severity"], errors="coerce")

    by_sev = df.groupby(["backbone", "method", "severity"], dropna=False)[metrics].mean(numeric_only=True).reset_index()
    by_method = df.groupby(["backbone", "method"], dropna=False)[metrics].mean(numeric_only=True).reset_index()

    by_sev.to_csv(out_dir / "mean_by_backbone_method_severity.csv", index=False)
    by_method.to_csv(out_dir / "mean_by_backbone_method.csv", index=False)

    print("Saved:")
    print(out_dir / "aggregate_summary_all_combos.csv")
    print(out_dir / "mean_by_backbone_method_severity.csv")
    print(out_dir / "mean_by_backbone_method.csv")
    show_cols = ["backbone", "method"] + [c for c in ["accuracy", "mean_uncertainty", "ece_15bins_percent", "mean_confidence"] if c in by_method.columns]
    print("\nMean by backbone/method:")
    print(by_method[show_cols].to_string(index=False))


if __name__ == "__main__":
    main()
