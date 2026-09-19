#!/usr/bin/env python3
"""Collect ViT ImageNet-C metrics and batch diagnostics for EATA/SAR objective combinations."""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

import pandas as pd

METHODS = [
    "eata_tent",
    "eata_tent_come",
    "eata_adaps_come",
    "sar_tent",
    "sar_tent_come",
    "sar_adaps_come",
]
METHOD_DISPLAY = {
    "eata_tent": "EATA + Tent",
    "eata_tent_come": "EATA + Tent-COME",
    "eata_adaps_come": "EATA + AdaPS-COME",
    "sar_tent": "SAR + Tent",
    "sar_tent_come": "SAR + Tent-COME",
    "sar_adaps_come": "SAR + AdaPS-COME",
}


def _read_csvs(files: List[Path]) -> pd.DataFrame:
    frames = []
    for f in files:
        try:
            df = pd.read_csv(f)
            df["source_file"] = str(f)
            frames.append(df)
        except Exception as e:
            print(f"Skip {f}: {e}")
    return pd.concat(frames, ignore_index=True, sort=False) if frames else pd.DataFrame()


def _sort_imagenet_c(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    df = df.copy()
    if "severity" in df.columns:
        df["_severity_num"] = pd.to_numeric(df["severity"], errors="coerce").fillna(-1).astype(int)
    if "batch_idx" in df.columns:
        df["_batch_idx_num"] = pd.to_numeric(df["batch_idx"], errors="coerce").fillna(-1).astype(int)
    if "combo_index" in df.columns:
        df["_combo_index_num"] = pd.to_numeric(df["combo_index"], errors="coerce").fillna(-1).astype(int)
    sort_cols = []
    if "corruption" in df.columns:
        sort_cols.append("corruption")
    if "_severity_num" in df.columns:
        sort_cols.append("_severity_num")
    if "_combo_index_num" in df.columns:
        sort_cols.append("_combo_index_num")
    if "_batch_idx_num" in df.columns:
        sort_cols.append("_batch_idx_num")
    if sort_cols:
        df = df.sort_values(sort_cols, kind="mergesort").reset_index(drop=True)
    df["stream_batch_idx"] = range(len(df))
    return df.drop(columns=[c for c in df.columns if c.startswith("_")])


def _metric_files(ds_dir: Path, seed: int) -> List[Path]:
    files = []
    for f in ds_dir.glob(f"*seed{seed}.csv"):
        name = f.name
        if "batch_metrics" in name or "sample_metrics" in name or name.startswith("combined_"):
            continue
        files.append(f)
    return sorted(files)


def _batch_files(ds_dir: Path, seed: int) -> List[Path]:
    return sorted([
        f for f in ds_dir.glob(f"*seed{seed}_batch_metrics.csv")
        if not f.name.startswith("combined_")
    ])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", type=str, default="/data/yuehan/COME_EATA_SAR_imagenet-c/Results")
    parser.add_argument("--backbone", type=str, default="vit_base_patch16_224")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--methods", type=str, default=",".join(METHODS))
    args = parser.parse_args()

    results = Path(args.results_dir)
    metric_out = results / "MethodMetrics" / args.backbone
    batch_out = results / "BatchDiagnostics" / args.backbone
    metric_out.mkdir(parents=True, exist_ok=True)
    batch_out.mkdir(parents=True, exist_ok=True)

    all_metric_frames = []
    all_batch_frames = []
    methods = [m.strip() for m in args.methods.split(",") if m.strip()]
    for method in methods:
        ds_dir = results / args.backbone / method / "imagenet-c"

        metrics = _read_csvs(_metric_files(ds_dir, args.seed))
        if not metrics.empty:
            metrics["method"] = method
            metrics["method_display"] = METHOD_DISPLAY.get(method, method)
            if "tta_model" not in metrics.columns:
                metrics["tta_model"] = method.split("_")[0]
            if "objective_method" not in metrics.columns:
                metrics["objective_method"] = method.replace(metrics["tta_model"].iloc[0] + "_", "")
            metrics = _sort_imagenet_c(metrics)
            out = metric_out / f"{method}_imagenet_c_metrics_seed{args.seed}.csv"
            metrics.to_csv(out, index=False)
            all_metric_frames.append(metrics)
            print(f"Wrote metrics: {out} rows={len(metrics)}")
        else:
            print(f"WARNING: no ImageNet-C metric files found for {method} under {ds_dir}")

        batch = _read_csvs(_batch_files(ds_dir, args.seed))
        if not batch.empty:
            batch["method"] = method
            batch["method_display"] = METHOD_DISPLAY.get(method, method)
            if "tta_model" not in batch.columns:
                batch["tta_model"] = method.split("_")[0]
            if "objective_method" not in batch.columns:
                batch["objective_method"] = method.replace(batch["tta_model"].iloc[0] + "_", "")
            batch = _sort_imagenet_c(batch)
            out = batch_out / f"{method}_imagenet_c_batch_diagnostics_seed{args.seed}.csv"
            batch.to_csv(out, index=False)
            all_batch_frames.append(batch)
            print(f"Wrote batch diagnostics: {out} rows={len(batch)}")
        else:
            print(f"WARNING: no ImageNet-C batch metric files found for {method} under {ds_dir}")

    if all_metric_frames:
        all_metrics = pd.concat(all_metric_frames, ignore_index=True, sort=False)
        out = metric_out / f"all_methods_imagenet_c_metrics_seed{args.seed}.csv"
        all_metrics.to_csv(out, index=False)
        print(f"Wrote all-method metrics: {out} rows={len(all_metrics)}")
    if all_batch_frames:
        all_batch = pd.concat(all_batch_frames, ignore_index=True, sort=False)
        out = batch_out / f"all_methods_imagenet_c_batch_diagnostics_seed{args.seed}.csv"
        all_batch.to_csv(out, index=False)
        print(f"Wrote all-method batch diagnostics: {out} rows={len(all_batch)}")


if __name__ == "__main__":
    main()
