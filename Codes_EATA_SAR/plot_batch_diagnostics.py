#!/usr/bin/env python3
"""Plot Fig.3-style qualitative batch diagnostics."""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

METHOD_ORDER = ["eata_tent", "eata_tent_come", "eata_adaps_come", "sar_tent", "sar_tent_come", "sar_adaps_come"]
METHOD_LABEL = {
    "eata_tent": "EATA + Tent",
    "eata_tent_come": "EATA + Tent-COME",
    "eata_adaps_come": "EATA + AdaPS-COME",
    "sar_tent": "SAR + Tent",
    "sar_tent_come": "SAR + Tent-COME",
    "sar_adaps_come": "SAR + AdaPS-COME",
}


def smooth(s: pd.Series, window: int) -> pd.Series:
    if window <= 1:
        return s
    return s.rolling(window=window, min_periods=1, center=False).mean()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=str, default="/data/yuehan/COME_EATA_SAR_imagenet-c/Results/BatchDiagnostics/vit_base_patch16_224/all_methods_imagenet_c_batch_diagnostics_seed0.csv")
    parser.add_argument("--dataset", type=str, default="imagenet-c", choices=["imagenet-c", "all"])
    parser.add_argument("--out", type=str, default="/data/yuehan/COME_EATA_SAR_imagenet-c/Results/BatchDiagnostics/vit_base_patch16_224/fig3_batch_diagnostics.png")
    parser.add_argument("--smooth", type=int, default=5)
    parser.add_argument("--max_vertical_lines", type=int, default=50)
    args = parser.parse_args()

    df = pd.read_csv(args.csv)
    if args.dataset != "all":
        df = df[df["dataset"] == args.dataset].copy()
    if df.empty:
        raise RuntimeError(f"No rows found for dataset={args.dataset} in {args.csv}")

    # Re-index within each method after filtering so the x-axis is comparable.
    parts = []
    for method, g in df.groupby("method", sort=False):
        g = g.sort_values("stream_batch_idx").copy()
        g["plot_batch_idx"] = range(len(g))
        parts.append(g)
    df = pd.concat(parts, ignore_index=True)

    fig, axes = plt.subplots(4, 1, figsize=(10.5, 8.2), sharex=True)
    metrics = [
        ("batch_accuracy", "(a) Online Accuracy", "Accuracy (%)"),
        ("batch_mean_uncertainty", r"(b) Batch-level Uncertainty $U_t$", r"$U_t$"),
        ("batch_adaptive_q", r"(c) Adaptive Threshold $q_t$ (AdaPS-COME)", r"$q_t$"),
        ("batch_mean_method_prediction_set_size", r"(d) Mean Prediction Set Size $\bar{s}_t$", r"$\bar{s}_t$"),
    ]

    for ax, (col, title, ylabel) in zip(axes, metrics):
        for method in METHOD_ORDER:
            g = df[df["method"] == method].sort_values("plot_batch_idx")
            if g.empty or col not in g.columns:
                continue
            y = pd.to_numeric(g[col], errors="coerce")
            if y.notna().sum() == 0:
                continue
            line_style = "--" if "adaps_come" in method else "-"
            ax.plot(g["plot_batch_idx"], smooth(y, args.smooth), linestyle=line_style, linewidth=1.4, label=METHOD_LABEL.get(method, method))
        ax.set_title(title, loc="left", fontsize=10, fontweight="bold")
        ax.set_ylabel(ylabel)
        ax.grid(True, linestyle="--", alpha=0.25)

    # Segment boundaries from the first available method.
    ref = df[df["method"] == METHOD_ORDER[0]].sort_values("plot_batch_idx")
    if not ref.empty and {"corruption", "severity"}.issubset(ref.columns):
        seg = ref[["plot_batch_idx", "dataset", "corruption", "severity"]].copy()
        seg["segment"] = seg["dataset"].astype(str) + ":" + seg["corruption"].astype(str) + ":" + seg["severity"].astype(str)
        changes = seg[seg["segment"].ne(seg["segment"].shift(1))]
        if len(changes) <= args.max_vertical_lines:
            for x in changes["plot_batch_idx"].tolist()[1:]:
                for ax in axes:
                    ax.axvline(x, linestyle="--", linewidth=0.8, alpha=0.35)

    axes[0].legend(ncol=3, frameon=False, loc="upper center", bbox_to_anchor=(0.5, 1.48))
    axes[-1].set_xlabel("Time (Batches)")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=300, bbox_inches="tight")
    print(f"Saved figure: {out}")


if __name__ == "__main__":
    main()
