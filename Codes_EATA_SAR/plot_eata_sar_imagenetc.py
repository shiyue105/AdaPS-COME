import argparse
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe


OBJECTIVE_STYLE = {
    "tent":       ("Tent",       "#2F6FDB", "-",  1.75, 0.98, 6),
    "tent_come":  ("Tent-COME",  "#7EAA72", "-.", 1.75, 0.98, 7),
    "adaps_come": ("AdaPS-COME", "#D94A45", "--", 1.80, 0.92, 8),
}
OBJECTIVE_ORDER = ["tent", "tent_come", "adaps_come"]


def normalize(x):
    return str(x).strip().lower()


def pick_col(df, candidates):
    for c in candidates:
        if c in df.columns and not df[c].isna().all():
            return c
    return None


def rolling_mean(values, window):
    values = pd.Series(values)
    if window <= 1:
        return values.to_numpy()
    return values.rolling(window=window, center=True, min_periods=1).mean().to_numpy()


def maybe_to_percent(values):
    values = np.asarray(values, dtype=float)
    finite = values[np.isfinite(values)]
    if len(finite) > 0 and finite.max() <= 1.5:
        return values * 100.0
    return values


def clip_upper(values, q):
    values = np.asarray(values, dtype=float)
    finite = values[np.isfinite(values)]
    if len(finite) == 0 or q is None or q <= 0 or q >= 1:
        return values
    upper = np.quantile(finite, q)
    return np.clip(values, None, upper)


def styled_plot(ax, x, y, color, linestyle, linewidth, alpha, zorder, label=None):
    line, = ax.plot(
        x, y,
        color=color,
        linestyle=linestyle,
        linewidth=linewidth,
        alpha=alpha,
        zorder=zorder,
        label=label,
        solid_capstyle="round",
        dash_capstyle="round",
    )
    line.set_path_effects([
        pe.Stroke(linewidth=linewidth + 0.9, foreground="white", alpha=0.88),
        pe.Normal()
    ])
    return line


def parse_objective_from_method(method_name, tta_name):
    """
    Convert eata_tent / eata_tent_come / eata_adaps_come
    or sar_tent / sar_tent_come / sar_adaps_come
    into tent / tent_come / adaps_come.
    """
    m = normalize(method_name)
    t = normalize(tta_name)

    prefix = t + "_"
    if m.startswith(prefix):
        return m[len(prefix):]

    # fallback: already objective-only
    if m in OBJECTIVE_ORDER:
        return m

    return None


def add_objective_column(df, tta_name):
    """
    Supports two possible CSV formats:
    1) method column contains eata_tent / sar_tent / ...
    2) tta/objective columns exist separately.
    """
    df = df.copy()

    tta_candidates = ["tta", "tta_model", "tta_method", "adapter", "base_tta"]
    obj_candidates = ["objective", "objective_method", "loss_objective"]

    tta_col = pick_col(df, tta_candidates)
    obj_col = pick_col(df, obj_candidates)

    if tta_col is not None and obj_col is not None:
        df[tta_col] = df[tta_col].map(normalize)
        df[obj_col] = df[obj_col].map(normalize)
        df = df[df[tta_col] == normalize(tta_name)].copy()
        df["objective_plot"] = df[obj_col]
        return df

    if "method" not in df.columns:
        raise ValueError("CSV must contain either method column or tta/objective columns.")

    df["method"] = df["method"].map(normalize)
    df["objective_plot"] = df["method"].apply(lambda x: parse_objective_from_method(x, tta_name))
    df = df[df["objective_plot"].isin(OBJECTIVE_ORDER)].copy()

    return df


def build_severity_major_x(df, objective_col, x_col, batches_per_corruption):
    """
    Build x-axis as:
    Severity 1: 19 corruptions × 1563 batches
    Severity 2: 19 corruptions × 1563 batches
    ...
    This avoids depending on the raw concatenation order of the CSV.
    """
    if "severity" not in df.columns or "corruption" not in df.columns:
        print("Warning: no corruption/severity columns. Use existing x_col.")
        df["plot_x"] = df[x_col]
        return df, None

    out_parts = []

    # severity order: numeric 1..5 if possible
    tmp = df.copy()
    tmp["severity_int"] = tmp["severity"].astype(float).astype(int)
    tmp["corruption_str"] = tmp["corruption"].astype(str)

    severity_values = sorted(tmp["severity_int"].dropna().unique().tolist())
    corruption_values = sorted(tmp["corruption_str"].dropna().unique().tolist())

    corr_to_idx = {c: i for i, c in enumerate(corruption_values)}
    sev_to_idx = {s: i for i, s in enumerate(severity_values)}

    for objective, g_obj in tmp.groupby(objective_col):
        for sev, g_sev in g_obj.groupby("severity_int"):
            for corr, g in g_sev.groupby("corruption_str"):
                g = g.copy().sort_values(x_col).reset_index(drop=True)
                g["local_batch_idx"] = np.arange(len(g))

                sev_idx = sev_to_idx[int(sev)]
                corr_idx = corr_to_idx[str(corr)]
                g["plot_x"] = (
                    sev_idx * len(corruption_values) * batches_per_corruption
                    + corr_idx * batches_per_corruption
                    + g["local_batch_idx"]
                )
                out_parts.append(g)

    out = pd.concat(out_parts, ignore_index=True)
    meta = {
        "num_severities": len(severity_values),
        "num_corruptions": len(corruption_values),
        "severity_values": severity_values,
        "corruption_values": corruption_values,
    }
    return out, meta


def add_severity_segment_lines(axes, total_batches, severity_segment_batches):
    boundaries = np.arange(severity_segment_batches, total_batches, severity_segment_batches)
    for bx in boundaries:
        for ax in axes:
            ax.axvline(
                bx,
                color="#9CA3AF",
                linestyle="--",
                linewidth=1.0,
                alpha=0.45,
                zorder=0
            )


def add_severity_labels(ax, total_batches, severity_segment_batches, num_severities, severity_values, y=0.985):
    for i in range(num_severities):
        start = i * severity_segment_batches
        end = min((i + 1) * severity_segment_batches, total_batches)
        center = (start + end) / 2.0
        sev_label = severity_values[i] if i < len(severity_values) else i + 1

        ax.text(
            center,
            y,
            f"Severity {sev_label}",
            transform=ax.get_xaxis_transform(),
            ha="center",
            va="top",
            fontsize=10.5,
            fontweight="semibold",
            color="#374151",
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.82, pad=1.2)
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True)
    parser.add_argument("--dataset", default="imagenet-c")
    parser.add_argument("--tta", choices=["eata", "sar"], required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--smooth_window", type=int, default=301)
    parser.add_argument("--plot_every", type=int, default=5)
    parser.add_argument("--pset_clip_quantile", type=float, default=0.995)
    parser.add_argument("--batches_per_corruption", type=int, default=1563)
    parser.add_argument("--q_ylim_low", type=float, default=0.90)
    parser.add_argument("--q_ylim_high", type=float, default=0.95)
    args = parser.parse_args()

    df = pd.read_csv(args.csv)

    if "dataset" in df.columns:
        df = df[df["dataset"] == args.dataset].copy()

    if len(df) == 0:
        raise ValueError(f"No rows found after filtering dataset={args.dataset}")

    df = add_objective_column(df, args.tta)

    if len(df) == 0:
        raise ValueError(f"No rows found for TTA={args.tta}. Check method names in CSV.")

    x_col = pick_col(df, ["stream_batch_idx", "global_batch_idx", "batch_idx", "local_batch_idx"])
    if x_col is None:
        df = df.reset_index(drop=True)
        df["stream_batch_idx"] = np.arange(len(df))
        x_col = "stream_batch_idx"

    acc_col = pick_col(df, ["batch_accuracy", "accuracy"])
    unc_col = pick_col(df, ["batch_mean_uncertainty", "mean_uncertainty", "uncertainty"])
    q_col = pick_col(df, ["batch_adaptive_q", "adaptive_q", "q_t"])
    pset_col = pick_col(df, ["batch_mean_adaps_prediction_set_size"])

    if acc_col is None:
        raise ValueError("Cannot find accuracy column.")
    if unc_col is None:
        raise ValueError("Cannot find uncertainty column.")
    if q_col is None:
        raise ValueError("Cannot find q_t column.")
    if pset_col is None:
        raise ValueError("Cannot find AdaPS prediction set size column.")

    df, meta = build_severity_major_x(
        df,
        objective_col="objective_plot",
        x_col=x_col,
        batches_per_corruption=args.batches_per_corruption
    )

    objectives_present = sorted(df["objective_plot"].dropna().unique().tolist())
    objectives = [m for m in OBJECTIVE_ORDER if m in objectives_present]

    print("Dataset:", args.dataset)
    print("TTA:", args.tta)
    print("Objectives:", objectives)
    print("Original x_col:", x_col)
    print("plot_x: severity-major order")
    print("acc_col:", acc_col)
    print("unc_col:", unc_col)
    print("q_col:", q_col)
    print("pset_col:", pset_col)
    if meta:
        print("Severity values:", meta["severity_values"])
        print("Number of corruptions:", meta["num_corruptions"])

    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 10.5,
        "axes.titlesize": 14,
        "axes.labelsize": 13,
        "legend.fontsize": 11.5,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "axes.linewidth": 0.9,
        "savefig.dpi": 300,
    })

    fig, axes = plt.subplots(
        4, 1,
        figsize=(14.4, 8.8),
        sharex=True,
        gridspec_kw={"hspace": 0.32}
    )

    legend_handles = []
    legend_labels = []

    for objective in objectives:
        sub = df[df["objective_plot"] == objective].copy().sort_values("plot_x").reset_index(drop=True)
        if len(sub) == 0:
            continue

        label, color, linestyle, linewidth, alpha, zorder = OBJECTIVE_STYLE[objective]

        x = sub["plot_x"].to_numpy()
        acc = maybe_to_percent(sub[acc_col].to_numpy(dtype=float))
        unc = sub[unc_col].to_numpy(dtype=float)

        acc_s = rolling_mean(acc, args.smooth_window)
        unc_s = rolling_mean(unc, args.smooth_window)

        if args.plot_every > 1:
            idx = np.arange(0, len(x), args.plot_every)
            x_plot = x[idx]
            acc_s = acc_s[idx]
            unc_s = unc_s[idx]
        else:
            idx = slice(None)
            x_plot = x

        h = styled_plot(
            axes[0], x_plot, acc_s,
            color=color, linestyle=linestyle,
            linewidth=linewidth, alpha=alpha,
            zorder=zorder, label=label
        )
        styled_plot(
            axes[1], x_plot, unc_s,
            color=color, linestyle=linestyle,
            linewidth=linewidth, alpha=alpha,
            zorder=zorder
        )

        legend_handles.append(h)
        legend_labels.append(label)

        if objective == "adaps_come":
            q = sub[q_col].to_numpy(dtype=float)
            q_s = rolling_mean(q, args.smooth_window)

            pset = sub[pset_col].to_numpy(dtype=float)
            pset = clip_upper(pset, args.pset_clip_quantile)
            pset_s = rolling_mean(pset, args.smooth_window)

            if args.plot_every > 1:
                q_s = q_s[idx]
                pset_s = pset_s[idx]

            styled_plot(
                axes[2], x_plot, q_s,
                color=color, linestyle=linestyle,
                linewidth=1.85, alpha=0.92,
                zorder=8
            )
            styled_plot(
                axes[3], x_plot, pset_s,
                color=color, linestyle=linestyle,
                linewidth=1.85, alpha=0.92,
                zorder=8
            )

    if meta:
        num_severities = meta["num_severities"]
        num_corruptions = meta["num_corruptions"]
        severity_values = meta["severity_values"]
    else:
        num_severities = 5
        num_corruptions = 19
        severity_values = [1, 2, 3, 4, 5]

    severity_segment_batches = args.batches_per_corruption * num_corruptions
    total_batches = severity_segment_batches * num_severities

    add_severity_segment_lines(
        axes,
        total_batches=total_batches,
        severity_segment_batches=severity_segment_batches
    )
    add_severity_labels(
        axes[0],
        total_batches=total_batches,
        severity_segment_batches=severity_segment_batches,
        num_severities=num_severities,
        severity_values=severity_values,
        y=0.985
    )

    tta_label = args.tta.upper()

    axes[0].set_title(f"(a) Online Accuracy ({tta_label})", loc="left", fontweight="bold")
    axes[1].set_title(rf"(b) Batch-level Uncertainty $U_t$ ({tta_label})", loc="left", fontweight="bold")
    axes[2].set_title(rf"(c) Adaptive Threshold $q_t$ ({tta_label} + AdaPS-COME)", loc="left", fontweight="bold")
    axes[3].set_title(rf"(d) Mean Prediction Set Size $\bar{{s}}_t$ ({tta_label} + AdaPS-COME)", loc="left", fontweight="bold")

    axes[0].set_ylabel("Accuracy (%)")
    axes[1].set_ylabel(r"$U_t$")
    axes[2].set_ylabel(r"$q_t$")
    axes[3].set_ylabel(r"$\bar{s}_t$")
    axes[3].set_xlabel("Time (Batches)")

    axes[0].set_ylim(0, 100)
    axes[2].set_ylim(args.q_ylim_low, args.q_ylim_high)

    for ax in axes:
        ax.grid(True, linestyle="--", linewidth=0.55, alpha=0.18)
        ax.margins(x=0.01)

    axes[3].set_xlim(0, total_batches)

    fig.legend(
        legend_handles,
        legend_labels,
        loc="upper center",
        ncol=3,
        frameon=False,
        bbox_to_anchor=(0.5, 0.995),
        handlelength=2.4,
        columnspacing=2.0
    )

    plt.tight_layout(rect=[0, 0, 1, 0.952])

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    plt.savefig(args.out, bbox_inches="tight")
    print(f"Saved figure to: {args.out}")


if __name__ == "__main__":
    main()
