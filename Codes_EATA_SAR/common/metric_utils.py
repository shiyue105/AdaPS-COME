"""Experiment summary metrics for ImageNet-C TTA ablations."""
from __future__ import annotations

from typing import Any, Dict, Iterable

import numpy as np


def _safe_mean(x: Iterable[Any]) -> float:
    x = np.asarray(list(x), dtype=float)
    if x.size == 0:
        return float("nan")
    return float(np.nanmean(x))


def _safe_std(x: Iterable[Any]) -> float:
    x = np.asarray(list(x), dtype=float)
    if x.size == 0:
        return float("nan")
    return float(np.nanstd(x))


def _roc_auc_score(y_true, score):
    y_true = np.asarray(y_true).astype(int)
    score = np.asarray(score).astype(float)
    good = np.isfinite(score)
    y_true, score = y_true[good], score[good]
    if len(np.unique(y_true)) < 2:
        return float("nan")
    try:
        from sklearn.metrics import roc_auc_score

        return float(roc_auc_score(y_true, score))
    except Exception:
        pos = score[y_true == 1]
        neg = score[y_true == 0]
        if len(pos) == 0 or len(neg) == 0:
            return float("nan")
        total = 0.0
        for p in pos:
            total += np.sum(p > neg) + 0.5 * np.sum(p == neg)
        return float(total / (len(pos) * len(neg)))


def _average_precision(y_true, score):
    y_true = np.asarray(y_true).astype(int)
    score = np.asarray(score).astype(float)
    good = np.isfinite(score)
    y_true, score = y_true[good], score[good]
    if len(np.unique(y_true)) < 2:
        return float("nan")
    try:
        from sklearn.metrics import average_precision_score

        return float(average_precision_score(y_true, score))
    except Exception:
        return float("nan")


def _fpr_at_tpr(y_true, score, target_tpr=0.95):
    y_true = np.asarray(y_true).astype(int)
    score = np.asarray(score).astype(float)
    good = np.isfinite(score)
    y_true, score = y_true[good], score[good]
    if len(np.unique(y_true)) < 2:
        return float("nan")
    try:
        from sklearn.metrics import roc_curve

        fpr, tpr, _ = roc_curve(y_true, score)
        idx = np.where(tpr >= target_tpr)[0]
        if len(idx) == 0:
            return float("nan")
        return float(fpr[idx[0]] * 100.0)
    except Exception:
        return float("nan")


def _ece_percent(correct, confidence, n_bins=15):
    correct = np.asarray(correct).astype(float)
    confidence = np.asarray(confidence).astype(float)
    good = np.isfinite(confidence)
    correct, confidence = correct[good], confidence[good]
    if confidence.size == 0:
        return float("nan")
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for lo, hi in zip(bins[:-1], bins[1:]):
        if hi == 1.0:
            mask = (confidence >= lo) & (confidence <= hi)
        else:
            mask = (confidence >= lo) & (confidence < hi)
        if mask.any():
            ece += mask.mean() * abs(correct[mask].mean() - confidence[mask].mean())
    return float(ece * 100.0)


def _group_metrics(out, prefix, mask, valid, labels, preds, conf, unc, rows):
    out[f"{prefix}_ratio"] = float(mask.mean()) if len(mask) else float("nan")
    gv = mask & valid
    if gv.any():
        out[f"{prefix}_accuracy"] = float(((preds == labels) & gv).sum() / gv.sum() * 100.0)
        out[f"{prefix}_error_rate"] = float(((preds != labels) & gv).sum() / gv.sum() * 100.0)
    else:
        out[f"{prefix}_accuracy"] = float("nan")
        out[f"{prefix}_error_rate"] = float("nan")
    out[f"{prefix}_mean_confidence"] = _safe_mean(conf[mask])
    out[f"{prefix}_mean_uncertainty"] = _safe_mean(unc[mask])
    out[f"{prefix}_mean_reliability"] = _safe_mean([r.get("reliability", np.nan) for m, r in zip(mask, rows) if m])
    out[f"{prefix}_mean_js_divergence"] = _safe_mean([r.get("js_divergence", np.nan) for m, r in zip(mask, rows) if m])


def compute_summary(
    rows,
    method: str,
    backbone: str,
    dataset: str,
    corruption=None,
    severity=None,
    seed=None,
    extra: Dict[str, Any] | None = None,
):
    if not rows:
        return {"run_status": "empty", "method": method, "backbone": backbone}

    labels = np.array([r.get("label", -1) for r in rows], dtype=int)
    preds = np.array([r.get("pred", -1) for r in rows], dtype=int)
    top5_correct_arr = np.array([r.get("top5_correct", 0) for r in rows], dtype=int)
    true_prob = np.array([r.get("true_prob", np.nan) for r in rows], dtype=float)
    conf = np.array([r.get("confidence", np.nan) for r in rows], dtype=float)
    unc = np.array([r.get("uncertainty", np.nan) for r in rows], dtype=float)

    valid = labels >= 0
    correct = (preds == labels) & valid
    wrong = (~correct) & valid
    num_valid = int(valid.sum())

    if valid.any():
        nll = float(np.nanmean(-np.log(np.clip(true_prob[valid], 1e-12, 1.0))))
        top1 = float(correct.sum() / max(num_valid, 1) * 100.0)
        top5 = float(top5_correct_arr[valid].sum() / max(num_valid, 1) * 100.0)
    else:
        nll = top1 = top5 = float("nan")

    out = {
        "backbone": backbone,
        "method": method,
        "dataset": dataset,
        "corruption": corruption if corruption is not None else "",
        "severity": severity if severity is not None else "",
        "seed": seed if seed is not None else "",
        "num_samples": int(len(rows)),
        "num_labeled_samples": num_valid,
        "accuracy": top1,
        "top1_accuracy": top1,
        "top5_accuracy": top5,
        "nll": nll,
        "ece_15bins_percent": _ece_percent(correct[valid], conf[valid], n_bins=15) if valid.any() else float("nan"),
        "mean_confidence": _safe_mean(conf),
        "std_confidence": _safe_std(conf),
        "mean_uncertainty": _safe_mean(unc),
        "std_uncertainty": _safe_std(unc),
        "correct_confidence": _safe_mean(conf[correct]),
        "wrong_confidence": _safe_mean(conf[wrong]),
        "confidence_gap_correct_minus_wrong": _safe_mean(conf[correct]) - _safe_mean(conf[wrong]) if correct.any() and wrong.any() else float("nan"),
        "correct_uncertainty": _safe_mean(unc[correct]),
        "wrong_uncertainty": _safe_mean(unc[wrong]),
        "uncertainty_gap_wrong_minus_correct": _safe_mean(unc[wrong]) - _safe_mean(unc[correct]) if wrong.any() and correct.any() else float("nan"),
        "run_status": "completed",
    }

    if valid.any():
        error_y = wrong[valid].astype(int)
        error_score = unc[valid]
        out["error_auroc_using_uncertainty"] = _roc_auc_score(error_y, error_score)
        out["error_auprc_using_uncertainty"] = _average_precision(error_y, error_score)
        out["error_fpr95_using_uncertainty"] = _fpr_at_tpr(error_y, error_score, target_tpr=0.95)
    else:
        out["error_auroc_using_uncertainty"] = float("nan")
        out["error_auprc_using_uncertainty"] = float("nan")
        out["error_fpr95_using_uncertainty"] = float("nan")

    if any("group" in r for r in rows):
        groups = np.array([r.get("group", "") for r in rows], dtype=object)
        for g in ["high", "middle", "low"]:
            _group_metrics(out, f"{g}_group", groups == g, valid, labels, preds, conf, unc, rows)
        out["mean_js_divergence"] = _safe_mean([r.get("js_divergence", np.nan) for r in rows])
        out["mean_stability"] = _safe_mean([r.get("stability", np.nan) for r in rows])
        out["mean_reliability"] = _safe_mean([r.get("reliability", np.nan) for r in rows])

    if any("prediction_set_size" in r for r in rows):
        out["mean_batch_uncertainty"] = _safe_mean([r.get("batch_uncertainty", np.nan) for r in rows])
        out["mean_adaptive_q"] = _safe_mean([r.get("adaptive_q", np.nan) for r in rows])
        out["mean_prediction_set_size"] = _safe_mean([r.get("prediction_set_size", np.nan) for r in rows])
        out["mean_sample_weight"] = _safe_mean([r.get("sample_weight", np.nan) for r in rows])
        out["std_sample_weight"] = _safe_std([r.get("sample_weight", np.nan) for r in rows])

    if extra:
        out.update(extra)
    return out
