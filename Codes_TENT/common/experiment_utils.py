"""Shared experiment driver for the five ImageNet-C TTA methods."""
from __future__ import annotations

import argparse
import csv
import json
import os
import random
import time
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
import torch.nn.functional as F

from .come_utils import softmax_entropy, come_opinion, get_sample_loss
from .consistency_utils import js_divergence, light_augment_batch
from .prediction_set_utils import prediction_set_size, weights_from_set_size
from .model_utils import (
    load_local_pretrained_model,
    load_source_checkpoint,
    save_source_checkpoint,
    configure_model_for_tta,
    make_optimizer,
    quick_forward_check,
    set_model_mode_for_prediction,
)
from .data_utils import make_loader, discover_imagenet_c_combos, load_imagenet_class_mapping, write_dataset_check
from .metric_utils import compute_summary

METHOD_CHOICES = ["source_only", "tent", "tent_come", "uc_consistency", "cuc_consistency", "adaps_come"]


def add_common_args(parser: argparse.ArgumentParser):
    parser.add_argument("--method", type=str, default="tent", choices=METHOD_CHOICES)
    parser.add_argument("--data_root", type=str, default="/data/share/datasets")
    parser.add_argument("--dataset", type=str, default="imagenet-c")
    parser.add_argument("--backbone", type=str, default="resnet50", choices=["resnet50", "vit_base_patch16_224"])

    # Local pretrained sources requested by the user.
    parser.add_argument("--resnet_local_ckpt", type=str, default="/data/share/cache/torch/hub/checkpoints/resnet50-19c8e357.pth")
    parser.add_argument("--vit_local_dir", type=str, default="/data/share/models/timm/vit_base_patch16_224")
    parser.add_argument("--vit_model_name", type=str, default="vit_base_patch16_224")

    # Saved source checkpoint created from the local pretrained source. All TTA methods reload this.
    parser.add_argument("--source_ckpt", type=str, default="")
    parser.add_argument("--save_source_ckpt", type=str, default="")
    parser.add_argument("--overwrite_source_ckpt", action="store_true")
    parser.add_argument("--auto_prepare_source_ckpt", action="store_true", default=True)
    parser.add_argument("--no_auto_prepare_source_ckpt", dest="auto_prepare_source_ckpt", action="store_false")

    parser.add_argument("--output_dir", type=str, default="/data/yuehan/COME_optimized_imagenet-c/Results")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--optimizer", type=str, default="adam", choices=["adam", "sgd"])
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--max_batches", type=int, default=-1)
    parser.add_argument("--max_samples", type=int, default=-1)
    parser.add_argument("--corruption", type=str, default="all")
    parser.add_argument("--severity", type=str, default="all")
    parser.add_argument("--shuffle", action="store_true")
    parser.add_argument("--save_batch_metrics", action="store_true")
    parser.add_argument("--save_sample_metrics", action="store_true")
    parser.add_argument("--run_name", type=str, default="")

    # Loss and COME uncertainty.
    parser.add_argument("--base_loss", type=str, default="tent_come", choices=["tent", "come", "tent_come"])
    parser.add_argument("--lambda_mix", type=float, default=0.03)
    parser.add_argument("--come_p_norm", type=float, default=2.0)
    parser.add_argument("--come_tau", type=float, default=1.0)
    parser.add_argument("--come_evidence_mode", type=str, default="exp_relu", choices=["exp_relu", "softplus", "exp_minus_one"])
    parser.add_argument("--uncertainty_eps", type=float, default=1e-12）

    # AdaPS-COME weighting.
    parser.add_argument("--q_min", type=float, default=0.90)
    parser.add_argument("--q_max", type=float, default=0.98)
    parser.add_argument("--weight_mode", type=str, default="inverse_sqrt_set_size", choices=["inverse_sqrt_set_size", "inverse_set_size"])

    parser.add_argument("--write_dataset_check", action="store_true")
    return parser


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def ensure_dir(path):
    Path(path).mkdir(parents=True, exist_ok=True)


def write_csv(path, rows: List[Dict]):
    ensure_dir(Path(path).parent)
    if not rows:
        return
    fieldnames: List[str] = []
    for row in rows:
        for k in row.keys():
            if k not in fieldnames:
                fieldnames.append(k)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def save_json(path, obj):
    ensure_dir(Path(path).parent)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=str)


def make_default_source_ckpt(args) -> str:
    return str(Path("/data/COME_optimized_imagenet-c/Checkpoints") / args.backbone / "imagenet-c" / "source_local_imagenet1k.pth")


def make_run_filename(args, method: str, corruption=None, severity=None) -> str:
    if args.run_name:
        name = args.run_name
    else:
        parts = [args.backbone, method, args.dataset]
        if corruption:
            parts.append(str(corruption))
        if severity not in [None, ""]:
            parts.append(f"sev{severity}")
        parts.append(f"seed{args.seed}")
        name = "_".join(parts)
    return name.replace("/", "_")


def _device(args) -> str:
    if args.device == "cpu":
        return "cpu"
    return "cuda" if torch.cuda.is_available() else "cpu"


def prepare_model_and_optimizer(args, method: str):
    device = _device(args)
    source_ckpt = args.source_ckpt or make_default_source_ckpt(args)

    if method == "source_only":
        model, local_weight = load_local_pretrained_model(
            args.backbone,
            device=device,
            resnet_local_ckpt=args.resnet_local_ckpt,
            vit_local_dir=args.vit_local_dir,
            vit_model_name=args.vit_model_name,
        )
        quick_forward_check(model, device=device)
        model.eval()
        save_path = args.save_source_ckpt or source_ckpt
        save_source_checkpoint(
            model,
            save_path,
            args.backbone,
            meta={
                "dataset": args.dataset,
                "local_weight": local_weight,
                "created_by": "source_only_from_local_imagenet1k_pretrained",
            },
            overwrite=args.overwrite_source_ckpt,
        )
        args.source_ckpt = save_path
        return model, None, device, 0, local_weight

    if not Path(source_ckpt).exists():
        if not args.auto_prepare_source_ckpt:
            raise FileNotFoundError(f"Source checkpoint not found: {source_ckpt}")
        model_src, local_weight = load_local_pretrained_model(
            args.backbone,
            device=device,
            resnet_local_ckpt=args.resnet_local_ckpt,
            vit_local_dir=args.vit_local_dir,
            vit_model_name=args.vit_model_name,
        )
        save_source_checkpoint(
            model_src,
            source_ckpt,
            args.backbone,
            meta={
                "dataset": args.dataset,
                "local_weight": local_weight,
                "created_by": "auto_prepare_before_tta",
            },
            overwrite=False,
        )
        del model_src
        if device != "cpu" and torch.cuda.is_available():
            torch.cuda.empty_cache()

    model = load_source_checkpoint(args.backbone, source_ckpt, device=device, vit_model_name=args.vit_model_name)
    quick_forward_check(model, device=device)
    params, trainable_count = configure_model_for_tta(model, args.backbone)
    optimizer = make_optimizer(params, args.lr, optimizer=args.optimizer, weight_decay=args.weight_decay)
    args.source_ckpt = source_ckpt
    return model, optimizer, device, trainable_count, source_ckpt


def record_predictions(logits, y, paths, args, extra_per_sample=None):
    probs = F.softmax(logits, dim=1)
    conf, pred = probs.max(dim=1)
    top5 = torch.topk(probs, k=5, dim=1).indices
    _, unc, _, strength, _ = come_opinion(
        logits,
        p_norm=args.come_p_norm,
        tau=args.come_tau,
        evidence_mode=args.come_evidence_mode,
        eps=args.uncertainty_eps,
    )
    rows = []
    extra_per_sample = extra_per_sample or {}
    for i in range(logits.shape[0]):
        yi = int(y[i].item()) if torch.is_tensor(y) else int(y[i])
        valid = 0 <= yi < probs.shape[1]
        true_prob = float(probs[i, yi].detach().cpu().item()) if valid else float("nan")
        top5_correct = int(valid and (top5[i] == yi).any().detach().cpu().item())
        row = {
            "label": yi,
            "pred": int(pred[i].detach().cpu().item()),
            "top1_correct": int(valid and int(pred[i].detach().cpu().item()) == yi),
            "top5_correct": top5_correct,
            "true_prob": true_prob,
            "confidence": float(conf[i].detach().cpu().item()),
            "uncertainty": float(unc[i].detach().cpu().item()),
            "dirichlet_strength": float(strength[i].detach().cpu().item()),
            "path": paths[i] if isinstance(paths, (list, tuple)) else "",
        }
        for k, v in extra_per_sample.items():
            if torch.is_tensor(v):
                val = v if v.ndim == 0 else v[i]
                if val.dtype in [torch.int8, torch.int16, torch.int32, torch.int64, torch.long, torch.bool]:
                    row[k] = int(val.detach().cpu().item())
                else:
                    row[k] = float(val.detach().cpu().item())
            elif isinstance(v, list):
                row[k] = v[i]
            else:
                row[k] = v
        rows.append(row)
    return rows


def _sample_loss(logits, args):
    return get_sample_loss(
        logits,
        base_loss=args.base_loss,
        lambda_mix=args.lambda_mix,
        p_norm=args.come_p_norm,
        tau=args.come_tau,
        evidence_mode=args.come_evidence_mode,
        eps=args.uncertainty_eps,
    )


def compute_loss_for_batch(model, x, args, method: str):
    logits = model(x)

    if method == "tent":
        return softmax_entropy(logits, eps=args.uncertainty_eps).mean(), {}, logits

    if method == "tent_come":
        return _sample_loss(logits, args).mean(), {}, logits

    if method == "adaps_come":
        probs = F.softmax(logits, dim=1)
        _, u, _, _, _ = come_opinion(
            logits,
            p_norm=args.come_p_norm,
            tau=args.come_tau,
            evidence_mode=args.come_evidence_mode,
            eps=args.uncertainty_eps,
        )
        batch_unc = u.mean().detach()
        q_t = args.q_min + (args.q_max - args.q_min) * batch_unc
        q_t = torch.clamp(q_t, min=args.q_min, max=args.q_max)
        set_size = prediction_set_size(probs.detach(), q_t.detach())
        weights = weights_from_set_size(set_size, mode=args.weight_mode).detach()
        per_sample_loss = _sample_loss(logits, args)
        loss = (weights * per_sample_loss).sum() / weights.sum().clamp_min(args.uncertainty_eps)
        extra = {
            "batch_uncertainty": batch_unc.expand(logits.shape[0]),
            "adaptive_q": q_t.detach().expand(logits.shape[0]),
            "prediction_set_size": set_size,
            "sample_weight": weights,
        }
        return loss, extra, logits

    if method in ["uc_consistency", "cuc_consistency"]:
        probs = F.softmax(logits, dim=1)
        confidence, _ = probs.max(dim=1)
        _, u, _, _, _ = come_opinion(
            logits,
            p_norm=args.come_p_norm,
            tau=args.come_tau,
            evidence_mode=args.come_evidence_mode,
            eps=args.uncertainty_eps,
        )

        x_aug = light_augment_batch(x, hflip_prob=args.hflip_prob, max_shift=args.max_shift)
        logits_aug = model(x_aug)
        probs_aug = F.softmax(logits_aug, dim=1)
        d = js_divergence(probs.detach(), probs_aug, eps=args.uncertainty_eps)
        stability = torch.exp(-d)
        uncertainty_reliability = (1.0 - u).clamp_min(0.0)
        if method == "uc_consistency":
            reliability = uncertainty_reliability * stability
            confidence_factor = torch.ones_like(confidence)
        else:
            confidence_factor = confidence.detach()
            reliability = confidence_factor * uncertainty_reliability * stability
        reliability = reliability.detach()

        lo = torch.quantile(reliability, args.low_quantile)
        hi = torch.quantile(reliability, args.high_quantile)
        high = reliability >= hi
        low = reliability <= lo
        middle = (~high) & (~low)

        per_sample_loss = _sample_loss(logits, args)
        loss_terms = []
        if high.any():
            loss_terms.append(per_sample_loss[high].mean())
        if middle.any():
            loss_terms.append(args.mu_consistency * d[middle].mean())
        loss = sum(loss_terms) if loss_terms else logits.sum() * 0.0

        group = ["high" if bool(high[i]) else ("low" if bool(low[i]) else "middle") for i in range(logits.shape[0])]
        extra = {
            "js_divergence": d.detach(),
            "stability": stability.detach(),
            "confidence_factor": confidence_factor.detach(),
            "uncertainty_reliability": uncertainty_reliability.detach(),
            "reliability": reliability.detach(),
            "group": group,
        }
        return loss, extra, logits

    raise ValueError(f"Unknown method={method}")


def _append_extra_to_rows(pred_rows, extra: Dict):
    for k, v in extra.items():
        if torch.is_tensor(v):
            v_cpu = v.detach().cpu()
            for i, row in enumerate(pred_rows):
                val = v_cpu if v_cpu.ndim == 0 else v_cpu[i]
                if val.dtype in [torch.int8, torch.int16, torch.int32, torch.int64, torch.long, torch.bool]:
                    row[k] = int(val.item())
                else:
                    row[k] = float(val.item())
        elif isinstance(v, list):
            for i, row in enumerate(pred_rows):
                row[k] = v[i]
        else:
            for row in pred_rows:
                row[k] = v


def run_one_combo(args, method: str, corruption=None, severity=None):
    set_seed(args.seed)
    if args.write_dataset_check:
        write_dataset_check(
            args.data_root,
            str(Path(args.output_dir) / "dataset_check.csv"),
            str(Path(args.output_dir) / "dataset_check.json"),
        )

    mapping = load_imagenet_class_mapping(args.data_root)
    model, optimizer, device, trainable_count, local_source = prepare_model_and_optimizer(args, method)
    loader = make_loader(
        args.data_root,
        args.dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        corruption=corruption,
        severity=severity,
        imagenet_mapping=mapping,
        shuffle=args.shuffle,
        max_samples=args.max_samples if args.max_samples > 0 else None,
    )

    scaler = torch.cuda.amp.GradScaler(enabled=bool(args.amp and device != "cpu"))
    all_rows = []
    batch_rows = []
    start = time.time()
    optimizer_steps = 0
    peak_mem = 0.0

    for batch_idx, batch in enumerate(loader):
        if args.max_batches > 0 and batch_idx >= args.max_batches:
            break
        x, y, _is_ood, paths = batch
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        # Online TTA protocol: evaluate current model on this batch, then update.
        set_model_mode_for_prediction(model, args.backbone, method)
        with torch.no_grad():
            with torch.cuda.amp.autocast(enabled=bool(args.amp and device != "cpu")):
                logits_pre = model(x)
        pred_rows = record_predictions(logits_pre, y, paths, args)

        if method != "source_only":
            optimizer.zero_grad(set_to_none=True)
            set_model_mode_for_prediction(model, args.backbone, method)
            with torch.cuda.amp.autocast(enabled=bool(args.amp and device != "cpu")):
                loss, extra, _ = compute_loss_for_batch(model, x, args, method)
            if torch.isfinite(loss):
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                optimizer_steps += 1
            else:
                extra = extra or {}
            _append_extra_to_rows(pred_rows, extra)
            batch_loss = float(loss.detach().cpu().item()) if torch.is_tensor(loss) else float("nan")
        else:
            batch_loss = float("nan")

        for row in pred_rows:
            row["batch_idx"] = batch_idx
        all_rows.extend(pred_rows)

        labels = np.array([r["label"] for r in pred_rows], dtype=int)
        preds = np.array([r["pred"] for r in pred_rows], dtype=int)
        valid = labels >= 0
        batch_acc = float((preds[valid] == labels[valid]).mean() * 100.0) if valid.any() else float("nan")
        batch_rows.append({
            "batch_idx": batch_idx,
            "batch_size": int(x.shape[0]),
            "loss": batch_loss,
            "batch_accuracy": batch_acc,
            "batch_mean_confidence": float(np.nanmean([r["confidence"] for r in pred_rows])),
            "batch_mean_uncertainty": float(np.nanmean([r["uncertainty"] for r in pred_rows])),
            "method": method,
            "dataset": args.dataset,
            "corruption": corruption if corruption is not None else "",
            "severity": severity if severity is not None else "",
        })

        if device != "cpu" and torch.cuda.is_available():
            peak_mem = max(peak_mem, torch.cuda.max_memory_allocated() / (1024 ** 2))

    elapsed = time.time() - start
    method_uses = {
        "uses_confidence_in_update": int(method == "cuc_consistency"),
        "uses_uncertainty_in_update": int(method in ["tent_come", "adaps_come"]),
        "uses_prediction_set_weighting": int(method == "adaps_come"),
        "uses_all_samples_for_loss": int(method in ["tent", "tent_come", "adaps_come"]),
    }
    extra = {
        "batch_size": args.batch_size,
        "learning_rate": args.lr,
        "lambda_mix": args.lambda_mix,
        "mu_consistency": args.mu_consistency,
        "q_min": args.q_min,
        "q_max": args.q_max,
        "base_loss": args.base_loss,
        "come_evidence_mode": args.come_evidence_mode,
        "come_p_norm": args.come_p_norm,
        "come_tau": args.come_tau,
        "trainable_parameters": trainable_count,
        "runtime_seconds": elapsed,
        "peak_gpu_memory_mb": peak_mem,
        "optimizer_steps": optimizer_steps,
        "source_ckpt": args.source_ckpt,
        "local_source": local_source,
        "resnet_local_ckpt": args.resnet_local_ckpt,
        "vit_local_dir": args.vit_local_dir,
        "vit_model_name": args.vit_model_name,
        "mapping_entries_found": len(mapping),
        **method_uses,
    }
    summary = compute_summary(
        all_rows,
        method=method,
        backbone=args.backbone,
        dataset=args.dataset,
        corruption=corruption,
        severity=severity,
        seed=args.seed,
        extra=extra,
    )

    out_base = Path(args.output_dir) / args.backbone / method / args.dataset
    ensure_dir(out_base)
    name = make_run_filename(args, method, corruption=corruption, severity=severity)
    summary_csv = out_base / f"{name}.csv"
    batch_csv = out_base / f"{name}_batch_metrics.csv"
    sample_csv = out_base / f"{name}_sample_metrics.csv"
    config_json = out_base / f"{name}_config.json"
    write_csv(summary_csv, [summary])
    if args.save_batch_metrics:
        write_csv(batch_csv, batch_rows)
    if args.save_sample_metrics:
        write_csv(sample_csv, all_rows)
    save_json(config_json, vars(args))
    (out_base / f"{name}.completed.flag").write_text("completed\n")
    print(
        f"COMPLETED {args.backbone} {method} {args.dataset} "
        f"corruption={corruption} severity={severity} accuracy={summary.get('accuracy')} "
        f"uncertainty={summary.get('mean_uncertainty')}"
    )
    return summary


def run_experiment_from_args(args):
    method = args.method
    if method == "tent":
        args.base_loss = "tent"
    elif method in ["tent_come", "uc_consistency", "cuc_consistency", "adaps_come"] and "--base_loss" not in os.sys.argv:
        args.base_loss = "tent_come"

    if args.dataset.lower() == "imagenet-c" and (
        args.corruption == "all" or args.severity == "all" or "," in str(args.severity) or "," in str(args.corruption)
    ):
        combos = discover_imagenet_c_combos(args.data_root, corruption=args.corruption, severity=args.severity)
        if not combos:
            raise RuntimeError(
                "No ImageNet-C corruption/severity combos found. Check /data/share/datasets/imagenet-c and run 00_check_environment_and_data.sh."
            )
        summaries = []
        for c, s, _ in combos:
            summaries.append(run_one_combo(args, method, corruption=c, severity=s))
        combined_path = Path(args.output_dir) / args.backbone / method / args.dataset / f"combined_{method}_{args.backbone}_seed{args.seed}.csv"
        write_csv(combined_path, summaries)
        return summaries

    sev = None if args.severity in ["all", "", None] else int(str(args.severity).split(",")[0])
    corr = None if args.corruption in ["all", "", None] else str(args.corruption).split(",")[0]
    return [run_one_combo(args, method, corruption=corr, severity=sev)]


def run_cli(description: str = "ImageNet-C five-method TTA runner"):
    parser = argparse.ArgumentParser(description=description)
    add_common_args(parser)
    args = parser.parse_args()
    return run_experiment_from_args(args)
