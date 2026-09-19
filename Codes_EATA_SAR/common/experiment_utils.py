"""ViT ImageNet-C experiments for EATA/SAR wrappers with Tent, Tent-COME, and AdaPS-COME objectives.

Protocol implemented here:
  * backbone: ViT-B/16, local ImageNet-1K pretrained weights;
  * datasets: ImageNet-C corruption/severity streams;
  * wrappers: EATA or SAR;
  * objectives inside the wrapper: Tent, Tent-COME, AdaPS-COME;
  * trainable parameters: affine parameters of LayerNorm modules for ViT;
  * EATA/SAR hyperparameters follow the referenced setting:
      E0 = 0.4 * ln(1000), epsilon = 0.05 for EATA redundancy,
      beta = 2000 for EATA Fisher regularization, SGD momentum = 0.9,
      batch size = 64, lr = 0.001 for ViT.

The runner evaluates predictions before updating on each unlabeled batch. Summary
metrics and batch diagnostics are therefore online TTA diagnostics rather than
post-hoc metrics on a fully adapted model.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import time
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import torch
import torch.nn.functional as F

from .come_utils import softmax_entropy, come_opinion, get_sample_loss
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

PROJECT_ROOT_DEFAULT = "/data/yuehan/COME_EATA_SAR_imagenet-c"
TTA_MODEL_CHOICES = ["eata", "sar"]
OBJECTIVE_CHOICES = ["tent", "tent_come", "adaps_come"]
OBJECTIVE_DISPLAY = {
    "tent": "Tent",
    "tent_come": "Tent-COME",
    "adaps_come": "AdaPS-COME",
}
TTA_DISPLAY = {"eata": "EATA", "sar": "SAR"}


def make_method_id(tta_model: str, objective: str) -> str:
    return f"{tta_model}_{objective}"


def make_method_display(tta_model: str, objective: str) -> str:
    return f"{TTA_DISPLAY.get(tta_model, tta_model)} + {OBJECTIVE_DISPLAY.get(objective, objective)}"


def add_common_args(parser: argparse.ArgumentParser):
    # Backward-compatible aliases: --method is treated as the objective method.
    parser.add_argument("--tta_model", type=str, default="eata", choices=TTA_MODEL_CHOICES,
                        help="Outer TTA algorithm: EATA filtering/Fisher or SAR sharpness-aware update.")
    parser.add_argument("--method", type=str, default="tent", choices=OBJECTIVE_CHOICES,
                        help="Objective optimized inside EATA/SAR: tent, tent_come, or adaps_come.")
    parser.add_argument("--objective", type=str, default="",
                        help="Optional alias for --method. If set, overrides --method.")

    parser.add_argument("--data_root", type=str, default="/data/share/datasets")
    parser.add_argument("--dataset", type=str, default="imagenet-c", choices=["imagenet-c"])
    parser.add_argument("--backbone", type=str, default="vit_base_patch16_224", choices=["vit_base_patch16_224"])

    parser.add_argument("--vit_local_dir", type=str, default="/data/share/models/timm/vit_base_patch16_224")
    parser.add_argument("--vit_model_name", type=str, default="vit_base_patch16_224")

    parser.add_argument("--source_ckpt", type=str, default="")
    parser.add_argument("--save_source_ckpt", type=str, default="")
    parser.add_argument("--overwrite_source_ckpt", action="store_true")
    parser.add_argument("--auto_prepare_source_ckpt", action="store_true", default=True)
    parser.add_argument("--no_auto_prepare_source_ckpt", dest="auto_prepare_source_ckpt", action="store_false")

    parser.add_argument("--output_dir", type=str, default=f"{PROJECT_ROOT_DEFAULT}/Results")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--optimizer", type=str, default="sgd", choices=["adam", "sgd"])
    parser.add_argument("--sgd_momentum", type=float, default=0.9)
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
    parser.add_argument("--save_summary_metrics", action="store_true", default=True)
    parser.add_argument("--no_save_summary_metrics", dest="save_summary_metrics", action="store_false")
    parser.add_argument("--run_name", type=str, default="")

    # Tent-COME / AdaPS-COME uncertainty settings.
    parser.add_argument("--base_loss", type=str, default="tent_come", choices=["tent", "come", "tent_come"])
    parser.add_argument("--lambda_mix", type=float, default=0.03)
    parser.add_argument("--come_p_norm", type=float, default=2.0)
    parser.add_argument("--come_tau", type=float, default=1.0)
    parser.add_argument("--come_evidence_mode", type=str, default="exp_relu", choices=["exp_relu", "softplus", "exp_minus_one"])
    parser.add_argument("--uncertainty_eps", type=float, default=1e-12)

    # AdaPS-COME prediction-set weighting and fixed prediction-set diagnostic.
    parser.add_argument("--q_min", type=float, default=0.90)
    parser.add_argument("--q_max", type=float, default=0.98)
    parser.add_argument("--fixed_ps_q", type=float, default=0.95)
    parser.add_argument("--weight_mode", type=str, default="inverse_sqrt_set_size", choices=["inverse_sqrt_set_size", "inverse_set_size"])

    # EATA hyperparameters from the referenced setting.
    parser.add_argument("--eata_entropy_margin", type=float, default=-1.0,
                        help="Reliable-sample entropy threshold. If negative, use 0.4*ln(1000).")
    parser.add_argument("--eata_redundant_eps", type=float, default=0.05,
                        help="EATA redundant-sample cosine-similarity threshold epsilon.")
    parser.add_argument("--eata_beta", type=float, default=2000.0,
                        help="Trade-off coefficient for the Fisher regularization term.")
    parser.add_argument("--eata_fisher_path", type=str, default="",
                        help="Path to precomputed EATA Fisher checkpoint.")
    parser.add_argument("--eata_model_prob_momentum", type=float, default=0.9)

    # SAR hyperparameters from the referenced setting.
    parser.add_argument("--sar_entropy_margin", type=float, default=-1.0,
                        help="SAR reliable-sample entropy threshold. If negative, use 0.4*ln(1000).")
    parser.add_argument("--sar_rho", type=float, default=0.05)
    parser.add_argument("--sar_ema_alpha", type=float, default=0.9)
    parser.add_argument("--sar_reset_threshold", type=float, default=-1.0,
                        help="SAR model-recovery threshold. If negative, use 0.4*ln(1000).")

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
    return str(Path(PROJECT_ROOT_DEFAULT) / "Checkpoints" / args.backbone / "imagenet-c" / "source_local_imagenet1k.pth")


def make_default_eata_fisher_path(args) -> str:
    return str(Path(PROJECT_ROOT_DEFAULT) / "Checkpoints" / args.backbone / "imagenet-c" / "eata_fisher_2000.pth")


def make_run_filename(args, method_id: str, corruption=None, severity=None) -> str:
    if args.run_name:
        name = args.run_name
    else:
        parts = [args.backbone, method_id, args.dataset]
        if corruption not in [None, ""]:
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


def _entropy_margin(value: float) -> float:
    return 0.4 * math.log(1000.0) if value is None or value < 0 else float(value)


def prepare_model_and_optimizer(args):
    device = _device(args)
    source_ckpt = args.source_ckpt or make_default_source_ckpt(args)

    if not Path(source_ckpt).exists():
        if not args.auto_prepare_source_ckpt:
            raise FileNotFoundError(f"Source checkpoint not found: {source_ckpt}")
        model_src, local_weight = load_local_pretrained_model(
            args.backbone,
            device=device,
            vit_local_dir=args.vit_local_dir,
            vit_model_name=args.vit_model_name,
        )
        quick_forward_check(model_src, device=device)
        save_source_checkpoint(
            model_src,
            source_ckpt,
            args.backbone,
            meta={"dataset": args.dataset, "local_weight": local_weight, "created_by": "auto_prepare_before_tta"},
            overwrite=False,
        )
        del model_src
        if device != "cpu" and torch.cuda.is_available():
            torch.cuda.empty_cache()

    model = load_source_checkpoint(args.backbone, source_ckpt, device=device, vit_model_name=args.vit_model_name)
    quick_forward_check(model, device=device)
    params, trainable_count = configure_model_for_tta(model, args.backbone)
    optimizer = make_optimizer(params, args.lr, optimizer=args.optimizer, weight_decay=args.weight_decay, momentum=args.sgd_momentum)
    args.source_ckpt = source_ckpt
    return model, optimizer, device, trainable_count, source_ckpt


def _trainable_named_params(model):
    return {name: p for name, p in model.named_parameters() if p.requires_grad}


def _zero_like_trainable(named_params: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    return {k: torch.zeros_like(v.detach()) for k, v in named_params.items()}


def prepare_tta_state(model, args, device: str) -> Dict:
    named = _trainable_named_params(model)
    state: Dict[str, object] = {}
    if args.tta_model == "eata":
        fisher_path = args.eata_fisher_path or make_default_eata_fisher_path(args)
        optpar = {k: v.detach().clone() for k, v in named.items()}
        fisher = _zero_like_trainable(named)
        fisher_source = "zero_fisher_fallback"
        if fisher_path and Path(fisher_path).exists():
            obj = torch.load(fisher_path, map_location=device)
            f_obj = obj.get("fisher", obj) if isinstance(obj, dict) else obj
            o_obj = obj.get("optpar", {}) if isinstance(obj, dict) else {}
            loaded = 0
            for k in named.keys():
                if isinstance(f_obj, dict) and k in f_obj:
                    fisher[k] = f_obj[k].detach().to(device=device, dtype=named[k].dtype)
                    loaded += 1
                if isinstance(o_obj, dict) and k in o_obj:
                    optpar[k] = o_obj[k].detach().to(device=device, dtype=named[k].dtype)
            fisher_source = str(fisher_path) if loaded > 0 else "fisher_file_without_matching_trainable_keys"
        else:
            print(f"WARNING: EATA Fisher file not found: {fisher_path}. Using zero Fisher regularizer.")
        state.update({
            "eata_current_model_probs": None,
            "eata_fisher": fisher,
            "eata_optpar": optpar,
            "eata_fisher_source": fisher_source,
        })
    elif args.tta_model == "sar":
        state.update({
            "sar_source_params": {k: v.detach().clone() for k, v in named.items()},
            "sar_ema_loss": None,
        })
    return state


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
    ent = softmax_entropy(logits, eps=args.uncertainty_eps)
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
            "entropy": float(ent[i].detach().cpu().item()),
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


def _objective_sample_loss_and_diag(logits, args):
    """Return per-sample objective loss and per-sample diagnostic tensors."""
    objective = args.method
    probs = F.softmax(logits, dim=1)
    ent = softmax_entropy(logits, eps=args.uncertainty_eps)
    diag = {"objective_entropy": ent.detach()}

    if objective == "tent":
        sample_loss = ent
    else:
        sample_loss = get_sample_loss(
            logits,
            base_loss="tent_come",
            lambda_mix=args.lambda_mix,
            p_norm=args.come_p_norm,
            tau=args.come_tau,
            evidence_mode=args.come_evidence_mode,
            eps=args.uncertainty_eps,
        )

    if objective == "adaps_come":
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
        diag.update({
            "batch_uncertainty": batch_unc.expand(logits.shape[0]),
            "adaptive_q": q_t.detach().expand(logits.shape[0]),
            "prediction_set_size": set_size.detach(),
            "sample_weight": weights.detach(),
        })
    else:
        weights = torch.ones_like(sample_loss).detach()

    diag["objective_sample_loss"] = sample_loss.detach()
    return sample_loss, weights, diag


def _weighted_mean_loss(loss_vec: torch.Tensor, weights: torch.Tensor, mask: Optional[torch.Tensor], eps: float) -> torch.Tensor:
    if mask is None:
        mask_f = torch.ones_like(loss_vec)
    else:
        mask_f = mask.to(dtype=loss_vec.dtype)
    w = weights.to(device=loss_vec.device, dtype=loss_vec.dtype) * mask_f
    denom = w.sum()
    if float(denom.detach().cpu().item()) <= 0.0:
        return loss_vec.sum() * 0.0
    return (loss_vec * w).sum() / denom.clamp_min(eps)


def _fisher_penalty(model, state: Dict, beta: float) -> torch.Tensor:
    named = _trainable_named_params(model)
    if not named:
        return torch.tensor(0.0, device=next(model.parameters()).device)
    penalty = None
    fisher = state.get("eata_fisher", {})
    optpar = state.get("eata_optpar", {})
    for k, p in named.items():
        f = fisher.get(k, None) if isinstance(fisher, dict) else None
        o = optpar.get(k, None) if isinstance(optpar, dict) else None
        if f is None or o is None:
            continue
        term = (f.to(device=p.device, dtype=p.dtype) * (p - o.to(device=p.device, dtype=p.dtype)).pow(2)).sum()
        penalty = term if penalty is None else penalty + term
    if penalty is None:
        penalty = torch.tensor(0.0, device=next(model.parameters()).device)
    return 0.5 * float(beta) * penalty


def eata_update_loss(model, logits, args, state: Dict):
    probs = F.softmax(logits, dim=1)
    ent = softmax_entropy(logits, eps=args.uncertainty_eps)
    e_margin = _entropy_margin(args.eata_entropy_margin)
    reliable = ent < e_margin
    cosine = torch.full_like(ent, float("nan"))
    current_probs = state.get("eata_current_model_probs", None)
    if current_probs is not None:
        cosine = F.cosine_similarity(current_probs.to(probs.device).unsqueeze(0), probs.detach(), dim=1)
        non_redundant = cosine < float(args.eata_redundant_eps)
        selected = reliable & non_redundant
    else:
        selected = reliable

    sample_loss, objective_weights, diag = _objective_sample_loss_and_diag(logits, args)

    coeff_all = torch.zeros_like(ent)
    if selected.any():
        coeff = torch.exp(torch.tensor(e_margin, device=ent.device, dtype=ent.dtype) - ent[selected].detach())
        coeff_all[selected] = coeff
        combined_weights = objective_weights.detach() * coeff_all.detach()
        objective_loss = _weighted_mean_loss(sample_loss, combined_weights, selected, args.uncertainty_eps)
        selected_probs = probs[selected].detach()
        mean_probs = selected_probs.mean(dim=0)
        if current_probs is None:
            state["eata_current_model_probs"] = mean_probs
        else:
            m = float(args.eata_model_prob_momentum)
            state["eata_current_model_probs"] = (m * current_probs.to(mean_probs.device) + (1.0 - m) * mean_probs).detach()
    else:
        objective_loss = logits.sum() * 0.0

    fisher_loss = _fisher_penalty(model, state, args.eata_beta)
    loss = objective_loss + fisher_loss
    diag.update({
        "eata_reliable": reliable.detach().long(),
        "eata_selected": selected.detach().long(),
        "eata_cosine_to_running_probs": cosine.detach(),
        "eata_entropy_coeff": coeff_all.detach(),
        "eata_entropy_margin": torch.full_like(ent, e_margin).detach(),
        "eata_redundant_eps": torch.full_like(ent, float(args.eata_redundant_eps)).detach(),
        "eata_fisher_penalty": torch.full_like(ent, float(fisher_loss.detach().cpu().item())).detach(),
        "eata_objective_loss": torch.full_like(ent, float(objective_loss.detach().cpu().item())).detach(),
    })
    return loss, diag


def _grad_norm(parameters, eps: float = 1e-12) -> torch.Tensor:
    shared_device = None
    norms = []
    for p in parameters:
        if p.grad is None:
            continue
        shared_device = p.device
        norms.append(torch.norm(p.grad.detach(), p=2))
    if not norms:
        return torch.tensor(0.0, device=shared_device or "cpu")
    return torch.norm(torch.stack(norms), p=2).clamp_min(eps)


def _restore_trainable_params(model, source_params: Dict[str, torch.Tensor]):
    named = _trainable_named_params(model)
    with torch.no_grad():
        for k, p in named.items():
            if k in source_params:
                p.copy_(source_params[k].to(device=p.device, dtype=p.dtype))


def sar_update(model, optimizer, x, args, state: Dict):
    params = [p for p in model.parameters() if p.requires_grad]
    e_margin = _entropy_margin(args.sar_entropy_margin)
    reset_threshold = _entropy_margin(args.sar_reset_threshold)

    optimizer.zero_grad(set_to_none=True)
    logits = model(x)
    ent = softmax_entropy(logits, eps=args.uncertainty_eps)
    reliable = ent < e_margin
    sample_loss, objective_weights, diag1 = _objective_sample_loss_and_diag(logits, args)
    if not reliable.any():
        extra = dict(diag1)
        extra.update({
            "sar_reliable": reliable.detach().long(),
            "sar_entropy_margin": torch.full_like(ent, e_margin).detach(),
            "sar_second_entropy": torch.full_like(ent, float("nan")).detach(),
            "sar_second_reliable": torch.zeros_like(ent).detach().long(),
            "sar_recovery_triggered": torch.zeros_like(ent).detach(),
        })
        return logits.sum() * 0.0, extra, 0

    loss1 = _weighted_mean_loss(sample_loss, objective_weights, reliable, args.uncertainty_eps)
    loss1.backward()
    gnorm = _grad_norm(params, eps=args.uncertainty_eps)
    if not torch.isfinite(gnorm) or float(gnorm.detach().cpu().item()) <= 0.0:
        optimizer.zero_grad(set_to_none=True)
        extra = dict(diag1)
        extra.update({
            "sar_reliable": reliable.detach().long(),
            "sar_entropy_margin": torch.full_like(ent, e_margin).detach(),
            "sar_grad_norm": torch.full_like(ent, float(gnorm.detach().cpu().item()) if torch.isfinite(gnorm) else float("nan")).detach(),
            "sar_second_entropy": torch.full_like(ent, float("nan")).detach(),
            "sar_second_reliable": torch.zeros_like(ent).detach().long(),
            "sar_recovery_triggered": torch.zeros_like(ent).detach(),
        })
        return logits.sum() * 0.0, extra, 0

    e_ws = []
    with torch.no_grad():
        scale = float(args.sar_rho) / gnorm
        for p in params:
            if p.grad is None:
                e_ws.append(None)
                continue
            e_w = p.grad.detach() * scale.to(device=p.device, dtype=p.dtype)
            p.add_(e_w)
            e_ws.append(e_w)

    optimizer.zero_grad(set_to_none=True)
    logits2 = model(x)
    ent2 = softmax_entropy(logits2, eps=args.uncertainty_eps)
    reliable2 = ent2 < e_margin
    sample_loss2, objective_weights2, diag2 = _objective_sample_loss_and_diag(logits2, args)
    loss2 = _weighted_mean_loss(sample_loss2, objective_weights2, reliable2, args.uncertainty_eps)
    loss2.backward()

    with torch.no_grad():
        for p, e_w in zip(params, e_ws):
            if e_w is not None:
                p.sub_(e_w)

    loss2_scalar = float(loss2.detach().cpu().item()) if torch.is_tensor(loss2) else float("nan")
    ema = state.get("sar_ema_loss", None)
    if ema is None or not np.isfinite(float(ema)):
        ema = loss2_scalar
    else:
        ema = float(args.sar_ema_alpha) * float(ema) + (1.0 - float(args.sar_ema_alpha)) * loss2_scalar
    state["sar_ema_loss"] = ema

    recovery = int(np.isfinite(ema) and ema > reset_threshold)
    if recovery:
        optimizer.zero_grad(set_to_none=True)
        _restore_trainable_params(model, state.get("sar_source_params", {}))
        state["sar_ema_loss"] = None
        did_step = 0
    else:
        optimizer.step()
        did_step = 1
    optimizer.zero_grad(set_to_none=True)

    extra = dict(diag1)
    # Second-step diagnostics are named explicitly to avoid overwriting first-step fields.
    if args.method == "adaps_come":
        for k, v in diag2.items():
            extra[f"sar_second_{k}"] = v.detach() if torch.is_tensor(v) else v
    extra.update({
        "sar_reliable": reliable.detach().long(),
        "sar_second_reliable": reliable2.detach().long(),
        "sar_entropy_margin": torch.full_like(ent, e_margin).detach(),
        "sar_second_entropy": ent2.detach(),
        "sar_grad_norm": torch.full_like(ent, float(gnorm.detach().cpu().item())).detach(),
        "sar_ema_loss": torch.full_like(ent, float(ema) if np.isfinite(ema) else float("nan")).detach(),
        "sar_recovery_triggered": torch.full_like(ent, float(recovery)).detach(),
    })
    return loss2.detach(), extra, did_step


def _append_extra_to_rows(pred_rows, extra: Dict):
    for k, v in (extra or {}).items():
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


def _safe_mean_from_rows(rows, key: str) -> float:
    vals = [r.get(key, np.nan) for r in rows]
    vals = np.asarray(vals, dtype=float)
    return float(np.nanmean(vals)) if vals.size else float("nan")


def _make_batch_diagnostic_row(
    *,
    args,
    method_id: str,
    dataset: str,
    corruption,
    severity,
    batch_idx: int,
    x_shape0: int,
    batch_loss: float,
    logits_pre: torch.Tensor,
    pred_rows: List[Dict],
):
    probs_pre = F.softmax(logits_pre.detach(), dim=1)
    fixed_q = float(args.fixed_ps_q)
    fixed_ps_size = prediction_set_size(probs_pre, fixed_q).detach().float().cpu().numpy()
    u_vals = np.asarray([r.get("uncertainty", np.nan) for r in pred_rows], dtype=float)
    batch_unc = float(np.nanmean(u_vals)) if u_vals.size else float("nan")

    adaptive_q = float("nan")
    adaps_ps_size = float("nan")
    method_ps_size = float(np.nanmean(fixed_ps_size))
    if args.method == "adaps_come" and np.isfinite(batch_unc):
        adaptive_q = float(np.clip(args.q_min + (args.q_max - args.q_min) * batch_unc, args.q_min, args.q_max))
        adaps_ps = prediction_set_size(probs_pre, adaptive_q).detach().float().cpu().numpy()
        adaps_ps_size = float(np.nanmean(adaps_ps))
        method_ps_size = adaps_ps_size

    labels = np.array([r["label"] for r in pred_rows], dtype=int)
    preds = np.array([r["pred"] for r in pred_rows], dtype=int)
    top5 = np.array([r.get("top5_correct", 0) for r in pred_rows], dtype=int)
    valid = labels >= 0
    correct_count = int(((preds == labels) & valid).sum()) if valid.any() else 0
    batch_acc = float(correct_count / max(int(valid.sum()), 1) * 100.0) if valid.any() else float("nan")
    batch_top5 = float(top5[valid].sum() / max(int(valid.sum()), 1) * 100.0) if valid.any() else float("nan")

    return {
        "batch_idx": batch_idx,
        "combo_local_batch_idx": batch_idx,
        "batch_size": int(x_shape0),
        "num_labeled_samples": int(valid.sum()),
        "num_correct": correct_count,
        "loss": batch_loss,
        "batch_accuracy": batch_acc,
        "batch_top5_accuracy": batch_top5,
        "batch_mean_confidence": _safe_mean_from_rows(pred_rows, "confidence"),
        "batch_mean_entropy": _safe_mean_from_rows(pred_rows, "entropy"),
        "batch_mean_uncertainty": batch_unc,
        "batch_mean_dirichlet_strength": _safe_mean_from_rows(pred_rows, "dirichlet_strength"),
        "batch_fixed_q": fixed_q,
        "batch_mean_prediction_set_size_q095": float(np.nanmean(fixed_ps_size)),
        "batch_mean_fixed_prediction_set_size": float(np.nanmean(fixed_ps_size)),
        "batch_adaptive_q": adaptive_q,
        "batch_mean_adaps_prediction_set_size": adaps_ps_size,
        "batch_mean_method_prediction_set_size": method_ps_size,
        "batch_mean_sample_weight": _safe_mean_from_rows(pred_rows, "sample_weight"),
        "eata_selected_ratio": _safe_mean_from_rows(pred_rows, "eata_selected"),
        "eata_reliable_ratio": _safe_mean_from_rows(pred_rows, "eata_reliable"),
        "eata_mean_cosine_to_running_probs": _safe_mean_from_rows(pred_rows, "eata_cosine_to_running_probs"),
        "eata_mean_entropy_coeff": _safe_mean_from_rows(pred_rows, "eata_entropy_coeff"),
        "eata_fisher_penalty": _safe_mean_from_rows(pred_rows, "eata_fisher_penalty"),
        "sar_reliable_ratio": _safe_mean_from_rows(pred_rows, "sar_reliable"),
        "sar_second_reliable_ratio": _safe_mean_from_rows(pred_rows, "sar_second_reliable"),
        "sar_grad_norm": _safe_mean_from_rows(pred_rows, "sar_grad_norm"),
        "sar_ema_loss": _safe_mean_from_rows(pred_rows, "sar_ema_loss"),
        "sar_recovery_ratio": _safe_mean_from_rows(pred_rows, "sar_recovery_triggered"),
        "method": method_id,
        "method_display": make_method_display(args.tta_model, args.method),
        "tta_model": args.tta_model,
        "objective_method": args.method,
        "backbone": args.backbone,
        "dataset": dataset,
        "corruption": corruption if corruption is not None else "",
        "severity": severity if severity is not None else "",
        "seed": args.seed,
    }


def run_one_combo(args, corruption=None, severity=None) -> Tuple[Dict, List[Dict]]:
    set_seed(args.seed)
    if args.write_dataset_check:
        write_dataset_check(
            args.data_root,
            str(Path(args.output_dir) / "dataset_check.csv"),
            str(Path(args.output_dir) / "dataset_check.json"),
        )

    mapping = load_imagenet_class_mapping(args.data_root)
    method_id = make_method_id(args.tta_model, args.method)
    model, optimizer, device, trainable_count, local_source = prepare_model_and_optimizer(args)
    tta_state = prepare_tta_state(model, args, device)
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

    scaler = torch.cuda.amp.GradScaler(enabled=bool(args.amp and device != "cpu" and args.tta_model == "eata"))
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

        # Online TTA protocol: evaluate current model on this batch, then update with the same unlabeled batch.
        set_model_mode_for_prediction(model, args.backbone, method_id)
        with torch.no_grad():
            with torch.cuda.amp.autocast(enabled=bool(args.amp and device != "cpu" and args.tta_model == "eata")):
                logits_pre = model(x)
        pred_rows = record_predictions(logits_pre, y, paths, args)

        set_model_mode_for_prediction(model, args.backbone, method_id)
        if args.tta_model == "sar":
            batch_loss_tensor, extra, did_step = sar_update(model, optimizer, x, args, tta_state)
            optimizer_steps += int(did_step)
            batch_loss = float(batch_loss_tensor.detach().cpu().item()) if torch.is_tensor(batch_loss_tensor) else float("nan")
        else:
            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=bool(args.amp and device != "cpu")):
                logits = model(x)
                loss, extra = eata_update_loss(model, logits, args, tta_state)
            if torch.isfinite(loss):
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                optimizer_steps += 1
            else:
                extra = extra or {}
            batch_loss = float(loss.detach().cpu().item()) if torch.is_tensor(loss) else float("nan")
            optimizer.zero_grad(set_to_none=True)
        _append_extra_to_rows(pred_rows, extra)

        for row in pred_rows:
            row["batch_idx"] = batch_idx
            row["combo_local_batch_idx"] = batch_idx
            row["method"] = method_id
            row["method_display"] = make_method_display(args.tta_model, args.method)
            row["tta_model"] = args.tta_model
            row["objective_method"] = args.method
            row["backbone"] = args.backbone
            row["dataset"] = args.dataset
            row["corruption"] = corruption if corruption is not None else ""
            row["severity"] = severity if severity is not None else ""
        all_rows.extend(pred_rows)

        batch_rows.append(_make_batch_diagnostic_row(
            args=args,
            method_id=method_id,
            dataset=args.dataset,
            corruption=corruption,
            severity=severity,
            batch_idx=batch_idx,
            x_shape0=int(x.shape[0]),
            batch_loss=batch_loss,
            logits_pre=logits_pre,
            pred_rows=pred_rows,
        ))

        if device != "cpu" and torch.cuda.is_available():
            peak_mem = max(peak_mem, torch.cuda.max_memory_allocated() / (1024 ** 2))

    elapsed = time.time() - start
    method_uses = {
        "uses_confidence_in_update": 0,
        "uses_uncertainty_in_update": int(args.method in ["tent_come", "adaps_come"]),
        "uses_eata_filtering": int(args.tta_model == "eata"),
        "uses_eata_fisher_regularization": int(args.tta_model == "eata"),
        "uses_sar_sam_update": int(args.tta_model == "sar"),
        "uses_prediction_set_weighting": int(args.method == "adaps_come"),
        "uses_reliable_sample_filter": 1,
        "uses_all_samples_for_loss": 0,
    }
    extra = {
        "method_display": make_method_display(args.tta_model, args.method),
        "tta_model": args.tta_model,
        "objective_method": args.method,
        "batch_size": args.batch_size,
        "learning_rate": args.lr,
        "optimizer": args.optimizer,
        "sgd_momentum": args.sgd_momentum,
        "lambda_mix": args.lambda_mix,
        "q_min": args.q_min,
        "q_max": args.q_max,
        "fixed_ps_q": args.fixed_ps_q,
        "base_loss": "tent" if args.method == "tent" else "tent_come",
        "come_evidence_mode": args.come_evidence_mode,
        "come_p_norm": args.come_p_norm,
        "come_tau": args.come_tau,
        "eata_entropy_margin": _entropy_margin(args.eata_entropy_margin),
        "eata_redundant_eps": args.eata_redundant_eps,
        "eata_beta": args.eata_beta,
        "eata_fisher_path": args.eata_fisher_path or make_default_eata_fisher_path(args),
        "eata_fisher_source": tta_state.get("eata_fisher_source", ""),
        "sar_entropy_margin": _entropy_margin(args.sar_entropy_margin),
        "sar_rho": args.sar_rho,
        "sar_reset_threshold": _entropy_margin(args.sar_reset_threshold),
        "trainable_parameters": trainable_count,
        "runtime_seconds": elapsed,
        "peak_gpu_memory_mb": peak_mem,
        "optimizer_steps": optimizer_steps,
        "source_ckpt": args.source_ckpt,
        "local_source": local_source,
        "vit_local_dir": args.vit_local_dir,
        "vit_model_name": args.vit_model_name,
        "mapping_entries_found": len(mapping),
        **method_uses,
    }
    summary = compute_summary(
        all_rows,
        method=method_id,
        backbone=args.backbone,
        dataset=args.dataset,
        corruption=corruption,
        severity=severity,
        seed=args.seed,
        extra=extra,
    )

    out_base = Path(args.output_dir) / args.backbone / method_id / args.dataset
    ensure_dir(out_base)
    name = make_run_filename(args, method_id, corruption=corruption, severity=severity)
    summary_csv = out_base / f"{name}.csv"
    batch_csv = out_base / f"{name}_batch_metrics.csv"
    sample_csv = out_base / f"{name}_sample_metrics.csv"
    config_json = out_base / f"{name}_config.json"
    if args.save_summary_metrics:
        write_csv(summary_csv, [summary])
    if args.save_batch_metrics:
        write_csv(batch_csv, batch_rows)
    if args.save_sample_metrics:
        write_csv(sample_csv, all_rows)
    save_json(config_json, vars(args))
    (out_base / f"{name}.completed.flag").write_text("completed\n")
    print(
        f"COMPLETED {args.backbone} {method_id} {args.dataset} "
        f"corruption={corruption} severity={severity} accuracy={summary.get('accuracy')} "
        f"uncertainty={summary.get('mean_uncertainty')} batches={len(batch_rows)}"
    )
    return summary, batch_rows


def run_experiment_from_args(args):
    if args.objective:
        args.method = args.objective
    if args.method == "tent":
        args.base_loss = "tent"
    else:
        args.base_loss = "tent_come"
    method_id = make_method_id(args.tta_model, args.method)

    summaries = []
    all_batch_rows: List[Dict] = []
    if args.dataset.lower() == "imagenet-c" and (
        args.corruption == "all" or args.severity == "all" or "," in str(args.severity) or "," in str(args.corruption)
    ):
        combos = discover_imagenet_c_combos(args.data_root, corruption=args.corruption, severity=args.severity)
        if not combos:
            raise RuntimeError(
                "No ImageNet-C corruption/severity combos found. Check /data/share/datasets/imagenet-c and run 00_check_environment_and_data.sh."
            )
        for combo_index, (c, s, _) in enumerate(combos):
            summary, batch_rows = run_one_combo(args, corruption=c, severity=s)
            summary["combo_index"] = combo_index
            summaries.append(summary)
            for br in batch_rows:
                br["combo_index"] = combo_index
                all_batch_rows.append(br)
        combined_base = Path(args.output_dir) / args.backbone / method_id / args.dataset
        if args.save_summary_metrics:
            combined_path = combined_base / f"combined_{method_id}_{args.backbone}_seed{args.seed}.csv"
            write_csv(combined_path, summaries)
        if args.save_batch_metrics:
            for i, br in enumerate(all_batch_rows):
                br["stream_batch_idx"] = i
            combined_batch_path = combined_base / f"combined_{method_id}_{args.backbone}_seed{args.seed}_batch_metrics.csv"
            write_csv(combined_batch_path, all_batch_rows)
        return summaries

    sev = None if args.severity in ["all", "", None] else int(str(args.severity).split(",")[0])
    corr = None if args.corruption in ["all", "", None] else str(args.corruption).split(",")[0]
    summary, batch_rows = run_one_combo(args, corruption=corr, severity=sev)
    summaries.append(summary)
    if args.save_batch_metrics:
        for i, br in enumerate(batch_rows):
            br["combo_index"] = 0
            br["stream_batch_idx"] = i
        combined_base = Path(args.output_dir) / args.backbone / method_id / args.dataset
        suffix = ""
        if corr is not None:
            suffix += f"_{corr}"
        if sev is not None:
            suffix += f"_sev{sev}"
        write_csv(combined_base / f"combined_{method_id}_{args.backbone}{suffix}_seed{args.seed}_batch_metrics.csv", batch_rows)
    return summaries


def run_cli(description: str = "ViT ImageNet-C EATA/SAR runner with Tent, Tent-COME, AdaPS-COME objectives"):
    parser = argparse.ArgumentParser(description=description)
    add_common_args(parser)
    args = parser.parse_args()
    return run_experiment_from_args(args)
