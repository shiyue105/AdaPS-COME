#!/usr/bin/env python3
"""Precompute the diagonal Fisher matrix used by EATA.

The paper setting uses 2,000 pre-collected in-distribution samples. This script
first tries to use a clean ImageNet-style folder if available under
/data/share/datasets/images_largescale. If it is unavailable, it falls back to the
first ImageNet-C corruption/severity combination so the experiment can still run;
the fallback source is recorded in the checkpoint metadata.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict

import torch
import torch.nn.functional as F

from common.data_utils import (
    discover_imagenet_c_combos,
    load_imagenet_class_mapping,
    make_loader,
    resolve_dataset_path,
)
from common.model_utils import load_source_checkpoint, configure_model_for_tta, quick_forward_check


def choose_fisher_source(data_root: str, requested: str):
    requested = requested.lower()
    if requested != "auto":
        if requested == "imagenet-c":
            combos = discover_imagenet_c_combos(data_root, corruption="all", severity="1,2,3,4,5")
            if not combos:
                raise RuntimeError("No ImageNet-C combo found for Fisher fallback.")
            c, s, _ = combos[0]
            return "imagenet-c", c, s, f"imagenet-c/{c}/{s}"
        return requested, None, None, requested

    clean_root = resolve_dataset_path(data_root, "images_largescale")
    if clean_root.exists():
        return "images_largescale", None, None, str(clean_root)

    combos = discover_imagenet_c_combos(data_root, corruption="all", severity="1,2,3,4,5")
    if not combos:
        raise RuntimeError("No images_largescale clean folder and no ImageNet-C fallback combo found.")
    c, s, _ = combos[0]
    return "imagenet-c", c, s, f"fallback_imagenet-c/{c}/{s}"


def compute_fisher(args):
    device = args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu"
    mapping = load_imagenet_class_mapping(args.data_root)
    model = load_source_checkpoint(
        args.backbone,
        args.source_ckpt,
        device=device,
        vit_model_name=args.vit_model_name,
    )
    quick_forward_check(model, device=device)
    params, trainable_count = configure_model_for_tta(model, args.backbone)
    named = {name: p for name, p in model.named_parameters() if p.requires_grad}
    fisher: Dict[str, torch.Tensor] = {k: torch.zeros_like(v.detach(), device=device) for k, v in named.items()}
    optpar: Dict[str, torch.Tensor] = {k: v.detach().clone().cpu() for k, v in named.items()}

    ds, corr, sev, source_desc = choose_fisher_source(args.data_root, args.fisher_dataset)
    loader = make_loader(
        args.data_root,
        ds,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        corruption=corr,
        severity=sev,
        imagenet_mapping=mapping,
        shuffle=False,
        max_samples=args.num_samples,
    )

    print(f"Fisher source: {source_desc}")
    print(f"Fisher samples requested: {args.num_samples}")
    print(f"Trainable parameters: {trainable_count}")

    model.eval()
    seen = 0
    for batch in loader:
        x, y, _is_ood, _paths = batch
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        model.zero_grad(set_to_none=True)
        logits = model(x)
        valid = (y >= 0) & (y < logits.shape[1])
        if valid.any():
            loss = F.cross_entropy(logits[valid], y[valid])
        else:
            pseudo = logits.detach().argmax(dim=1)
            loss = F.cross_entropy(logits, pseudo)
        loss.backward()
        bs = int(x.shape[0])
        for k, p in named.items():
            if p.grad is not None:
                fisher[k] += p.grad.detach().pow(2) * bs
        seen += bs
        if seen >= args.num_samples:
            break

    if seen <= 0:
        raise RuntimeError("No samples were seen while computing Fisher.")
    fisher_cpu = {k: (v / float(seen)).detach().cpu() for k, v in fisher.items()}

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "fisher": fisher_cpu,
        "optpar": optpar,
        "meta": {
            "method": "EATA",
            "backbone": args.backbone,
            "num_samples_seen": seen,
            "requested_num_samples": args.num_samples,
            "source_description": source_desc,
            "dataset": ds,
            "corruption": corr,
            "severity": sev,
            "trainable_parameters": trainable_count,
            "warning": "If source_description starts with fallback_imagenet-c, this is a runnable fallback rather than clean in-distribution ImageNet Fisher.",
        },
    }
    torch.save(payload, str(out))
    print(f"Saved EATA Fisher checkpoint: {out}")
    print(json.dumps(payload["meta"], indent=2, default=str))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--backbone", type=str, default="vit_base_patch16_224", choices=["vit_base_patch16_224"])
    parser.add_argument("--source_ckpt", type=str, required=True)
    parser.add_argument("--vit_model_name", type=str, default="vit_base_patch16_224")
    parser.add_argument("--data_root", type=str, default="/data/share/datasets")
    parser.add_argument("--fisher_dataset", type=str, default="auto",
                        help="auto, images_largescale, or imagenet-c. auto prefers clean images_largescale if available.")
    parser.add_argument("--num_samples", type=int, default=2000)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--output", type=str, required=True)
    args = parser.parse_args()
    compute_fisher(args)


if __name__ == "__main__":
    main()
