#!/usr/bin/env python3
"""Create a ViT-B/16 source checkpoint from local ImageNet-1K pretrained weights."""
from __future__ import annotations

import argparse
from pathlib import Path

import torch

from common.model_utils import load_local_pretrained_model, save_source_checkpoint, quick_forward_check


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--backbone", type=str, choices=["vit_base_patch16_224"], default="vit_base_patch16_224")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--vit_local_dir", type=str, default="/data/share/models/timm/vit_base_patch16_224")
    parser.add_argument("--vit_model_name", type=str, default="vit_base_patch16_224")
    parser.add_argument("--checkpoint_root", type=str, default="/data/yuehan/COME_EATA_SAR_imagenet-c/Checkpoints")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    device = args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu"
    model, local_weight = load_local_pretrained_model(
        args.backbone,
        device=device,
        vit_local_dir=args.vit_local_dir,
        vit_model_name=args.vit_model_name,
    )
    quick_forward_check(model, device=device)
    out = Path(args.checkpoint_root) / args.backbone / "imagenet-c" / "source_local_imagenet1k.pth"
    save_source_checkpoint(
        model,
        str(out),
        args.backbone,
        meta={"local_weight": local_weight, "created_by": "prepare_source_ckpts.py", "project_root": "/data/yuehan/COME_EATA_SAR_imagenet-c"},
        overwrite=args.overwrite,
    )
    print(f"Saved {args.backbone} source checkpoint: {out}")
    print(f"Loaded local pretrained source: {local_weight}")


if __name__ == "__main__":
    main()
