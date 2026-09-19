#!/usr/bin/env python3
"""Create saved source checkpoints from local ImageNet-1K pretrained weights, without training/evaluation."""
from __future__ import annotations

import argparse
from pathlib import Path

import torch

from common.model_utils import load_local_pretrained_model, save_source_checkpoint, quick_forward_check


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--backbone", type=str, choices=["resnet50", "vit_base_patch16_224", "both"], default="both")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--resnet_local_ckpt", type=str, default="/data/share/cache/torch/hub/checkpoints/resnet50-19c8e357.pth")
    parser.add_argument("--vit_local_dir", type=str, default="/data/share/models/timm/vit_base_patch16_224")
    parser.add_argument("--vit_model_name", type=str, default="vit_base_patch16_224")
    parser.add_argument("--checkpoint_root", type=str, default="/data/yuehan/COME_optimized_imagenet-c/Checkpoints")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    device = args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu"
    backbones = ["resnet50", "vit_base_patch16_224"] if args.backbone == "both" else [args.backbone]

    for backbone in backbones:
        model, local_weight = load_local_pretrained_model(
            backbone,
            device=device,
            resnet_local_ckpt=args.resnet_local_ckpt,
            vit_local_dir=args.vit_local_dir,
            vit_model_name=args.vit_model_name,
        )
        quick_forward_check(model, device=device)
        out = Path(args.checkpoint_root) / backbone / "imagenet-c" / "source_local_imagenet1k.pth"
        save_source_checkpoint(
            model,
            str(out),
            backbone,
            meta={"local_weight": local_weight, "created_by": "prepare_source_ckpts.py"},
            overwrite=args.overwrite,
        )
        print(f"Saved {backbone} source checkpoint: {out}")
        print(f"Loaded local pretrained source: {local_weight}")
        del model
        if device != "cpu" and torch.cuda.is_available():
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
