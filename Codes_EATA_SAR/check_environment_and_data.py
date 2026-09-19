#!/usr/bin/env python3
"""Check local ViT checkpoint, GPU, and ImageNet-C folder structures."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from common.data_utils import discover_imagenet_c_combos, write_dataset_check, load_imagenet_class_mapping, resolve_dataset_path
from common.model_utils import find_vit_weight_file


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, default="/data/share/datasets")
    parser.add_argument("--output_dir", type=str, default="/data/yuehan/COME_EATA_SAR_imagenet-c/Results")
    parser.add_argument("--vit_local_dir", type=str, default="/data/share/models/timm/vit_base_patch16_224")
    parser.add_argument("--corruption", type=str, default="all")
    parser.add_argument("--severity", type=str, default="1,2,3,4,5")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = write_dataset_check(
        args.data_root,
        str(out_dir / "dataset_check.csv"),
        str(out_dir / "dataset_check.json"),
    )
    combos = discover_imagenet_c_combos(args.data_root, corruption=args.corruption, severity=args.severity)
    mapping = load_imagenet_class_mapping(args.data_root)

    checks = {
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_device_count_visible": torch.cuda.device_count() if torch.cuda.is_available() else 0,
        "cuda_device_name_0_visible": torch.cuda.get_device_name(0) if torch.cuda.is_available() and torch.cuda.device_count() else "",
        "vit_local_dir": args.vit_local_dir,
        "vit_local_dir_exists": Path(args.vit_local_dir).exists(),
        "vit_weight_file": "",
        "vit_weight_file_exists": False,
        "imagenet_mapping_entries_found": len(mapping),
        "imagenet_c_combos_found": len(combos),
        "dataset_rows": rows,
    }
    try:
        vit_weight = find_vit_weight_file(args.vit_local_dir)
        checks["vit_weight_file"] = str(vit_weight)
        checks["vit_weight_file_exists"] = vit_weight.exists()
    except Exception as e:
        checks["vit_weight_file_error"] = str(e)

    (out_dir / "environment_check.json").write_text(json.dumps(checks, indent=2, default=str))

    print("Environment/data check saved to:", out_dir)
    print("ImageNet-C corruption/severity combinations found:", len(combos))
    print("ImageNet synset mapping entries found:", len(mapping))
    print("ViT dir exists:", checks["vit_local_dir_exists"], args.vit_local_dir)
    print("ViT weight file:", checks.get("vit_weight_file") or checks.get("vit_weight_file_error", "not found"))
    for row in rows:
        print(f"{row['dataset_name']}: exists={row['exists']} images={row['number_of_images']} mapping={row['label_mapping_status']}")
    if not torch.cuda.is_available():
        print("WARNING: CUDA is not available. Runs will fall back to CPU and be very slow.")


if __name__ == "__main__":
    main()
