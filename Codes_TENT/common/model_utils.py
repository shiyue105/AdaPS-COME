"""Model loading and TTA parameter configuration.

This project deliberately loads local ImageNet-1K pretrained checkpoints:
  ViT-B/16: /data/share/models/timm/vit_base_patch16_224/
No clean ImageNet training is performed here.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

import torch
import torch.nn as nn


def _extract_state_dict(obj: Any) -> Any:
    if isinstance(obj, dict):
        for key in ["state_dict", "model", "model_state", "model_state_dict", "net", "module"]:
            if key in obj and isinstance(obj[key], dict):
                return obj[key]
    return obj


def _clean_state_dict_keys(state: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    state = _extract_state_dict(state)
    cleaned = {}
    for k, v in state.items():
        nk = str(k)
        for pref in ["module.", "model.", "net."]:
            if nk.startswith(pref):
                nk = nk[len(pref):]
        cleaned[nk] = v
    return cleaned


def _load_checkpoint_file(path: Path):
    suffix = path.suffix.lower()
    if suffix == ".safetensors":
        try:
            from safetensors.torch import load_file
        except Exception as e:
            raise RuntimeError(
                f"Found safetensors checkpoint {path}, but safetensors is not installed. "
                "Install it with: pip install safetensors"
            ) from e
        return load_file(str(path), device="cpu")
    return torch.load(str(path), map_location="cpu")



def build_vit_base_patch16_224(device: str = "cuda", vit_model_name: str = "vit_base_patch16_224") -> nn.Module:
    try:
        import timm
    except Exception as e:
        raise RuntimeError("timm is required for vit_base_patch16_224. Install/use the tta_env containing timm.") from e
    model = timm.create_model(vit_model_name, pretrained=False, num_classes=1000)
    model.to(device)
    return model


def find_vit_weight_file(vit_local_dir_or_config: str) -> Path:
    root = Path(vit_local_dir_or_config)
    if root.is_file() and root.name == "config.json":
        root = root.parent
    if not root.exists():
        raise FileNotFoundError(f"ViT local directory/config not found: {vit_local_dir_or_config}")

    preferred_names = [
        "model.safetensors",
        "pytorch_model.bin",
        "pytorch_model.safetensors",
        "checkpoint.pth",
        "model.pth",
        "model.pt",
    ]
    for name in preferred_names:
        p = root / name
        if p.exists():
            return p

    candidates = []
    for pattern in ["*.safetensors", "*.bin", "*.pth", "*.pt"]:
        candidates.extend(root.rglob(pattern))
    candidates = [p for p in candidates if p.is_file()]
    if candidates:
        # Prefer the largest file; config/adaptor files are usually smaller.
        return sorted(candidates, key=lambda p: p.stat().st_size, reverse=True)[0]

    raise FileNotFoundError(
        f"No ViT weight file found under {root}. The directory must contain model.safetensors, "
        "pytorch_model.bin, or a .pth/.pt checkpoint in addition to config.json."
    )


def load_local_pretrained_model(
    backbone: str,
    device: str = "cuda",
    resnet_local_ckpt: str = "/data/share/cache/torch/hub/checkpoints/resnet50-19c8e357.pth",
    vit_local_dir: str = "/data/share/models/timm/vit_base_patch16_224",
    vit_model_name: str = "vit_base_patch16_224",
):

    if backbone == "vit_base_patch16_224":
        weight_file = find_vit_weight_file(vit_local_dir)
        model = build_vit_base_patch16_224(device=device, vit_model_name=vit_model_name)
        state = _clean_state_dict_keys(_load_checkpoint_file(weight_file))
        missing, unexpected = model.load_state_dict(state, strict=False)
        serious_missing = [k for k in missing if not k.startswith("head_dist")]
        if serious_missing or unexpected:
            # A second pass sometimes fixes HF/timm wrappers with an extra base prefix.
            alt = {}
            for k, v in state.items():
                nk = k
                for pref in ["vit.", "encoder.", "backbone."]:
                    if nk.startswith(pref):
                        nk = nk[len(pref):]
                alt[nk] = v
            missing, unexpected = model.load_state_dict(alt, strict=False)
            serious_missing = [k for k in missing if not k.startswith("head_dist")]
        if serious_missing or unexpected:
            raise RuntimeError(
                f"ViT checkpoint mismatch for model_name={vit_model_name}, file={weight_file}. "
                f"missing={serious_missing[:30]}, unexpected={unexpected[:30]}"
            )
        return model, str(weight_file)

    raise ValueError(f"Unknown backbone={backbone}")


def build_empty_model(backbone: str, device: str = "cuda", vit_model_name: str = "vit_base_patch16_224"):
    if backbone == "resnet50":
        return build_resnet50(device=device)
    if backbone == "vit_base_patch16_224":
        return build_vit_base_patch16_224(device=device, vit_model_name=vit_model_name)
    raise ValueError(f"Unknown backbone={backbone}")


def save_source_checkpoint(
    model: nn.Module,
    path: str,
    backbone: str,
    meta: Optional[Dict[str, Any]] = None,
    overwrite: bool = False,
) -> str:
    path_obj = Path(path)
    path_obj.parent.mkdir(parents=True, exist_ok=True)
    if path_obj.exists() and not overwrite:
        return str(path_obj)
    payload = {
        "state_dict": model.state_dict(),
        "backbone": backbone,
        "num_classes": 1000,
        "checkpoint_type": "local_imagenet1k_pretrained_source",
        "meta": meta or {},
    }
    torch.save(payload, str(path_obj))
    return str(path_obj)


def load_source_checkpoint(backbone: str, source_ckpt: str, device: str = "cuda", vit_model_name: str = "vit_base_patch16_224"):
    if not source_ckpt:
        raise ValueError("--source_ckpt is required for TTA methods")
    path = Path(source_ckpt)
    if not path.exists():
        raise FileNotFoundError(f"Source checkpoint not found: {path}")
    obj = torch.load(str(path), map_location="cpu")
    ckpt_backbone = obj.get("backbone", backbone) if isinstance(obj, dict) else backbone
    if ckpt_backbone != backbone:
        raise RuntimeError(f"Checkpoint backbone mismatch: checkpoint={ckpt_backbone}, requested={backbone}")
    model = build_empty_model(backbone, device=device, vit_model_name=vit_model_name)
    state = _clean_state_dict_keys(obj)
    missing, unexpected = model.load_state_dict(state, strict=False)
    serious_missing = [k for k in missing if not (backbone == "vit_base_patch16_224" and k.startswith("head_dist"))]
    if serious_missing or unexpected:
        raise RuntimeError(
            f"Source checkpoint mismatch for {backbone}. missing={serious_missing[:30]}, unexpected={unexpected[:30]}"
        )
    model.to(device)
    return model


def set_trainable_params(model: nn.Module, backbone: str) -> int:
    """Freeze the model except normalization affine parameters used for TTA."""
    for p in model.parameters():
        p.requires_grad_(False)

    count = 0
    if backbone == "resnet50":
        for m in model.modules():
            if isinstance(m, nn.modules.batchnorm._BatchNorm):
                if m.weight is not None:
                    m.weight.requires_grad_(True)
                    count += m.weight.numel()
                if m.bias is not None:
                    m.bias.requires_grad_(True)
                    count += m.bias.numel()
                # Tent protocol: use current test batch statistics.
                m.track_running_stats = False
                m.running_mean = None
                m.running_var = None
    elif backbone == "vit_base_patch16_224":
        for m in model.modules():
            if isinstance(m, nn.LayerNorm):
                if m.weight is not None:
                    m.weight.requires_grad_(True)
                    count += m.weight.numel()
                if m.bias is not None:
                    m.bias.requires_grad_(True)
                    count += m.bias.numel()
    else:
        raise ValueError(f"Unknown backbone={backbone}")
    return count


def configure_model_for_tta(model: nn.Module, backbone: str):
    trainable = set_trainable_params(model, backbone)
    if backbone == "resnet50":
        model.train()
    else:
        model.eval()  # ViT has no BN; keep dropout disabled while LayerNorm gets gradients.
    params = [p for p in model.parameters() if p.requires_grad]
    if not params:
        raise RuntimeError(f"No trainable parameters configured for {backbone}")
    return params, trainable


def set_model_mode_for_prediction(model: nn.Module, backbone: str, method: str):
    if method == "source_only":
        model.eval()
    elif backbone == "resnet50":
        model.train()
    else:
        model.eval()


def make_optimizer(params, lr: float, optimizer: str = "adam", weight_decay: float = 0.0):
    opt = optimizer.lower()
    if opt == "adam":
        return torch.optim.Adam(params, lr=lr, weight_decay=weight_decay)
    if opt == "sgd":
        return torch.optim.SGD(params, lr=lr, momentum=0.9, weight_decay=weight_decay)
    raise ValueError(f"Unknown optimizer={optimizer}")


def quick_forward_check(model: nn.Module, device: str = "cuda"):
    was_training = model.training
    model.eval()
    with torch.no_grad():
        x = torch.zeros(1, 3, 224, 224, device=device)
        y = model(x)
    if was_training:
        model.train()
    if tuple(y.shape) != (1, 1000):
        raise RuntimeError(f"Forward output shape must be [B,1000], got {tuple(y.shape)}")
    return tuple(y.shape)
