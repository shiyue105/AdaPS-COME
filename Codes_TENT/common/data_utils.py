"""Dataset loading for ImageNet-C and related ImageNet-style folder datasets."""
from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Dict, List, Optional

from PIL import Image, ImageFile
import torch
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from torchvision import datasets, transforms

ImageFile.LOAD_TRUNCATED_IMAGES = True

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".JPEG", ".JPG", ".PNG"}

DATASET_PATHS = {
    "imagenet-c": ["imagenet-c", "ImageNet-C", "imagenet_C"],
    "images_largescale": ["images_largescale", "ImageNet", "imagenet", "ILSVRC2012", "ILSVRC2012_img_val"],
}

IMAGENET_LABELED_DATASETS = {"imagenet-c", "images_largescale"}


def resolve_dataset_path(data_root: str, dataset: str) -> Path:
    root = Path(data_root)
    rels = DATASET_PATHS.get(dataset, [dataset])
    candidates = [root / rel for rel in rels]
    for p in candidates:
        if p.exists():
            return p
    return candidates[0]


def _folder_names(root: Path) -> List[str]:
    if not root.exists():
        return []
    return sorted([p.name for p in root.iterdir() if p.is_dir()])


def _numeric_class_mapping(root: Path) -> Dict[str, int]:
    names = _folder_names(root)
    if names and all(n.isdigit() for n in names):
        return {n: int(n) for n in names}
    return {}


def _requires_imagenet_mapping(dataset: str, root: Path) -> bool:
    if dataset.lower() not in IMAGENET_LABELED_DATASETS:
        return False
    names = _folder_names(root)
    if not names:
        return False
    return any(n.startswith("n") and len(n) >= 8 for n in names)


def _sorted_wnid_fallback_mapping(root: Path) -> Dict[str, int]:
    """Fallback for full 1000-class ImageNet-C folders.

    Standard ImageNet-C uses the 1000 ILSVRC2012 WordNet-ID folders. In common
    torchvision/timm class order, these WNIDs are lexicographically sorted. When a
    local imagenet_class_index.json is missing, this fallback keeps ImageNet-C
    accuracy runnable instead of incorrectly using arbitrary folder order.
    """
    names = _folder_names(root)
    wnids = [n for n in names if n.startswith("n") and len(n) >= 8]
    if len(wnids) >= 900 and len(wnids) == len(names):
        return {wnid: idx for idx, wnid in enumerate(sorted(wnids))}
    return {}


def _mapping_for_root(dataset: str, root: Path, imagenet_mapping: Optional[Dict[str, int]]):
    numeric = _numeric_class_mapping(root)
    if numeric:
        return numeric

    imagenet_mapping = imagenet_mapping or {}
    if imagenet_mapping:
        return imagenet_mapping

    fallback = _sorted_wnid_fallback_mapping(root)
    if fallback:
        return fallback

    if _requires_imagenet_mapping(dataset, root):
        raise RuntimeError(
            f"{dataset} uses ImageNet synset folders under {root}, but no synset->index mapping was found. "
            "Put imagenet_class_index.json under /data/share/datasets, or use the full 1000-class ImageNet-C "
            "folder tree so the sorted-WNID fallback can be used."
        )
    return {}


def imagenet_transform():
    return transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ])


def list_image_files(root: Path) -> List[Path]:
    if not root.exists():
        return []
    return [p for p in root.rglob("*") if p.is_file() and p.suffix in IMG_EXTS]


class ImageListDataset(Dataset):
    def __init__(self, root: str, transform=None, label: int = -1, is_ood: int = 0):
        self.root = Path(root)
        self.files = list_image_files(self.root)
        self.transform = transform
        self.label = int(label)
        self.is_ood = int(is_ood)

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int):
        path = self.files[idx]
        img = Image.open(path).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        return img, self.label, self.is_ood, str(path)


class WrappedImageFolder(datasets.ImageFolder):
    def __init__(self, root, transform=None, class_to_imagenet_idx: Optional[Dict[str, int]] = None, is_ood=0):
        super().__init__(root=root, transform=transform)
        self.is_ood = int(is_ood)
        self.mapping_status = "sequential_folder_indices"
        if class_to_imagenet_idx:
            mapped_samples = []
            missing = []
            for path, _target in self.samples:
                cls = Path(path).parent.name
                if cls not in class_to_imagenet_idx:
                    missing.append(cls)
                    continue
                mapped_samples.append((path, int(class_to_imagenet_idx[cls])))
            if missing:
                raise RuntimeError(f"Missing class mapping for {len(set(missing))} class folders, e.g. {sorted(set(missing))[:10]}")
            self.samples = mapped_samples
            self.targets = [t for _, t in self.samples]
            self.mapping_status = "imagenet_synset_or_numeric_mapping"

    def __getitem__(self, index: int):
        img, target = super().__getitem__(index)
        return img, int(target), self.is_ood, self.samples[index][0]


def _load_mapping_json(path: Path) -> Dict[str, int]:
    obj = json.loads(path.read_text())
    mapping = {}
    # Keras/torchvision style: {"0": ["n01440764", "tench"], ...}
    for k, v in obj.items():
        try:
            idx = int(k)
        except Exception:
            continue
        if isinstance(v, list) and v:
            mapping[str(v[0])] = idx
        elif isinstance(v, str):
            mapping[v.split()[0]] = idx
    return mapping


def load_imagenet_class_mapping(data_root: str) -> Dict[str, int]:
    """Load synset->ImageNet-1K index mapping from local files if available."""
    root = Path(data_root)
    candidates = [
        root / "imagenet_class_index.json",
        root / "ILSVRC2012_class_index.json",
        root / "imagenet_class_index.txt",
        root / "synset_words.txt",
        root / "ImageNet" / "meta.bin",
        root / "imagenet" / "meta.bin",
        root / "images_largescale" / "meta.bin",
    ]

    for p in candidates:
        if not p.exists():
            continue
        try:
            if p.suffix == ".json":
                mapping = _load_mapping_json(p)
                if len(mapping) >= 900:
                    return mapping
            elif p.name == "meta.bin":
                # torchvision ImageNet meta.bin stores (wnid_to_classes, val_wnids).
                obj = torch.load(str(p), map_location="cpu")
                if isinstance(obj, tuple) and len(obj) >= 2 and isinstance(obj[1], list):
                    mapping = {str(wnid): idx for idx, wnid in enumerate(obj[1])}
                    if len(mapping) >= 900:
                        return mapping
            else:
                mapping = {}
                for idx, line in enumerate(p.read_text(errors="ignore").splitlines()):
                    parts = line.strip().split()
                    if parts and parts[0].startswith("n"):
                        mapping[parts[0]] = idx
                if len(mapping) >= 900:
                    return mapping
        except Exception:
            continue

    # Some timm versions expose ImageNetInfo. Use it opportunistically.
    try:
        from timm.data.imagenet_info import ImageNetInfo

        info = ImageNetInfo()
        for attr in ["_synset_to_idx", "synset_to_idx", "wnid_to_idx"]:
            maybe = getattr(info, attr, None)
            if isinstance(maybe, dict) and len(maybe) >= 900:
                return {str(k): int(v) for k, v in maybe.items()}
        if hasattr(info, "index_to_synset"):
            mapping = {info.index_to_synset(i): i for i in range(1000)}
            if len(mapping) >= 900:
                return mapping
    except Exception:
        pass

    return {}


def discover_imagenet_c_combos(data_root: str, corruption="all", severity="all"):
    root = resolve_dataset_path(data_root, "imagenet-c")
    if not root.exists():
        return []

    corruptions = sorted([p.name for p in root.iterdir() if p.is_dir()])
    if corruption and corruption != "all":
        wanted = [c.strip() for c in str(corruption).split(",") if c.strip()]
        corruptions = [c for c in corruptions if c in wanted]

    if severity in ["all", None, ""]:
        severities = [1, 2, 3, 4, 5]
    else:
        severities = [int(s) for s in str(severity).split(",") if str(s).strip()]

    combos = []
    for c in corruptions:
        for s in severities:
            p = root / c / str(s)
            if p.exists():
                combos.append((c, s, p))
    return combos


def get_dataset_root_for_combo(data_root: str, dataset: str, corruption=None, severity=None):
    base = resolve_dataset_path(data_root, dataset)
    if dataset.lower() == "imagenet-c":
        if corruption is None or severity is None:
            raise ValueError("ImageNet-C requires corruption and severity")
        return base / str(corruption) / str(severity)
    return base


def make_single_dataset(
    data_root: str,
    dataset: str,
    corruption=None,
    severity=None,
    transform=None,
    imagenet_mapping: Optional[Dict[str, int]] = None,
):
    root = get_dataset_root_for_combo(data_root, dataset, corruption, severity)
    transform = transform or imagenet_transform()
    if not root.exists():
        raise FileNotFoundError(f"Dataset path not found: {root}")

    subdirs = [p for p in root.iterdir() if p.is_dir()]
    has_class_folders = len(subdirs) > 0 and any(list_image_files(p) for p in subdirs[: min(20, len(subdirs))])
    if has_class_folders:
        class_mapping = _mapping_for_root(dataset, root, imagenet_mapping)
        return WrappedImageFolder(str(root), transform=transform, class_to_imagenet_idx=class_mapping, is_ood=0)
    return ImageListDataset(str(root), transform=transform, label=-1, is_ood=0)


def make_loader(
    data_root: str,
    dataset: str,
    batch_size: int = 64,
    num_workers: int = 8,
    corruption=None,
    severity=None,
    imagenet_mapping=None,
    shuffle: bool = False,
    max_samples: Optional[int] = None,
):
    ds = make_single_dataset(
        data_root,
        dataset,
        corruption=corruption,
        severity=severity,
        transform=imagenet_transform(),
        imagenet_mapping=imagenet_mapping,
    )
    if max_samples is not None and max_samples > 0 and max_samples < len(ds):
        ds = torch.utils.data.Subset(ds, list(range(max_samples)))
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
    )


def _infer_mapping_status(path: Path, dataset: str, imagenet_mapping: Optional[Dict[str, int]] = None) -> str:
    if not path.exists():
        return "missing"
    roots_to_check = []
    if dataset.lower() == "imagenet-c":
        for c in sorted([p for p in path.iterdir() if p.is_dir()]):
            sev_dirs = sorted([p for p in c.iterdir() if p.is_dir() and p.name.isdigit()], key=lambda x: int(x.name))
            if sev_dirs:
                roots_to_check.append(sev_dirs[0])
                break
    else:
        roots_to_check.append(path)

    if not roots_to_check:
        return "no_class_folder_found"
    names = _folder_names(roots_to_check[0])
    if not names:
        return "no_class_folder_found"
    if all(n.isdigit() for n in names):
        return "numeric_folder_labels"
    if any(n.startswith("n") and len(n) >= 8 for n in names):
        mapping = imagenet_mapping or {}
        if mapping:
            ok = sum(1 for n in names if n in mapping)
            return "imagenet_synset_mapping" if ok == len(names) else f"partial_synset_mapping_{ok}_of_{len(names)}"
        fallback = _sorted_wnid_fallback_mapping(roots_to_check[0])
        if fallback:
            return f"sorted_wnid_fallback_{len(fallback)}_classes"
        return "missing_synset_mapping"
    return "sequential_folder_indices"


def scan_dataset(data_root: str, dataset: str) -> Dict[str, object]:
    path = resolve_dataset_path(data_root, dataset)
    mapping = load_imagenet_class_mapping(data_root)
    row = {
        "dataset_name": dataset,
        "path": str(path),
        "exists": path.exists(),
        "number_of_images": 0,
        "number_of_classes": 0,
        "directory_structure": "missing",
        "severity_levels": "",
        "corruptions": "",
        "label_mapping_status": "unchecked",
        "corrupt_files_first50": 0,
        "mapping_entries_found": len(mapping),
    }
    if not path.exists():
        return row
    images = list_image_files(path)
    row["number_of_images"] = len(images)
    subdirs = [p for p in path.iterdir() if p.is_dir()]
    row["number_of_classes"] = len(subdirs)
    row["directory_structure"] = "folder_tree"
    if dataset.lower() == "imagenet-c":
        corruptions = sorted([p.name for p in path.iterdir() if p.is_dir()])
        sevs = set()
        for c in corruptions:
            cp = path / c
            for p in cp.iterdir() if cp.exists() else []:
                if p.is_dir() and p.name.isdigit():
                    sevs.add(p.name)
        row["corruptions"] = ",".join(corruptions)
        row["severity_levels"] = ",".join(sorted(sevs, key=lambda x: int(x)))
    row["label_mapping_status"] = _infer_mapping_status(path, dataset, mapping)

    bad = 0
    for p in images[:50]:
        try:
            Image.open(p).verify()
        except Exception:
            bad += 1
    row["corrupt_files_first50"] = bad
    return row


def write_dataset_check(data_root: str, out_csv: str, out_json: str | None = None):
    rows = [scan_dataset(data_root, "imagenet-c"), scan_dataset(data_root, "images_largescale")]
    Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    if out_json:
        Path(out_json).parent.mkdir(parents=True, exist_ok=True)
        with open(out_json, "w") as f:
            json.dump(rows, f, indent=2)
    return rows
