import hashlib
import json
import os
import re
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent
DEFAULT_DATA = ROOT.parent / "datasets" / "test_dataset_1500"


def now():
    return datetime.now(timezone.utc).isoformat()


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def json_hash(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def atomic_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    temporary.write_text(json.dumps(obj, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    temporary.replace(path)


def safe_file(directory, name):
    if not isinstance(name, str) or not name or "\\" in name:
        raise ValueError(f"Invalid filename: {name!r}")
    relative = Path(name)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"Unsafe filename: {name!r}")
    root = Path(directory).resolve()
    path = (root / relative).resolve()
    if root not in path.parents or not path.is_file():
        raise ValueError(f"Missing or unsafe file: {name!r}")
    return path


def load_samples(dataset):
    dataset = Path(dataset).resolve()
    data = json.loads((dataset / "prompts.json").read_text())["data"]
    if not isinstance(data, dict) or not data:
        raise ValueError("prompts.json must contain a nonempty data mapping")
    samples = []
    seen_images, seen_masks = set(), set()
    for sid, row in sorted(data.items()):
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", sid):
            raise ValueError(f"Unsafe sample ID: {sid!r}")
        prompt = row.get("prompt")
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError(f"Empty prompt for {sid}")
        image = safe_file(dataset / "reference_images", row["image_file"])
        mask = safe_file(dataset / "masks", row["mask_file"])
        if image in seen_images or mask in seen_masks:
            raise ValueError(f"Duplicate image or mask for {sid}")
        seen_images.add(image)
        seen_masks.add(mask)
        samples.append({"id": sid, "prompt": prompt, "image": str(image), "mask": str(mask)})
    return samples


def load_pair(sample):
    with Image.open(sample["image"]) as image:
        rgb = image.convert("RGB")
    with Image.open(sample["mask"]) as image:
        mask = image.convert("L")
    if rgb.width * mask.height != rgb.height * mask.width:
        raise ValueError(f"Image/mask aspect ratios differ: {sample['id']}")
    return rgb, mask


def binary_mask(mask):
    return np.asarray(mask.convert("L")) > 127


@contextmanager
def exclusive_lock(path):
    import fcntl
    with Path(path).open("a") as lock:
        try:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError(f"Another process owns {path}") from exc
        yield


def runtime_versions():
    from importlib.metadata import PackageNotFoundError, version
    names = ["torch", "torchvision", "diffusers", "transformers", "accelerate", "huggingface-hub", "numpy", "Pillow", "scipy", "lpips", "pytorch-fid", "pytorchfwd", "ptwt"]
    result = {}
    for name in names:
        try:
            result[name] = version(name)
        except PackageNotFoundError:
            result[name] = None
    return result
