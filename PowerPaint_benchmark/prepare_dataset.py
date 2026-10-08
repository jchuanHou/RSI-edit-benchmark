import argparse
import collections
import json
import shutil
import stat
import zipfile
from pathlib import Path, PurePosixPath

import numpy as np
from PIL import Image

from common import DEFAULT_DATA, ROOT, atomic_json, binary_mask, json_hash, load_pair, load_samples, now, sha256


def extract_checked(archive, target):
    if target.exists():
        return
    with zipfile.ZipFile(archive) as z:
        entries = z.infolist()
        if len(entries) > 20000 or sum(i.file_size for i in entries) > 8 * 1024**3:
            raise ValueError("Archive exceeds dataset limits")
        seen = set()
        for item in entries:
            p = PurePosixPath(item.filename)
            if p.is_absolute() or ".." in p.parts or "\\" in item.filename or stat.S_ISLNK(item.external_attr >> 16):
                raise ValueError(f"Unsafe ZIP member: {item.filename}")
            if item.filename in seen:
                raise ValueError(f"Duplicate ZIP member: {item.filename}")
            seen.add(item.filename)
            if item.flag_bits & 1:
                raise ValueError("Encrypted member is not supported")
        bad = z.testzip()
        if bad:
            raise ValueError(f"ZIP CRC failed: {bad}")
        target.mkdir(parents=True)
        for item in entries:
            destination = target.joinpath(*PurePosixPath(item.filename).parts)
            if item.is_dir():
                destination.mkdir(parents=True, exist_ok=True)
            else:
                destination.parent.mkdir(parents=True, exist_ok=True)
                with z.open(item) as source, destination.open("xb") as output:
                    shutil.copyfileobj(source, output)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive", type=Path, default=ROOT.parent / "datasets" / "test_dataset_1500.zip")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATA)
    args = parser.parse_args()
    extract_checked(args.archive, args.dataset)
    samples = load_samples(args.dataset)
    records, dimensions, nonbinary, empty, full = [], collections.Counter(), [], [], []
    fractions = []
    for sample in samples:
        for field in ("image", "mask"):
            with Image.open(sample[field]) as im:
                im.verify()
        image, mask = load_pair(sample)
        dimensions[f"image={image.size},mask={mask.size}"] += 1
        array = np.asarray(mask)
        binary = binary_mask(mask)
        coverage = float(binary.mean())
        fractions.append(coverage)
        if not np.isin(array, [0, 255]).all():
            nonbinary.append(sample["id"])
        if not binary.any():
            empty.append(sample["id"])
        if binary.all():
            full.append(sample["id"])
        records.append({"id": sample["id"], "image_file": Path(sample["image"]).name, "mask_file": Path(sample["mask"]).name, "prompt": sample["prompt"], "image_sha256": sha256(sample["image"]), "mask_sha256": sha256(sample["mask"]), "size": list(image.size), "mask_fraction": coverage})
    declared = json.loads((args.dataset / "dataset_info.json").read_text())
    declared_names = {r["filename"] for r in declared["images"]}
    actual_names = {r["image_file"] for r in records}
    report = {
        "checked_at": now(), "dataset_root": str(args.dataset.resolve()),
        "archive_sha256": sha256(args.archive), "prompts_sha256": sha256(args.dataset / "prompts.json"),
        "manifest_sha256": json_hash(records), "num_samples": len(records),
        "dimensions": dict(dimensions), "distinct_prompts": len({r["prompt"] for r in records}),
        "mask_fraction_min_mean_max": [min(fractions), float(np.mean(fractions)), max(fractions)],
        "nonbinary_mask_ids": nonbinary, "empty_mask_ids": empty, "full_mask_ids": full,
        "dataset_info_only": sorted(declared_names - actual_names), "actual_only": sorted(actual_names - declared_names),
        "reference_semantics": "Original image, NOT an edited target ground truth.",
        "manifest_authority": "prompts.json plus actual image/mask files; original dataset_info.json is unchanged.",
        "samples": records,
    }
    atomic_json(ROOT / "dataset_manifest.json", report)
    if empty:
        raise ValueError(f"Empty masks cannot be scored with Local CLIP: {empty}")
    print(json.dumps({k: v for k, v in report.items() if k != "samples"}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
