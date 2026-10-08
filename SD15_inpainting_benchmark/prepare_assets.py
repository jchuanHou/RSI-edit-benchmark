import argparse
import json
import shutil
from pathlib import Path

from common import ROOT, atomic_json, safe_file, sha256

BASE_REPOSITORY = "stable-diffusion-v1-5/stable-diffusion-inpainting"
BASE_REVISION = "8a4288a76071f7280aedbdb3253bdb9e9d5d84bb"
CLIP_REPOSITORY = "openai/clip-vit-large-patch14"


def copy_verified(source, target, expected):
    if sha256(source) != expected:
        raise ValueError(f"Source checksum mismatch: {source}")
    if target.exists():
        if sha256(target) != expected:
            raise ValueError(f"Existing destination differs; refusing overwrite: {target}")
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(target.name + ".copying")
    try:
        shutil.copyfile(source, temporary)
        if sha256(temporary) != expected:
            raise ValueError(f"Copy checksum mismatch: {target}")
        temporary.replace(target)
    finally:
        temporary.unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-project", type=Path, default=ROOT.parent / "PowerPaint_benchmark")
    args = parser.parse_args()
    source_root = args.source_project.resolve()
    sources = json.loads((source_root / "models" / "sources.json").read_text())
    if sources["base"]["repo_id"] != BASE_REPOSITORY or sources["base"]["revision"] != BASE_REVISION:
        raise ValueError("Expected the unmodified SD1.5 Inpainting base, not PowerPaint or Text2Earth weights")
    if sources["clip"]["repo_id"] != CLIP_REPOSITORY:
        raise ValueError("Unexpected evaluation CLIP")
    selected = {}
    for name in ("base", "clip"):
        metadata = dict(sources[name])
        if not metadata.get("files_sha256"):
            raise ValueError(f"No verified file manifest: {name}")
        for relative, expected in metadata["files_sha256"].items():
            source = safe_file(source_root / "models" / name, relative)
            target = ROOT / "models" / name / relative
            copy_verified(source, target, expected)
        metadata["copied_from"] = str(source_root / "models" / name)
        metadata["local_dir"] = str(ROOT / "models" / name)
        selected[name] = metadata
    unet = json.loads((ROOT / "models" / "base" / "unet" / "config.json").read_text())
    text = json.loads((ROOT / "models" / "base" / "text_encoder" / "config.json").read_text())
    index = json.loads((ROOT / "models" / "base" / "model_index.json").read_text())
    if unet["in_channels"] != 9 or unet["cross_attention_dim"] != 768 or text["hidden_size"] != 768:
        raise ValueError("Not the expected nine-channel SD1.5 inpainting architecture")
    if index["_class_name"] != "StableDiffusionInpaintPipeline":
        raise ValueError("Unexpected model pipeline")
    alex = source_root / "cache" / "torch" / "alexnet-owt-7be5be79.pth"
    digest = sha256(alex)
    if not digest.startswith("7be5be79"):
        raise ValueError("Unexpected AlexNet checksum")
    target = ROOT / "cache" / "torch" / alex.name
    copy_verified(alex, target, digest)
    selected["lpips_backbone"] = {"file": str(target), "sha256": digest, "copied_from": str(alex)}
    atomic_json(ROOT / "models" / "sources.json", selected)
    print("Verified independent local copies: SD1.5 Inpainting base, CLIP ViT-L/14 and AlexNet. No task-specific weights copied.")


if __name__ == "__main__":
    main()
