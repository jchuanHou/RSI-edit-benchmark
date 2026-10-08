import json
import os
import shutil
import time

import requests
from huggingface_hub import snapshot_download

from common import ROOT, atomic_json, exclusive_lock, sha256

SPECS = {
    "base": {
        "repo_id": "stable-diffusion-v1-5/stable-diffusion-inpainting",
        "revision": "8a4288a76071f7280aedbdb3253bdb9e9d5d84bb",
        "allow_patterns": ["model_index.json", "scheduler/*.json", "tokenizer/*", "text_encoder/config.json", "text_encoder/model.fp16.safetensors", "unet/config.json", "unet/diffusion_pytorch_model.fp16.safetensors", "vae/config.json", "vae/diffusion_pytorch_model.fp16.safetensors", "feature_extractor/*.json"],
        "provenance": "Public mirror of runwayml/stable-diffusion-inpainting; 9-channel inpainting base, not ordinary SD1.5.",
    },
    "powerpaint": {
        "repo_id": "JunhaoZhuang/PowerPaint-v1",
        "revision": "b4174b7cade590ab185dbdcfa09eee2c0f63410b",
        "allow_patterns": ["unet/unet.safetensors", "text_encoder/text_encoder.safetensors"],
        "provenance": "Official original PowerPaint v1; not the BrushNet-based v2.",
    },
    "tokenizer": {
        "repo_id": "stable-diffusion-v1-5/stable-diffusion-v1-5",
        "revision": "451f4fe16113bff5a5d2269ed5ad43b0592e9a14",
        "allow_patterns": ["tokenizer/*"],
        "provenance": "SD1.5 tokenizer, as in the official v1 loader.",
    },
    "clip": {
        "repo_id": "openai/clip-vit-large-patch14",
        "revision": "32bd64288804d66eefd0ccbe215aa642df71cc41",
        "allow_patterns": ["*.json", "merges.txt", "model.safetensors"],
        "provenance": "OpenAI CLIP ViT-L/14 for evaluation, same revision as BrushNet benchmark.",
    },
}


def main():
    models = ROOT / "models"
    models.mkdir(parents=True, exist_ok=True)
    with exclusive_lock(models / ".download.lock"):
        sources = {}
        for name, spec in SPECS.items():
            destination = models / name
            previous_root = ROOT.parent / "BrushNet_benchmark" / "models"
            previous_sources = previous_root / "sources.json"
            if name == "clip" and previous_sources.is_file() and not destination.exists():
                previous = json.loads(previous_sources.read_text()).get("clip", {})
                if previous.get("repo_id") == spec["repo_id"] and previous.get("revision") == spec["revision"]:
                    shutil.copytree(previous_root / "clip", destination)
            if name != "clip" or not (destination / "model.safetensors").is_file():
                for attempt in range(8):
                    try:
                        snapshot_download(repo_id=spec["repo_id"], revision=spec["revision"],
                                          allow_patterns=spec["allow_patterns"], local_dir=str(destination),
                                          local_dir_use_symlinks=False, cache_dir=str(ROOT / "cache" / "huggingface"),
                                          endpoint="https://huggingface.co", token=os.environ.get("HF_TOKEN", False),
                                          max_workers=2, resume_download=True)
                        break
                    except requests.RequestException as exc:
                        if attempt == 7:
                            raise
                        print(f"Download interrupted ({type(exc).__name__}); resume attempt {attempt + 2}/8", flush=True)
                        time.sleep(5 * (attempt + 1))
            files = {str(p.relative_to(destination)): sha256(p) for p in sorted(destination.rglob("*")) if p.is_file() and not any(part.startswith(".") for part in p.relative_to(destination).parts)}
            if not files:
                raise RuntimeError(f"No files downloaded for {name}")
            sources[name] = dict(spec, local_dir=str(destination), files_sha256=files)
            atomic_json(models / "sources.json", sources)
            print(f"READY {name}: {len(files)} verified local files", flush=True)
        previous_alex = ROOT.parent / "BrushNet_benchmark" / "cache" / "torch" / "alexnet-owt-7be5be79.pth"
        if previous_alex.is_file():
            target = ROOT / "cache" / "torch" / previous_alex.name
            target.parent.mkdir(parents=True, exist_ok=True)
            if not target.exists():
                shutil.copy2(previous_alex, target)
            if not sha256(target).startswith("7be5be79"):
                raise ValueError("AlexNet checksum mismatch")
        print("All model assets ready", flush=True)


if __name__ == "__main__":
    main()
