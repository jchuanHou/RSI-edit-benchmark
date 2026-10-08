import argparse
import json
import os
from pathlib import Path

from huggingface_hub import snapshot_download

ROOT = Path(__file__).resolve().parent
SPECS = {
    "base": {
        "repo_id": "stable-diffusion-v1-5/stable-diffusion-v1-5",
        "revision": "451f4fe16113bff5a5d2269ed5ad43b0592e9a14",
        "allow_patterns": ["model_index.json", "scheduler/*.json", "tokenizer/*", "text_encoder/config.json", "text_encoder/model.fp16.safetensors", "unet/config.json", "unet/diffusion_pytorch_model.fp16.safetensors", "vae/config.json", "vae/diffusion_pytorch_model.fp16.safetensors", "feature_extractor/*.json"],
        "provenance": "Community mirror of the original SD1.5 base, fp16 variant; not SD1.5 Inpainting.",
    },
    "brushnet": {
        "repo_id": "Sanster/brushnet_random_mask",
        "revision": "2201b8b01f0aa4456789fb6b525a8d15d2bde6e1",
        "allow_patterns": ["config.json", "diffusion_pytorch_model.fp16.safetensors"],
        "provenance": "Community fp16 distribution of random-mask BrushNet; not hosted by TencentARC. Official source: TencentARC/BrushNet README Google Drive checkpoint collection.",
    },
    "clip": {
        "repo_id": "openai/clip-vit-large-patch14",
        "revision": "32bd64288804d66eefd0ccbe215aa642df71cc41",
        "allow_patterns": ["*.json", "merges.txt", "model.safetensors"],
        "provenance": "OpenAI CLIP ViT-L/14 for evaluation, not generation.",
    },
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", choices=list(SPECS), default=list(SPECS))
    args = parser.parse_args()
    os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "120")
    (ROOT / "models").mkdir(exist_ok=True)
    lock_path = ROOT / "models" / "sources.json"
    lock = json.loads(lock_path.read_text()) if lock_path.exists() else {}
    for name in args.models:
        spec = SPECS[name]
        print(f"Downloading {name}: {spec['repo_id']} @ {spec['revision']}", flush=True)
        path = snapshot_download(
            repo_id=spec["repo_id"], revision=spec["revision"],
            allow_patterns=spec["allow_patterns"], local_dir=str(ROOT / "models" / name),
            cache_dir=str(ROOT / "cache" / "huggingface"),
            local_dir_use_symlinks=False, resume_download=True, max_workers=3,
        )
        lock[name] = {**spec, "local_dir": path}
        temporary = lock_path.with_suffix(".tmp")
        temporary.write_text(json.dumps(lock, ensure_ascii=False, indent=2) + "\n")
        temporary.replace(lock_path)
        print(f"READY {name}: {path}", flush=True)


if __name__ == "__main__":
    main()
