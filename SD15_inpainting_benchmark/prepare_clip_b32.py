import gc
import json
import shutil
from pathlib import Path

import torch
from huggingface_hub import hf_hub_download
from safetensors.torch import save_file

from common import ROOT, atomic_json, exclusive_lock, now, sha256

REPO = "openai/clip-vit-base-patch32"
REVISION = "3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268"
WEIGHT_SHA256 = "a63082132ba4f97a80bea76823f544493bffa8082296d62d71581a4feff1576f"
FILES = (
    "config.json", "merges.txt", "preprocessor_config.json",
    "special_tokens_map.json", "tokenizer.json", "tokenizer_config.json", "vocab.json",
)


def main():
    target = ROOT / "models" / "clip_b32"
    target.mkdir(parents=True, exist_ok=True)
    with exclusive_lock(target / ".prepare.lock"):
        source_path = target / "sources.json"
        if source_path.exists():
            source = json.loads(source_path.read_text())
            if source["repo_id"] != REPO or source["revision"] != REVISION:
                raise ValueError("Existing B/32 model has a different identity")
            for name, digest in source["files_sha256"].items():
                if name not in (*FILES, "model.safetensors") or sha256(target / name) != digest:
                    raise ValueError("Existing B/32 asset checksum mismatch")
            print("B/32 assets already verified", flush=True)
            return
        cache = ROOT / "cache" / "clip_b32_download"
        for name in FILES:
            path = hf_hub_download(REPO, name, revision=REVISION, cache_dir=cache, token=False)
            shutil.copyfile(path, target / name)
        path = Path(hf_hub_download(REPO, "pytorch_model.bin", revision=REVISION, cache_dir=cache, token=False))
        if sha256(path) != WEIGHT_SHA256:
            raise ValueError("Official B/32 weight checksum mismatch")
        state = torch.load(path, map_location="cpu", weights_only=True)
        if not isinstance(state, dict) or not state or not all(isinstance(k, str) and isinstance(v, torch.Tensor) for k, v in state.items()):
            raise ValueError("Expected a tensor-only CLIP state dictionary")
        tensors = {key: tensor.detach().contiguous() for key, tensor in state.items()}
        save_file(tensors, str(target / "model.safetensors"), metadata={"format": "pt"})
        del tensors, state
        gc.collect()
        source = {
            "repo_id": REPO,
            "revision": REVISION,
            "local_dir": str(target),
            "prepared_at": now(),
            "original_weight_file": "pytorch_model.bin",
            "original_weight_sha256": WEIGHT_SHA256,
            "conversion": "Official fixed-revision file SHA256 verified; torch.load(weights_only=True, map_location=cpu); unchanged tensors saved as safetensors.",
            "files_sha256": {name: sha256(target / name) for name in (*FILES, "model.safetensors")},
        }
        atomic_json(source_path, source)
        print(f"Prepared verified B/32 assets: {target}", flush=True)


if __name__ == "__main__":
    main()
