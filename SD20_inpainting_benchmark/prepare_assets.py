import argparse
import hashlib
import json
import os
import shutil
import time
from pathlib import Path

from huggingface_hub import HfApi, hf_hub_download
from requests import RequestException

from common import ROOT, atomic_json, exclusive_lock, safe_file, sha256
from model_spec import BASE_REPOSITORY, BASE_REVISION, CLIP_REPOSITORY, CLIP_REVISION, CROSSCHECK_REPOSITORY, CROSSCHECK_REVISION, MODEL_FILES, ORIGINAL_REPOSITORY, WEIGHTS_SHA256, validate_architecture


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


def download_base():
    token = os.environ.get("HF_TOKEN", False)
    api = HfApi(endpoint="https://huggingface.co", token=token)
    metadata = api.model_info(BASE_REPOSITORY, revision=BASE_REVISION, files_metadata=True)
    if metadata.sha != BASE_REVISION:
        raise ValueError("Unexpected model revision")
    remote = {item.rfilename: item for item in metadata.siblings}
    if not set(MODEL_FILES).issubset(remote):
        raise ValueError("Incomplete model snapshot")
    destination = ROOT / "models" / "base"
    digests, git_blobs = {}, {}
    for relative in MODEL_FILES:
        item = remote[relative]
        expected = WEIGHTS_SHA256.get(relative)
        if expected and (not item.lfs or item.lfs["sha256"] != expected):
            raise ValueError(f"Remote weights differ from cross-checked release: {relative}")
        for attempt in range(3):
            try:
                downloaded = hf_hub_download(
                    BASE_REPOSITORY, relative, revision=BASE_REVISION,
                    local_dir=str(destination), local_dir_use_symlinks=False,
                    cache_dir=str(ROOT / "cache" / "huggingface"),
                    endpoint="https://huggingface.co", token=token, resume_download=True,
                )
                break
            except RequestException:
                if attempt == 2:
                    raise
                print(f"Interrupted download; retrying {relative}", flush=True)
                time.sleep(5 * (attempt + 1))
        path = Path(downloaded)
        if path.stat().st_size != item.size:
            raise ValueError(f"Downloaded size mismatch: {relative}")
        digest = sha256(path)
        if expected and digest != expected:
            raise ValueError(f"Downloaded weight checksum mismatch: {relative}")
        if not expected:
            content = path.read_bytes()
            blob = hashlib.sha1(b"blob " + str(len(content)).encode() + b"\0" + content).hexdigest()
            if blob != item.blob_id:
                raise ValueError(f"Downloaded Git blob checksum mismatch: {relative}")
            git_blobs[relative] = blob
        digests[relative] = digest
        print(f"Verified base/{relative}", flush=True)
    configs = [json.loads((destination / name).read_text()) for name in (
        "unet/config.json", "text_encoder/config.json", "scheduler/scheduler_config.json", "model_index.json",
    )]
    validate_architecture(*configs)
    return {
        "repo_id": BASE_REPOSITORY, "revision": BASE_REVISION, "original_repo_id": ORIGINAL_REPOSITORY,
        "local_dir": str(destination), "files_sha256": digests, "config_git_blob_ids": git_blobs,
        "provenance": "Public sd2-community mirror of the deprecated Stability AI SD2.0 inpainting release; not affiliated with Stability AI. Original URL returned HTTP 401 during preparation. No Text2Earth or other fine-tuned weights.",
        "crosscheck": {"repo_id": CROSSCHECK_REPOSITORY, "revision": CROSSCHECK_REVISION,
                       "matching_lfs_sha256": WEIGHTS_SHA256,
                       "scope": "Both public repositories advertise identical LFS hashes; original repository unavailable for direct verification."},
        "variant": "fp16", "prediction_type": "epsilon", "license": "CreativeML Open RAIL++-M",
    }


def main():
    parser = argparse.ArgumentParser(description="Download pinned SD2.0 inpainting safetensors and copy aligned evaluation assets")
    parser.add_argument("--source-project", type=Path, default=ROOT.parent / "SD15_inpainting_benchmark")
    args = parser.parse_args()
    source_root = args.source_project.resolve()
    models = ROOT / "models"
    models.mkdir(parents=True, exist_ok=True)
    with exclusive_lock(models / ".assets.lock"):
        previous = json.loads((source_root / "models" / "sources.json").read_text())
        clip = dict(previous["clip"])
        if clip["repo_id"] != CLIP_REPOSITORY or clip["revision"] != CLIP_REVISION or not clip.get("files_sha256"):
            raise ValueError("Unexpected evaluation CLIP")
        for relative, expected in clip["files_sha256"].items():
            source = safe_file(source_root / "models" / "clip", relative)
            copy_verified(source, models / "clip" / relative, expected)
        clip.update(copied_from=str(source_root / "models" / "clip"), local_dir=str(models / "clip"))
        alex = source_root / "cache" / "torch" / "alexnet-owt-7be5be79.pth"
        digest = previous["lpips_backbone"]["sha256"]
        if not digest.startswith("7be5be79"):
            raise ValueError("Unexpected AlexNet checksum")
        target = ROOT / "cache" / "torch" / alex.name
        copy_verified(alex, target, digest)
        selected = {
            "base": download_base(), "clip": clip,
            "lpips_backbone": {"file": str(target), "sha256": digest, "copied_from": str(alex)},
        }
        atomic_json(models / "sources.json", selected)
        print("Ready: pinned original SD2.0 inpainting mirror, CLIP ViT-L/14 and AlexNet; no custom pipeline or adapters.", flush=True)


if __name__ == "__main__":
    main()
