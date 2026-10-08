import argparse
import json
import os
import random
import subprocess
import time
import traceback
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import numpy as np
import torch
from PIL import Image
from diffusers import BrushNetModel, StableDiffusionBrushNetPipeline, UniPCMultistepScheduler

from common import DEFAULT_DATA, ROOT, atomic_json, binary_mask, exclusive_lock, json_hash, load_pair, load_samples, now, runtime_versions, sha256


def arguments():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", type=Path, default=DEFAULT_DATA)
    p.add_argument("--run-dir", type=Path, default=ROOT / "runs" / "brushnet_sd15_random_s42")
    p.add_argument("--size", type=int, default=512)
    p.add_argument("--output-size", type=int, default=256)
    p.add_argument("--steps", type=int, default=50)
    p.add_argument("--guidance", type=float, default=7.5)
    p.add_argument("--conditioning-scale", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--limit", type=int, default=0, help="Maximum samples attempted this invocation; 0 means all. Resume uses identical per-ID seeds.")
    return p.parse_args()


def run(args):
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this benchmark")
    if args.size < 64 or args.size % 8 or args.output_size < 64 or args.steps < 1 or args.limit < 0:
        raise ValueError("Invalid size, steps or limit")
    samples = load_samples(args.dataset)
    manifest = json.loads((ROOT / "dataset_manifest.json").read_text())
    records = {r["id"]: r for r in manifest["samples"]}
    if Path(manifest["dataset_root"]).resolve() != args.dataset.resolve():
        raise ValueError("Run prepare_dataset.py for this dataset first")
    if sha256(args.dataset / "prompts.json") != manifest["prompts_sha256"] or {s["id"] for s in samples} != set(records):
        raise ValueError("Dataset annotations changed; regenerate and review manifest")
    for sample in samples:
        record = records[sample["id"]]
        if any(sha256(sample[k]) != record[k + "_sha256"] for k in ("image", "mask")):
            raise ValueError(f"Dataset content changed: {sample['id']}")
    sources = json.loads((ROOT / "models" / "sources.json").read_text())
    for name in ("base", "brushnet"):
        if name not in sources:
            raise RuntimeError(f"Download incomplete: {name}")
    commit = subprocess.check_output(["git", "-C", str(ROOT / "vendor" / "BrushNet"), "rev-parse", "HEAD"], text=True).strip()
    config = {
        "schema_version": 1, "method": "BrushNet-SD15-random-mask", "brushnet_commit": commit,
        "models": {k: sources[k] for k in ("base", "brushnet")}, "dataset_root": str(args.dataset.resolve()),
        "manifest_sha256": manifest["manifest_sha256"], "sample_ids": [s["id"] for s in samples],
        "parameters": {"work_size": args.size, "output_size": args.output_size, "steps": args.steps, "guidance_scale": args.guidance, "brushnet_conditioning_scale": args.conditioning_scale, "seed": args.seed, "seed_policy": "reset Python, NumPy, torch CPU/CUDA and explicit CUDA generator to the same seed for each ID", "scheduler": "UniPCMultistepScheduler", "negative_prompt": "", "dtype": "float16", "image_resize": "PIL BILINEAR", "mask_resize": "PIL NEAREST; white >127 means edit", "postprocess": "resize only, no original-image blending", "safety_checker": "disabled for local remote-sensing benchmark; not a public service", "tf32": False},
        "versions": runtime_versions(), "gpu": torch.cuda.get_device_name(0),
        "script_sha256": {n: sha256(ROOT / n) for n in ("infer.py", "common.py")},
    }
    cfgpath = args.run_dir / "config.json"
    if cfgpath.exists():
        if json_hash(json.loads(cfgpath.read_text())) != json_hash(config):
            raise ValueError("Run configuration differs; use a new --run-dir instead of mixing results")
    else:
        atomic_json(cfgpath, config)
    generated, logs = args.run_dir / "generated", args.run_dir / "samples"
    generated.mkdir(exist_ok=True)
    logs.mkdir(exist_ok=True)
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    print("Loading local safetensors models...", flush=True)
    brush = BrushNetModel.from_pretrained(ROOT / "models" / "brushnet", variant="fp16", torch_dtype=torch.float16, use_safetensors=True, local_files_only=True)
    pipe = StableDiffusionBrushNetPipeline.from_pretrained(
        ROOT / "models" / "base", brushnet=brush, variant="fp16", torch_dtype=torch.float16,
        use_safetensors=True, local_files_only=True, safety_checker=None, requires_safety_checker=False,
        low_cpu_mem_usage=False,
    ).to("cuda")
    for name in ("unet", "brushnet", "vae", "text_encoder"):
        getattr(pipe, name).eval().requires_grad_(False)
    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
    atomic_json(args.run_dir / "scheduler_config.json", dict(pipe.scheduler.config))
    pipe.set_progress_bar_config(disable=True)
    selected = samples[:args.limit] if args.limit else samples
    started, failed, skipped, new_count = time.perf_counter(), 0, 0, 0
    for index, sample in enumerate(selected):
        sid = sample["id"]
        outpath, recordpath = generated / f"{sid}.png", logs / f"{sid}.json"
        if outpath.exists() and recordpath.exists():
            previous = json.loads(recordpath.read_text())
            if previous.get("status") == "success" and previous.get("output_sha256") == sha256(outpath):
                skipped += 1
                continue
        record = {"id": sid, "seed": args.seed, "started_at": now(), "status": "running"}
        atomic_json(recordpath, record)
        try:
            random.seed(args.seed)
            np.random.seed(args.seed)
            torch.manual_seed(args.seed)
            torch.cuda.manual_seed_all(args.seed)
            image, mask = load_pair(sample)
            image = image.resize((args.size, args.size), Image.Resampling.BILINEAR)
            mask = mask.resize((args.size, args.size), Image.Resampling.NEAREST)
            m = binary_mask(mask)
            masked = Image.fromarray(np.asarray(image) * (~m[..., None]))
            mask_rgb = Image.fromarray((m.astype(np.uint8) * 255)).convert("RGB")
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
            begin = time.perf_counter()
            with torch.inference_mode():
                result = pipe(prompt=sample["prompt"], image=masked, mask=mask_rgb,
                              height=args.size, width=args.size, num_inference_steps=args.steps,
                              guidance_scale=args.guidance, negative_prompt="",
                              generator=torch.Generator("cuda").manual_seed(args.seed),
                              brushnet_conditioning_scale=args.conditioning_scale).images[0]
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - begin
            result = result.resize((args.output_size, args.output_size), Image.Resampling.BILINEAR)
            temp = outpath.with_suffix(".tmp.png")
            result.save(temp, format="PNG")
            temp.replace(outpath)
            record.update(status="success", finished_at=now(), seconds=elapsed, output_sha256=sha256(outpath), peak_vram_mib=torch.cuda.max_memory_allocated() / 2**20)
            new_count += 1
            print(f"[{index+1}/{len(samples)}] {sid}: {elapsed:.2f}s, VRAM {record['peak_vram_mib']:.0f} MiB", flush=True)
        except Exception as exc:
            failed += 1
            record.update(status="failed", finished_at=now(), error=f"{type(exc).__name__}: {exc}")
            traceback.print_exc()
            torch.cuda.empty_cache()
        atomic_json(recordpath, record)
        atomic_json(args.run_dir / "progress.json", {"updated_at": now(), "expected": len(samples), "new_successes": new_count, "skipped_this_invocation": skipped, "failed_this_invocation": failed, "last_id": sid, "invocation_seconds": time.perf_counter() - started})
    valid, timings = [], []
    for sample in samples:
        recordpath, outpath = logs / f"{sample['id']}.json", generated / f"{sample['id']}.png"
        if recordpath.exists() and outpath.exists():
            r = json.loads(recordpath.read_text())
            if r.get("status") == "success" and r.get("output_sha256") == sha256(outpath):
                valid.append(sample["id"])
                timings.append(r["seconds"])
    atomic_json(args.run_dir / "generation_summary.json", {
        "finished_at": now(), "status": "complete" if len(valid) == len(samples) else "partial",
        "expected": len(samples), "generated": len(valid), "missing_ids": sorted(set(config["sample_ids"]) - set(valid)),
        "new_successes": new_count, "skipped_this_invocation": skipped, "failed_this_invocation": failed,
        "total_generation_seconds": sum(timings), "mean_generation_seconds": float(np.mean(timings)) if timings else None,
        "invocation_seconds": time.perf_counter() - started,
    })
    if failed or (not args.limit and len(valid) != len(samples)):
        raise RuntimeError("Generation incomplete; see samples/*.json and rerun unchanged command")


if __name__ == "__main__":
    args = arguments()
    args.run_dir.mkdir(parents=True, exist_ok=True)
    with exclusive_lock(args.run_dir / ".inference.lock"):
        run(args)
