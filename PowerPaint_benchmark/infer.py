import argparse
import json
import math
import os
import random
import subprocess
import time
import traceback
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import numpy as np
import torch
from PIL import Image

from common import DEFAULT_DATA, ROOT, atomic_json, binary_mask, exclusive_lock, json_hash, load_pair, load_samples, now, runtime_versions, sha256
from gpu_guard import gpu_state, wait_for_memory
from powerpaint_adapter import QUALITY_NEGATIVE, VENDOR, VENDOR_COMMIT, check_prompt, load_pipeline, prompts_for


def arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--run-dir", type=Path, default=ROOT / "runs" / "powerpaint_v1_unipc_s42")
    parser.add_argument("--size", type=int, default=512)
    parser.add_argument("--output-size", type=int, default=256)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--guidance", type=float, default=7.5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--scheduler", choices=["unipc", "base"], default="unipc")
    parser.add_argument("--limit", type=int, default=0, help="First N sorted IDs; 0 = all. Not a random subset.")
    parser.add_argument("--memory-fraction", type=float, default=0.4)
    parser.add_argument("--gpu-wait-timeout", type=float, default=0, help="Seconds; 0 waits indefinitely without interrupting other jobs")
    return parser.parse_args()


def validated_inputs(dataset):
    samples = load_samples(dataset)
    manifest = json.loads((ROOT / "dataset_manifest.json").read_text())
    records = {r["id"]: r for r in manifest["samples"]}
    if Path(manifest["dataset_root"]).resolve() != dataset.resolve() or json_hash(manifest["samples"]) != manifest["manifest_sha256"]:
        raise ValueError("Run prepare_dataset.py for this dataset first")
    if sha256(dataset / "prompts.json") != manifest["prompts_sha256"] or {s["id"] for s in samples} != set(records):
        raise ValueError("Dataset annotations changed")
    for sample in samples:
        if any(sha256(sample[k]) != records[sample["id"]][k + "_sha256"] for k in ("image", "mask")):
            raise ValueError(f"Dataset content changed: {sample['id']}")
    sources = json.loads((ROOT / "models" / "sources.json").read_text())
    for name in ("base", "powerpaint", "tokenizer"):
        if name not in sources or not sources[name].get("files_sha256"):
            raise ValueError(f"Missing verified model: {name}")
        for relative, expected in sources[name]["files_sha256"].items():
            if sha256(ROOT / "models" / name / relative) != expected:
                raise ValueError(f"Model file changed: {name}/{relative}")
    return samples, manifest, sources


def run(args):
    if args.size < 64 or args.size % 8 or args.output_size < 64 or args.steps < 1 or args.limit < 0 or not 0 <= args.seed < 2**32:
        raise ValueError("Invalid size, steps, limit or seed")
    if not math.isfinite(args.guidance) or args.guidance < 1 or not 0 < args.memory_fraction <= 0.5 or args.gpu_wait_timeout < 0:
        raise ValueError("Invalid guidance, memory budget or wait timeout")
    samples, manifest, sources = validated_inputs(args.dataset)
    commit = subprocess.check_output(["git", "-C", str(VENDOR), "rev-parse", "HEAD"], text=True).strip()
    if commit != VENDOR_COMMIT:
        raise ValueError("Unexpected PowerPaint source revision")
    total = gpu_state()["total_mib"]
    admission = wait_for_memory(math.ceil(total * args.memory_fraction) + 2048, args.gpu_wait_timeout)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    torch.cuda.set_per_process_memory_fraction(args.memory_fraction)
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    config = {
        "schema_version": 1, "method": "PowerPaint-v1-text-guided", "powerpaint_commit": commit,
        "models": {k: sources[k] for k in ("base", "powerpaint", "tokenizer")},
        "dataset_root": str(args.dataset.resolve()), "manifest_sha256": manifest["manifest_sha256"],
        "sample_ids": [s["id"] for s in samples],
        "parameters": {"work_size": args.size, "output_size": args.output_size, "steps": args.steps,
                       "guidance_scale": args.guidance, "seed": args.seed,
                       "seed_policy": "reset Python, NumPy, torch CPU/CUDA and explicit CUDA generator to the same seed for each ID",
                       "scheduler": "UniPCMultistepScheduler" if args.scheduler == "unipc" else "base model scheduler",
                       "positive_prompt": "dataset prompt + ' P_obj' (10 learned vectors)",
                       "negative_prompt": QUALITY_NEGATIVE + "P_obj", "tradeoff": 1.0,
                       "dtype": "float16", "image_resize": "PIL BILINEAR", "mask_resize": "PIL NEAREST; white >127 means edit",
                       "masked_image": "Original RGB supplied; official pipeline masks after normalization, NOT black RGB prefill",
                       "postprocess": "resize only, no original-image blending", "tf32": False,
                       "safety_checker": "disabled for local remote-sensing benchmark; not a public service",
                       "memory_fraction": args.memory_fraction},
        "versions": runtime_versions(), "gpu": torch.cuda.get_device_name(0),
        "script_sha256": {n: sha256(ROOT / n) for n in ("infer.py", "common.py", "powerpaint_adapter.py", "gpu_guard.py")},
        "vendor_files_sha256": {n: sha256(VENDOR / n) for n in ("powerpaint/pipelines/pipeline_PowerPaint.py", "powerpaint/utils/utils.py")},
    }
    cfgpath = args.run_dir / "config.json"
    if cfgpath.exists() and json_hash(json.loads(cfgpath.read_text())) != json_hash(config):
        raise ValueError("Run configuration differs; choose a new --run-dir instead of mixing results")
    atomic_json(cfgpath, config)
    atomic_json(args.run_dir / "gpu_admission.json", {"checked_at": now(), **admission})
    generated, logs = args.run_dir / "generated", args.run_dir / "samples"
    generated.mkdir(exist_ok=True)
    logs.mkdir(exist_ok=True)
    print("Loading official local PowerPaint v1 safetensors...", flush=True)
    pipe = load_pipeline(args.scheduler)
    prompt_lengths = {}
    for sample in samples:
        positive, negative = prompts_for(sample["prompt"])
        prompt_lengths[sample["id"]] = [check_prompt(pipe.tokenizer, positive), check_prompt(pipe.tokenizer, negative)]
    atomic_json(args.run_dir / "prompt_lengths.json", prompt_lengths)
    atomic_json(args.run_dir / "scheduler_config.json", {"class": type(pipe.scheduler).__name__, "config": dict(pipe.scheduler.config)})
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
            wait_for_memory(1536, args.gpu_wait_timeout)
            random.seed(args.seed)
            np.random.seed(args.seed)
            torch.manual_seed(args.seed)
            torch.cuda.manual_seed_all(args.seed)
            image, mask = load_pair(sample)
            image = image.resize((args.size, args.size), Image.Resampling.BILINEAR)
            mask = Image.fromarray(binary_mask(mask.resize((args.size, args.size), Image.Resampling.NEAREST)).astype(np.uint8) * 255)
            positive, negative = prompts_for(sample["prompt"])
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
            begin = time.perf_counter()
            with torch.inference_mode():
                output = pipe(promptA=positive, promptB=positive, negative_promptA=negative, negative_promptB=negative,
                              tradoff=1.0, tradoff_nag=1.0, image=image, mask=mask,
                              height=args.size, width=args.size, num_inference_steps=args.steps,
                              guidance_scale=args.guidance, generator=torch.Generator("cuda").manual_seed(args.seed))
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - begin
            result = output.images[0].resize((args.output_size, args.output_size), Image.Resampling.BILINEAR)
            temp = outpath.with_suffix(".tmp.png")
            result.save(temp, format="PNG")
            temp.replace(outpath)
            record.update(status="success", finished_at=now(), seconds=elapsed, output_sha256=sha256(outpath),
                          peak_vram_mib=torch.cuda.max_memory_allocated() / 2**20,
                          peak_reserved_mib=torch.cuda.max_memory_reserved() / 2**20,
                          positive_prompt=positive, negative_prompt=negative, nsfw_content_detected=output.nsfw_content_detected)
            new_count += 1
            print(f"[{index+1}/{len(samples)}] {sid}: {elapsed:.2f}s, allocated {record['peak_vram_mib']:.0f} MiB, reserved {record['peak_reserved_mib']:.0f} MiB", flush=True)
        except Exception as exc:
            failed += 1
            record.update(status="failed", finished_at=now(), error=f"{type(exc).__name__}: {exc}")
            traceback.print_exc()
        atomic_json(recordpath, record)
        atomic_json(args.run_dir / "progress.json", {"updated_at": now(), "expected": len(samples), "new_successes": new_count, "skipped_this_invocation": skipped, "failed_this_invocation": failed, "last_id": sid, "invocation_seconds": time.perf_counter() - started})
        if failed:
            raise RuntimeError("Stopping on first failed sample to release GPU; inspect record and resume unchanged command")
    valid, timings = [], []
    for sample in samples:
        recordpath, outpath = logs / f"{sample['id']}.json", generated / f"{sample['id']}.png"
        if recordpath.exists() and outpath.exists():
            record = json.loads(recordpath.read_text())
            if record.get("status") == "success" and record.get("output_sha256") == sha256(outpath):
                valid.append(sample["id"])
                timings.append(record["seconds"])
    atomic_json(args.run_dir / "generation_summary.json", {
        "finished_at": now(), "status": "complete" if len(valid) == len(samples) else "partial",
        "expected": len(samples), "generated": len(valid), "missing_ids": sorted(set(config["sample_ids"]) - set(valid)),
        "new_successes": new_count, "skipped_this_invocation": skipped, "failed_this_invocation": failed,
        "total_generation_seconds": sum(timings), "mean_generation_seconds": float(np.mean(timings)) if timings else None,
        "invocation_seconds": time.perf_counter() - started,
    })
    if not args.limit and len(valid) != len(samples):
        raise RuntimeError("Generation incomplete")


if __name__ == "__main__":
    args = arguments()
    args.run_dir.mkdir(parents=True, exist_ok=True)
    with exclusive_lock(args.run_dir / ".inference.lock"):
        run(args)
