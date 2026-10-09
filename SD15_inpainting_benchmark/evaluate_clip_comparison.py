import argparse
import gc
import json
import os
import sys
import time
from contextlib import ExitStack
from importlib.metadata import version
from pathlib import Path

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import numpy as np
import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

from common import DEFAULT_DATA, ROOT, atomic_json, exclusive_lock, json_hash, load_samples, now, safe_file, sha256
from evaluate import clip_scores, metric_stats

WORKSPACE = ROOT.parent
RUNS = (
    ("BrushNet", "BrushNet_benchmark", "brushnet_sd15_random_s42", "BrushNet-SD15-random-mask"),
    ("PowerPaint", "PowerPaint_benchmark", "powerpaint_v1_unipc_s42", "PowerPaint-v1-text-guided"),
    ("SD1.5-Inpainting", "SD15_inpainting_benchmark", "sd15_inpainting_unipc_s42", "SD1.5-Inpainting-original"),
    ("SD2.0-Inpainting", "SD20_inpainting_benchmark", "sd20_inpainting_unipc_s42", "SD2.0-Inpainting-original"),
)
METRICS = ("global_clip_score", "local_clip_score")
DEFINITIONS = {
    "score": "100 times cosine similarity of L2-normalized projected image/text features; no clamping, no logit_scale, no softmax.",
    "global": "Generated RGB image versus the verbatim dataset prompt.",
    "local": "Same generated image with pixels outside the edit mask set to RGB black; NOT bounding-box crop; same prompt as Global.",
    "mask": "Convert to grayscale, resize NEAREST to generated image size, threshold strictly greater than 127.",
    "preprocessing": "Use the selected model's CLIPProcessor; resize shortest edge 224 BICUBIC, center crop 224, rescale and normalize; text padding and truncation enabled, maximum 77 tokens including special tokens.",
    "precision": "float32, eval, inference_mode, CUDA matmul TF32 disabled; two images (global/local) per forward batch, exactly reusing evaluate.clip_scores.",
    "aggregation": "Image-weighted arithmetic mean; population standard deviation (ddof=0); all 1550 samples required for each metric.",
    "comparability": "Compare methods within the same evaluator/metric, not absolute values across different CLIP models; single generation seed 42.",
}


def read_json(path):
    return json.loads(Path(path).read_text())


def verify_run(project, run, method, samples):
    project_dir = WORKSPACE / project
    run_dir = project_dir / "runs" / run
    config = read_json(run_dir / "config.json")
    manifest = read_json(project_dir / "dataset_manifest.json")
    ids = [sample["id"] for sample in samples]
    digest = json_hash(manifest["samples"])
    if len(samples) != 1550 or manifest["num_samples"] != 1550:
        raise ValueError("This comparison requires exactly 1550 samples")
    if digest != manifest["manifest_sha256"] or digest != config["manifest_sha256"]:
        raise ValueError(f"Manifest mismatch: {project}")
    if sha256(DEFAULT_DATA / "prompts.json") != manifest["prompts_sha256"]:
        raise ValueError("Dataset prompts changed")
    if config["sample_ids"] != ids or config["method"] != method:
        raise ValueError(f"Unexpected sample set or method: {project}")
    if Path(config["dataset_root"]).resolve() != DEFAULT_DATA.resolve():
        raise ValueError(f"Dataset location mismatch: {project}")
    records = {row["id"]: row for row in manifest["samples"]}
    if set(records) != set(ids) or len(records) != len(manifest["samples"]):
        raise ValueError("Manifest sample IDs are not unique and complete")
    generated = run_dir / "generated"
    if {p.name for p in generated.glob("*.png")} != {f"{sid}.png" for sid in ids}:
        raise ValueError(f"Generated PNG set is not exactly complete: {project}")
    config_digest = json_hash(config)
    outputs = {}
    config_bound = 0
    for sample in samples:
        sid = sample["id"]
        if sample["prompt"] != records[sid]["prompt"]:
            raise ValueError(f"Prompt mismatch: {sid}")
        for key in ("image", "mask"):
            if sha256(sample[key]) != records[sid][key + "_sha256"]:
                raise ValueError(f"Dataset input changed: {sid}/{key}")
        path = safe_file(generated, f"{sid}.png")
        record = read_json(safe_file(run_dir / "samples", f"{sid}.json"))
        outputs[sid] = sha256(path)
        if record.get("id") != sid or record.get("status") != "success" or record.get("output_sha256") != outputs[sid]:
            raise ValueError(f"Unverified generated output: {project}/{sid}")
        if "config_sha256" in record:
            if record["config_sha256"] != config_digest:
                raise ValueError(f"Per-sample configuration changed: {project}/{sid}")
            config_bound += 1
        elif project.startswith(("SD15_", "SD20_")):
            raise ValueError(f"Missing per-sample configuration digest: {project}/{sid}")
        with Image.open(path) as image:
            if image.mode != "RGB" or image.size != (config["parameters"]["output_size"],) * 2:
                raise ValueError(f"Unexpected generated image format: {project}/{sid}")
            image.verify()
    old = read_json(run_dir / "eval_results.json")
    if old["outputs_sha256"] != json_hash(outputs) or old["evaluated"] != 1550:
        raise ValueError(f"Outputs differ from the previous evaluation: {project}")
    protected = {name: sha256(run_dir / name) for name in (
        "config.json", "generation_summary.json", "job_status.json", "eval_results.json",
        "eval_results.txt", "eval_results_per_sample.json",
    )}
    return {
        "project": project, "run_dir": str(run_dir), "method": method,
        "sample_set_sha256": json_hash(ids), "dataset_manifest_sha256": digest,
        "outputs_sha256": json_hash(outputs), "outputs": outputs,
        "config_sha256": config_digest, "config_bound_samples": config_bound,
        "generation_parameters": config["parameters"], "protected_files_sha256": protected,
        "old_clip_metrics": {key: old["metrics"][key] for key in METRICS},
    }


def verify_assets(key):
    if key == "b32":
        directory = ROOT / "models" / "clip_b32"
        source = read_json(directory / "sources.json")
        expected = ("openai/clip-vit-base-patch32", "3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268", 32, 512)
    else:
        directory = ROOT / "models" / "clip"
        source = read_json(ROOT / "models" / "sources.json")["clip"]
        expected = ("openai/clip-vit-large-patch14", "32bd64288804d66eefd0ccbe215aa642df71cc41", 14, 768)
    if (source["repo_id"], source["revision"]) != expected[:2]:
        raise ValueError(f"Wrong evaluator identity: {key}")
    required = {"config.json", "model.safetensors", "preprocessor_config.json", "tokenizer.json", "tokenizer_config.json", "merges.txt", "vocab.json", "special_tokens_map.json"}
    if not required.issubset(source["files_sha256"]):
        raise ValueError(f"Incomplete model provenance: {key}")
    for filename, digest in source["files_sha256"].items():
        if sha256(safe_file(directory, filename)) != digest:
            raise ValueError(f"Model asset changed: {key}/{filename}")
    config = read_json(directory / "config.json")
    if config["vision_config"]["patch_size"] != expected[2] or config["projection_dim"] != expected[3]:
        raise ValueError(f"Wrong model architecture: {key}")
    return directory, source


def evaluate(args, status):
    torch.set_num_threads(4)
    torch.set_default_dtype(torch.float32)
    torch.backends.cuda.matmul.allow_tf32 = False
    if args.device == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but unavailable")
        free, total = torch.cuda.mem_get_info()
        if free < 8 * 1024 ** 3:
            raise RuntimeError("At least 8 GiB free GPU memory is required")
        torch.cuda.set_per_process_memory_fraction(0.4)
    samples = load_samples(DEFAULT_DATA)
    ids = [sample["id"] for sample in samples]
    versions = {name: version(name) for name in ("torch", "torchvision", "transformers", "tokenizers", "huggingface-hub", "safetensors", "numpy", "Pillow", "scipy", "scikit-image")}
    versions["python"] = sys.version
    versions["cuda"] = torch.version.cuda
    device = torch.cuda.get_device_name() if args.device == "cuda" else "CPU"
    started = time.perf_counter()
    with ExitStack() as locks:
        for _, project, run, _ in RUNS:
            for name in (".inference.lock", ".evaluation.lock"):
                locks.enter_context(exclusive_lock(WORKSPACE / project / "runs" / run / name))
        provenance = {}
        for label, project, run, method in RUNS:
            print(f"VERIFY {label}", flush=True)
            provenance[label] = verify_run(project, run, method, samples)
        if len({p["dataset_manifest_sha256"] for p in provenance.values()}) != 1:
            raise ValueError("Benchmarks do not share the same dataset manifest")
        atomic_json(args.output_dir / "input_provenance.json", provenance)
        results, models = [], {}
        for key in ("b32", "l14"):
            directory, source = verify_assets(key)
            model, loading = CLIPModel.from_pretrained(directory, use_safetensors=True, local_files_only=True, output_loading_info=True)
            if any(loading.get(k) for k in ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs")):
                raise ValueError(f"CLIP loading mismatch: {loading}")
            model = model.to(args.device).eval().requires_grad_(False)
            processor = CLIPProcessor.from_pretrained(directory, local_files_only=True)
            tokenized = processor.tokenizer([sample["prompt"] for sample in samples], truncation=False, padding=False)
            lengths = [len(tokens) for tokens in tokenized["input_ids"]]
            models[key] = {
                "source": source, "image_processor": processor.image_processor.to_dict(),
                "tokenizer_max_length": processor.tokenizer.model_max_length,
                "max_prompt_token_length": max(lengths),
                "truncated_prompt_count": sum(n > processor.tokenizer.model_max_length for n in lengths),
                "loading_info": loading,
            }
            atomic_json(args.output_dir / "models_metadata.json", models)
            for label, _, _, _ in RUNS:
                run_info = provenance[label]
                run_dir = Path(run_info["run_dir"])
                status.update(status="running", current_model=key, current_benchmark=label, updated_at=now())
                atomic_json(args.output_dir / "status.json", status)
                print(f"START {key} {label}: 1550 samples", flush=True)
                if args.device == "cuda":
                    torch.cuda.synchronize()
                group_started = time.perf_counter()
                rows = clip_scores(model, processor, samples, run_dir / "generated", args.device)
                if args.device == "cuda":
                    torch.cuda.synchronize()
                seconds = time.perf_counter() - group_started
                if list(rows) != ids:
                    raise ValueError("Incomplete or reordered CLIP result rows")
                metrics = {name: metric_stats([row[name] for row in rows.values()]) for name in METRICS}
                if any(value["num_images"] != 1550 for value in metrics.values()):
                    raise ValueError("Metric dropped samples")
                for sid, digest in run_info["outputs"].items():
                    if sha256(run_dir / "generated" / f"{sid}.png") != digest:
                        raise ValueError("Generated output changed during evaluation")
                regression = {}
                if key == "l14":
                    previous = {r["id"]: r for r in read_json(run_dir / "eval_results_per_sample.json")}
                    if set(previous) != set(rows):
                        raise ValueError("Old per-sample report has a different sample set")
                    for name in METRICS:
                        delta = metrics[name]["value"] - run_info["old_clip_metrics"][name]["value"]
                        maximum = max(abs(row[name] - previous[sid][name]) for sid, row in rows.items())
                        regression[name] = {"mean_delta_vs_previous": delta, "max_abs_per_sample_delta": maximum, "within_tolerance": abs(delta) <= 1e-4 and maximum <= 1e-3}
                report = {
                    "benchmark": label, "clip_model": key, "status": "complete", "expected": 1550,
                    "evaluated": len(rows), "missing_ids": [], "finished_at": now(),
                    "metrics": metrics, "evaluation_seconds": seconds,
                    "sample_set_sha256": run_info["sample_set_sha256"],
                    "dataset_manifest_sha256": run_info["dataset_manifest_sha256"],
                    "outputs_sha256": run_info["outputs_sha256"],
                    "config_sha256": run_info["config_sha256"], "clip": source,
                    "versions": versions, "device": device, "definitions": DEFINITIONS,
                    "l14_regression": regression,
                }
                stem = f"{run_info['project']}_{key}"
                atomic_json(args.output_dir / f"{stem}_per_sample.json", list(rows.values()))
                atomic_json(args.output_dir / f"{stem}.json", report)
                report["per_sample_file"] = f"{stem}_per_sample.json"
                report["per_sample_sha256"] = sha256(args.output_dir / report["per_sample_file"])
                results.append(report)
                status["completed_groups"] = len(results)
                atomic_json(args.output_dir / "status.json", status)
                print(f"DONE {key} {label}: {json.dumps(metrics)}; {seconds:.2f}s", flush=True)
            del model, processor
            gc.collect()
            if args.device == "cuda":
                torch.cuda.empty_cache()
        for info in provenance.values():
            for name, digest in info["protected_files_sha256"].items():
                if sha256(Path(info["run_dir"]) / name) != digest:
                    raise ValueError(f"Original report/config changed: {info['project']}/{name}")
        if len(results) != 8 or any(not item["within_tolerance"] for report in results for item in report["l14_regression"].values()):
            raise ValueError("Eight-group completion or L/14 regression validation failed")
        summary = {
            "status": "complete", "started_at": status["started_at"], "finished_at": now(),
            "groups": 8, "samples_per_group": 1550, "image_evaluator_pairs": 12400,
            "scalar_scores": 24800, "unique_prompts": len({sample["prompt"] for sample in samples}),
            "elapsed_seconds": time.perf_counter() - started, "definitions": DEFINITIONS,
            "versions": versions, "device": device, "models": models, "results": results,
            "original_reports_unchanged": True,
            "source_files_sha256": {name: sha256(ROOT / name) for name in ("evaluate_clip_comparison.py", "evaluate.py", "common.py", "prepare_clip_b32.py")},
        }
        atomic_json(args.output_dir / "summary.json", summary)
        status.update(status="complete", completed_groups=8, finished_at=now(), exit_code=0)
        atomic_json(args.output_dir / "status.json", status)
        print(f"COMPLETE: {args.output_dir / 'summary.json'}", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    args = parser.parse_args()
    args.output_dir = args.output_dir.resolve()
    if not args.output_dir.is_relative_to(WORKSPACE):
        parser.error("Output directory must be inside the workspace")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    status = {"status": "validating", "started_at": now(), "pid": os.getpid(), "completed_groups": 0, "total_groups": 8}
    atomic_json(args.output_dir / "status.json", status)
    try:
        evaluate(args, status)
    except BaseException as exc:
        status.update(status="failed", finished_at=now(), exit_code=1, error=f"{type(exc).__name__}: {exc}")
        atomic_json(args.output_dir / "status.json", status)
        raise


if __name__ == "__main__":
    main()
