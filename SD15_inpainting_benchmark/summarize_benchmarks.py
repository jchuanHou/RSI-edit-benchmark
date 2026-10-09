import argparse
import itertools
import json
from pathlib import Path

import numpy as np

from common import ROOT, atomic_json, json_hash, now, safe_file, sha256
from evaluate_clip_comparison import RUNS

DIRECTIONS = {
    "b32_global_clip_score": 1, "b32_local_clip_score": 1,
    "l14_global_clip_score": 1, "l14_local_clip_score": 1,
    "background_mae": -1, "background_psnr": 1, "background_ssim": 1,
    "boundary_mae_vs_original": -1, "lpips": -1, "lpips_outside": -1,
    "lpips_inside_vs_original": -1, "boundary_lpips_vs_original": -1,
}
REGIONS = {"small_le_0.10": (0, 0.10), "medium_0.10_to_0.25": (0.10, 0.25), "large_gt_0.25": (0.25, 1.0)}


def read_json(path):
    return json.loads(Path(path).read_text())


def array_stats(values):
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 1 or not len(values) or not np.isfinite(values).all():
        raise ValueError("Expected nonempty finite one-dimensional data")
    count = len(values)
    trim = int(count * 0.05)
    ordered = np.sort(values)
    trimmed = ordered[trim:count - trim] if trim else ordered
    return {
        "n": count, "mean": float(values.mean()), "std": float(values.std()),
        "median": float(np.median(values)), "p05": float(np.quantile(values, .05)),
        "p95": float(np.quantile(values, .95)), "min": float(values.min()),
        "max": float(values.max()), "trimmed_mean_5pct_each_tail": float(trimmed.mean()),
    }


def rows_by_id(path, ids):
    rows = read_json(path)
    if not isinstance(rows, list) or len(rows) != len(ids):
        raise ValueError(f"Wrong record count: {path}")
    result = {row["id"]: row for row in rows}
    if len(result) != len(rows) or set(result) != set(ids):
        raise ValueError(f"Duplicate or mismatched IDs: {path}")
    return result


def main(output):
    workspace = ROOT.parent
    clip_dir = ROOT / "runs" / "clip_comparison_20261009"
    clip = read_json(clip_dir / "summary.json")
    audit = read_json(clip_dir / "audit.json")
    provenance = read_json(clip_dir / "input_provenance.json")
    if clip["status"] != "complete" or audit["status"] != "passed" or audit["summary_sha256"] != sha256(clip_dir / "summary.json"):
        raise ValueError("CLIP comparison is not verified")
    manifest = read_json(ROOT / "dataset_manifest.json")
    ids = [row["id"] for row in manifest["samples"]]
    if len(ids) != 1550 or len(set(ids)) != 1550 or json_hash(manifest["samples"]) != manifest["manifest_sha256"]:
        raise ValueError("Dataset manifest mismatch")
    fractions = np.asarray([row["mask_fraction"] for row in manifest["samples"]])
    prompts = [row["prompt"] for row in manifest["samples"]]
    prompt_groups = [[i for i, prompt in enumerate(prompts) if prompt == unique] for unique in sorted(set(prompts))]
    region_indices = {name: np.flatnonzero((fractions > low) & (fractions <= high)) for name, (low, high) in REGIONS.items()}
    if sum(len(indices) for indices in region_indices.values()) != len(ids):
        raise ValueError("Mask-area groups do not cover all samples")
    methods, vectors = {}, {}
    for label, project, run, method in RUNS:
        run_dir = workspace / project / "runs" / run
        config = read_json(run_dir / "config.json")
        generation = read_json(run_dir / "generation_summary.json")
        job = read_json(run_dir / "job_status.json")
        report = read_json(run_dir / "eval_results.json")
        if generation["status"] != "complete" or generation["generated"] != 1550 or generation["missing_ids"]:
            raise ValueError(f"Incomplete generation: {label}")
        if report["status"] != "complete" or report["evaluated"] != 1550 or report["missing_ids"] or report["errors"] or job["exit_code"] != 0:
            raise ValueError(f"Incomplete evaluation: {label}")
        if config["sample_ids"] != ids or config["method"] != method or report["sample_set_sha256"] != json_hash(ids) or report["dataset_manifest_sha256"] != manifest["manifest_sha256"]:
            raise ValueError("Sample set or method mismatch")
        for name, digest in provenance[label]["protected_files_sha256"].items():
            if sha256(run_dir / name) != digest:
                raise ValueError(f"Original report changed: {label}/{name}")
        old_rows = rows_by_id(run_dir / "eval_results_per_sample.json", ids)
        data = {key: np.asarray([old_rows[sid][key] for sid in ids], dtype=np.float64) for key in DIRECTIONS if not key.startswith(("b32_", "l14_"))}
        for key, values in data.items():
            metric = report["metrics"][key]
            if metric["num_images"] != 1550 or not np.isfinite([metric["value"], metric["std"]]).all() or not np.isfinite(values).all() or abs(float(values.mean()) - metric["value"]) > 1e-10 or abs(float(values.std()) - metric["std"]) > 1e-10:
                raise ValueError(f"Non-CLIP summary mismatch: {label}/{key}")
        for model in ("b32", "l14"):
            item = next(r for r in clip["results"] if r["benchmark"] == label and r["clip_model"] == model)
            path = safe_file(clip_dir, item["per_sample_file"])
            if sha256(path) != item["per_sample_sha256"] or item["outputs_sha256"] != report["outputs_sha256"]:
                raise ValueError("CLIP report/output identity mismatch")
            rows = rows_by_id(path, ids)
            for key in ("global_clip_score", "local_clip_score"):
                values = np.asarray([rows[sid][key] for sid in ids], dtype=np.float64)
                if not np.isfinite(values).all() or abs(float(values.mean()) - item["metrics"][key]["value"]) > 1e-10:
                    raise ValueError("CLIP summary mismatch")
                data[f"{model}_{key}"] = values
        records, output_hashes = [], {}
        for sid in ids:
            record = read_json(safe_file(run_dir / "samples", f"{sid}.json"))
            digest = sha256(safe_file(run_dir / "generated", f"{sid}.png"))
            if record["id"] != sid or record["status"] != "success" or record["output_sha256"] != digest:
                raise ValueError(f"Generated output mismatch: {label}/{sid}")
            output_hashes[sid] = digest
            records.append(record)
        if json_hash(output_hashes) != report["outputs_sha256"]:
            raise ValueError("Output set changed")
        timings = [r["seconds"] for r in records]
        if abs(sum(timings) - generation["total_generation_seconds"]) > 1e-6 or abs(float(np.mean(timings)) - generation["mean_generation_seconds"]) > 1e-10:
            raise ValueError("Generation timing summary mismatch")
        efficiency = {"pipeline_seconds": array_stats(timings), "peak_allocated_mib": array_stats([r["peak_vram_mib"] for r in records]), "recorded_pipeline_seconds_sum": sum(timings), "invocation_seconds": generation["invocation_seconds"], "new_successes": generation["new_successes"], "reused": generation["skipped_this_invocation"]}
        if all("peak_reserved_mib" in r for r in records):
            efficiency["peak_reserved_mib"] = array_stats([r["peak_reserved_mib"] for r in records])
        methods[label] = {
            "run_dir": str(run_dir.relative_to(workspace)), "status": "complete", "n": 1550,
            "finished_at": job["finished_at"], "config_sha256": json_hash(config),
            "config_bound_samples": sum("config_sha256" in r for r in records),
            "outputs_sha256": report["outputs_sha256"],
            "metrics": {key: array_stats(values) for key, values in data.items()},
            "frechet_wavelet_distance": report["metrics"]["frechet_wavelet_distance"],
            "background_exact_matches": report["background_exact_matches"], "efficiency": efficiency,
            "prompt_macro_means": {key: float(np.mean([values[indices].mean() for indices in prompt_groups])) for key, values in data.items()},
            "mask_area_means": {name: {key: float(values[indices].mean()) if len(indices) else None for key, values in data.items()} for name, indices in region_indices.items()},
            "generation_parameters": config["parameters"],
            "models": {key: {k: value[k] for k in ("repo_id", "revision") if k in value} for key, value in config["models"].items() if isinstance(value, dict)},
            "evaluation_versions": report["versions"],
            "source_sha256": {name: sha256(run_dir / name) for name in ("eval_results.json", "eval_results_per_sample.json", "generation_summary.json", "config.json", "job_status.json")},
        }
        vectors[label] = data
    paired = []
    for a, b in itertools.combinations(methods, 2):
        metrics = {}
        for key, direction in DIRECTIONS.items():
            delta = vectors[a][key] - vectors[b][key]
            advantage = delta * direction
            metrics[key] = {
                "raw_a_minus_b": array_stats(delta), "a_favorable_count": int((advantage > 0).sum()),
                "b_favorable_count": int((advantage < 0).sum()), "tie_count": int((advantage == 0).sum()),
                "a_favorable_pct": float((advantage > 0).mean() * 100),
            }
        paired.append({"a": a, "b": b, "n": len(ids), "metrics": metrics})
    result = {
        "status": "verified", "generated_at": now(), "n_per_method": len(ids),
        "dataset_manifest_sha256": manifest["manifest_sha256"], "sample_set_sha256": json_hash(ids),
        "distinct_prompts": len(prompt_groups), "prompt_frequency_min_max": [min(map(len, prompt_groups)), max(map(len, prompt_groups))],
        "mask_fraction": array_stats(fractions), "mask_area_groups": {name: {"lower_exclusive": REGIONS[name][0], "upper_inclusive": REGIONS[name][1], "n": len(indices)} for name, indices in region_indices.items()},
        "metric_directions": DIRECTIONS, "methods": methods, "paired_comparisons": paired,
        "clip_models": {key: {"repo_id": value["source"]["repo_id"], "revision": value["source"]["revision"], "files_sha256": value["source"]["files_sha256"], "max_prompt_token_length": value["max_prompt_token_length"], "truncated_prompt_count": value["truncated_prompt_count"]} for key, value in clip["models"].items()},
        "clip_audit": audit, "clip_summary_sha256": sha256(clip_dir / "summary.json"),
        "definitions": {
            "reference": "Non-CLIP distances compare generated images with ORIGINAL images, not edited ground truth.",
            "std": "Population standard deviation, ddof=0; not a confidence interval.",
            "paired": "Exact ID pairing; favorable count follows metric direction, not editing-success rate; no statistical significance test.",
            "trimmed_mean": "Remove floor(0.05*n) observations from each tail independently; 77 per tail for n=1550.",
            "prompt_macro": "Average images per exact prompt, then average over the 250 unique prompts; not main image-weighted protocol.",
            "mask_fraction": "Use original manifest mask_fraction; interval endpoints are left-open, right-closed.",
            "efficiency": "Historical synchronized pipeline-call seconds and process-local torch max_memory_allocated; GPU exclusivity was not controlled.",
            "publication": "Aggregate-only export: no sample IDs, prompts, source images, masks, generated images or weights.",
        },
        "analysis_script_sha256": sha256(__file__),
    }
    output = output.resolve()
    if not output.is_relative_to(workspace / "docs") or output.exists():
        raise ValueError("Choose a new output file under workspace/docs")
    atomic_json(output, result)
    print(f"Verified aggregate report: {output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    main(parser.parse_args().output)
