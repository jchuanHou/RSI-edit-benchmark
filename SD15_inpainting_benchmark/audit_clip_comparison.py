import argparse
import json
import math
from datetime import datetime
from pathlib import Path

import numpy as np
from transformers import CLIPProcessor

from common import DEFAULT_DATA, ROOT, atomic_json, json_hash, load_samples, now, safe_file, sha256

METRICS = ("global_clip_score", "local_clip_score")


def read_json(path):
    return json.loads(Path(path).read_text())


def validated_rows(path, expected_ids):
    rows = read_json(path)
    if not isinstance(rows, list) or len(rows) != len(expected_ids):
        raise ValueError(f"Wrong row count: {path}")
    ids = [row["id"] for row in rows]
    if len(set(ids)) != len(ids) or set(ids) != set(expected_ids):
        raise ValueError(f"Duplicate or missing sample IDs: {path}")
    for row in rows:
        for name in METRICS:
            value = row[name]
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                raise ValueError(f"Non-finite/invalid metric: {path}/{row['id']}/{name}")
    return {row["id"]: row for row in rows}


def main(directory):
    summary = read_json(directory / "summary.json")
    status = read_json(directory / "status.json")
    provenance = read_json(directory / "input_provenance.json")
    if summary["status"] != "complete" or status["status"] != "complete" or status["exit_code"] != 0:
        raise ValueError("Evaluation has not completed successfully")
    samples = load_samples(DEFAULT_DATA)
    ids = [row["id"] for row in samples]
    if len(ids) != 1550 or len(summary["results"]) != 8:
        raise ValueError("Unexpected sample or group count")
    started = datetime.fromisoformat(summary["started_at"]).timestamp()
    changed_paths = []
    for path in [DEFAULT_DATA / "prompts.json"] + [Path(row[k]) for row in samples for k in ("image", "mask")]:
        stat = path.stat()
        if stat.st_mtime > started or stat.st_ctime > started:
            changed_paths.append(str(path))
    if changed_paths:
        raise ValueError(f"Dataset files changed during or after evaluation: {changed_paths[:5]}")
    for label, info in provenance.items():
        run = Path(info["run_dir"])
        manifest = read_json(ROOT.parent / info["project"] / "dataset_manifest.json")
        if sha256(DEFAULT_DATA / "prompts.json") != manifest["prompts_sha256"] or json_hash(manifest["samples"]) != info["dataset_manifest_sha256"]:
            raise ValueError("Dataset manifest changed")
        records = {row["id"]: row for row in manifest["samples"]}
        for row in samples:
            for key in ("image", "mask"):
                if sha256(row[key]) != records[row["id"]][key + "_sha256"]:
                    raise ValueError("Dataset bytes changed")
        for name, digest in info["protected_files_sha256"].items():
            if sha256(run / name) != digest:
                raise ValueError(f"Protected file changed: {label}/{name}")
        if any(sha256(run / "generated" / f"{sid}.png") != digest for sid, digest in info["outputs"].items()):
            raise ValueError(f"Generated images changed: {label}")
    expected_groups = {(label, model) for label in provenance for model in ("b32", "l14")}
    actual_groups = [(report["benchmark"], report["clip_model"]) for report in summary["results"]]
    if len(set(actual_groups)) != 8 or set(actual_groups) != expected_groups:
        raise ValueError("The eight groups are not unique and complete")
    groups = []
    for report in summary["results"]:
        label, model = report["benchmark"], report["clip_model"]
        path = safe_file(directory, report["per_sample_file"])
        if sha256(path) != report["per_sample_sha256"]:
            raise ValueError("Per-sample report hash mismatch")
        rows = validated_rows(path, ids)
        if report["sample_set_sha256"] != json_hash(ids) or report["outputs_sha256"] != provenance[label]["outputs_sha256"]:
            raise ValueError("Mismatched sample/output identity")
        for metric in METRICS:
            values = np.asarray([rows[sid][metric] for sid in ids], dtype=np.float64)
            recorded = report["metrics"][metric]
            if not all(isinstance(recorded[key], (int, float)) and not isinstance(recorded[key], bool) and math.isfinite(recorded[key]) for key in ("value", "std")):
                raise ValueError("Non-finite or invalid summary statistics")
            if report["status"] != "complete" or report["evaluated"] != 1550 or report["missing_ids"]:
                raise ValueError("Incomplete group report")
            if recorded["num_images"] != 1550 or abs(float(values.mean()) - recorded["value"]) > 1e-10 or abs(float(values.std()) - recorded["std"]) > 1e-10:
                raise ValueError("Summary statistics do not match per-sample results")
        regression = {}
        if model == "l14":
            old = validated_rows(Path(provenance[label]["run_dir"]) / "eval_results_per_sample.json", ids)
            for metric in METRICS:
                regression[metric] = max(abs(rows[sid][metric] - old[sid][metric]) for sid in ids)
                if regression[metric] > 1e-3:
                    raise ValueError("L/14 regression failed")
        groups.append({"benchmark": label, "model": model, "count": len(rows), "all_scores_finite": True, "l14_max_abs_delta": regression})
    tokens, processors = {}, {}
    for key, metadata in summary["models"].items():
        source = metadata["source"]
        model_dir = Path(source["local_dir"])
        for name, digest in source["files_sha256"].items():
            if sha256(safe_file(model_dir, name)) != digest:
                raise ValueError("Model assets changed")
        processor = CLIPProcessor.from_pretrained(model_dir, local_files_only=True)
        tokens[key] = processor.tokenizer([row["prompt"] for row in samples], truncation=False, padding=False)["input_ids"]
        processors[key] = processor.image_processor.to_dict()
    for name, digest in summary["source_files_sha256"].items():
        if sha256(safe_file(ROOT, name)) != digest:
            raise ValueError("Evaluation source changed after the recorded run")
    audit = {
        "status": "passed", "audited_at": now(), "groups": groups,
        "original_reports_unchanged": True, "dataset_hashes_reverified": True,
        "dataset_mtime_ctime_before_evaluation_start": True,
        "same_prompt_token_ids_between_models": tokens["b32"] == tokens["l14"],
        "same_image_processor_between_models": processors["b32"] == processors["l14"],
        "summary_sha256": sha256(directory / "summary.json"),
        "audit_script_sha256": sha256(__file__),
    }
    atomic_json(directory / "audit.json", audit)
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", type=Path, required=True)
    args = parser.parse_args()
    directory = args.results_dir.resolve()
    if not directory.is_relative_to(ROOT.parent):
        parser.error("Results must be inside the workspace")
    main(directory)
