import argparse
import gc
import json
import os
import tempfile
import time
from pathlib import Path

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from scipy.ndimage import binary_dilation, binary_erosion
from skimage.metrics import structural_similarity
from transformers import CLIPModel, CLIPProcessor

from common import DEFAULT_DATA, ROOT, atomic_json, binary_mask, exclusive_lock, json_hash, load_pair, load_samples, now, runtime_versions, sha256


def metric_stats(values):
    valid = np.asarray([v for v in values if v is not None], dtype=np.float64)
    if not len(valid):
        return {"value": None, "std": None, "num_images": 0}
    if not np.isfinite(valid).all():
        raise ValueError("Non-finite metric values")
    return {"value": float(valid.mean()), "std": float(valid.std()), "num_images": len(valid)}


def load_lpips(device):
    import lpips
    directory = ROOT / "cache" / "torch"
    directory.mkdir(parents=True, exist_ok=True)
    weights = directory / "alexnet-owt-7be5be79.pth"
    if not weights.exists():
        torch.hub.download_url_to_file("https://download.pytorch.org/models/alexnet-owt-7be5be79.pth", str(weights), hash_prefix="7be5be79", progress=True)
    if not sha256(weights).startswith("7be5be79"):
        raise ValueError("AlexNet checksum mismatch")
    alex_state = torch.load(weights, map_location="cpu", weights_only=True)
    model = lpips.LPIPS(net="alex", pretrained=False, pnet_rand=True, verbose=False)
    backbone = {key: alex_state["features." + key.split(".", 1)[1]] for key in model.net.state_dict()}
    model.net.load_state_dict(backbone, strict=True)
    linear_path = Path(lpips.__file__).parent / "weights" / "v0.1" / "alex.pth"
    linear_state = torch.load(linear_path, map_location="cpu", weights_only=True)
    result = model.load_state_dict(linear_state, strict=False)
    if result.unexpected_keys or any(k.startswith("lin") and not k.startswith("lins.") for k in result.missing_keys):
        raise ValueError(f"Invalid LPIPS linear weights: {result}")
    return model.to(device).eval().requires_grad_(False)


def to_tensor(image, device):
    arr = np.asarray(image).copy()
    return torch.from_numpy(arr).permute(2, 0, 1).float().unsqueeze(0).to(device) / 127.5 - 1.0


def pixel_metrics(reference, generated, mask, band_width):
    a, b = np.asarray(reference, dtype=np.float32) / 255, np.asarray(generated, dtype=np.float32) / 255
    outside = ~mask
    error = np.mean((a - b) ** 2, axis=-1)
    mse = float(error[outside].mean()) if outside.any() else None
    _, ssim_map = structural_similarity(a, b, data_range=1.0, channel_axis=-1, full=True, win_size=7)
    if ssim_map.ndim == 3:
        ssim_map = ssim_map.mean(axis=-1)
    band = binary_dilation(mask, iterations=band_width) & ~binary_erosion(mask, iterations=band_width)
    return {
        "background_mae": float(np.abs(a - b)[outside].mean()) if outside.any() else None,
        "background_psnr": float(-10 * np.log10(mse)) if mse is not None and mse > 0 else None,
        "background_exact_match": bool(mse == 0) if mse is not None else False,
        "background_ssim": float(ssim_map[outside].mean()) if outside.any() else None,
        "boundary_mae_vs_original": float(np.abs(a - b)[band].mean()) if band.any() else None,
    }, band


def clip_scores(model, processor, samples, generated_dir, device):
    rows = {}
    for index, sample in enumerate(samples):
        sid = sample["id"]
        with Image.open(generated_dir / f"{sid}.png") as im:
            image = im.convert("RGB")
        _, mask = load_pair(sample)
        m = binary_mask(mask.resize(image.size, Image.Resampling.NEAREST))
        local = Image.fromarray(np.asarray(image) * m[..., None])
        inputs = processor(text=[sample["prompt"], sample["prompt"]], images=[image, local], return_tensors="pt", padding=True, truncation=True)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.inference_mode():
            imf = F.normalize(model.get_image_features(pixel_values=inputs["pixel_values"]).float(), dim=-1)
            txt = F.normalize(model.get_text_features(input_ids=inputs["input_ids"], attention_mask=inputs.get("attention_mask")).float(), dim=-1)
            scores = (imf * txt).sum(-1).cpu().tolist()
        rows[sid] = {"id": sid, "global_clip_score": scores[0] * 100, "local_clip_score": scores[1] * 100}
        if index % 50 == 0 or index == len(samples) - 1:
            print(f"CLIP {index+1}/{len(samples)}", flush=True)
    return rows


def distribution_metrics(samples, generated_dir, resize, fwd, fid, device):
    values, errors = {}, {}
    if not fwd and not fid:
        return values, errors
    if len(samples) < 2:
        raise ValueError("Distribution metrics require at least two samples")
    with tempfile.TemporaryDirectory(prefix="metric-inputs-", dir=str(ROOT / "cache")) as temporary:
        real, fake = Path(temporary) / "real", Path(temporary) / "fake"
        real.mkdir()
        fake.mkdir()
        for sample in samples:
            original, _ = load_pair(sample)
            original.save(real / f"{sample['id']}.png")
            os.link(generated_dir / f"{sample['id']}.png", fake / f"{sample['id']}.png")
        if fwd:
            dtype = torch.get_default_dtype()
            try:
                import pytorchfwd.fwd as backend
                backend.NUM_PROCESSES = 0
                torch.set_default_dtype(torch.float64)
                with torch.inference_mode():
                    value = backend.compute_fwd(paths=[str(real), str(fake)], wavelet="haar", max_level=4, log_scale=False, batch_size=32, resize=resize)
                if not np.isfinite(value):
                    raise ValueError("Non-finite FWD")
                values["frechet_wavelet_distance"] = {"value": float(value), "num_images": len(samples), "backend": "pytorchfwd", "wavelet": "haar", "max_level": 4, "log_scale": False, "resize": resize}
            except Exception as exc:
                errors["frechet_wavelet_distance"] = f"{type(exc).__name__}: {exc}"
            finally:
                torch.set_default_dtype(dtype)
                gc.collect()
                torch.cuda.empty_cache()
        if fid:
            try:
                from pytorch_fid.fid_score import calculate_fid_given_paths
                value = calculate_fid_given_paths([str(real), str(fake)], batch_size=16, device=device, dims=2048, num_workers=0)
                if not np.isfinite(value):
                    raise ValueError("Non-finite FID")
                values["fid"] = {"value": float(value), "num_images": len(samples), "backend": "pytorch-fid", "dims": 2048}
            except Exception as exc:
                errors["fid"] = f"{type(exc).__name__}: {exc}"
    return values, errors


def main(args):
    torch.set_num_threads(4)
    torch.set_default_dtype(torch.float32)
    torch.backends.cuda.matmul.allow_tf32 = False
    dataset = args.dataset.resolve()
    samples = load_samples(dataset)
    expected = len(samples)
    generated_dir = args.run_dir / "generated"
    missing = [s["id"] for s in samples if not (generated_dir / f"{s['id']}.png").is_file()]
    if missing and not args.allow_partial:
        raise ValueError(f"Missing {len(missing)}/{expected} outputs; refuse silently reduced evaluation")
    samples = [s for s in samples if s["id"] not in set(missing)]
    if not samples:
        raise ValueError("No generated images to evaluate")
    config = json.loads((args.run_dir / "config.json").read_text())
    manifest = json.loads((ROOT / "dataset_manifest.json").read_text())
    if manifest["manifest_sha256"] != config["manifest_sha256"] or sha256(dataset / "prompts.json") != manifest["prompts_sha256"]:
        raise ValueError("Dataset manifest no longer matches inference")
    records = {r["id"]: r for r in manifest["samples"]}
    outputs = {}
    for sample in samples:
        sid = sample["id"]
        for key in ("image", "mask"):
            if sha256(sample[key]) != records[sid][key + "_sha256"]:
                raise ValueError(f"Input changed: {sid}")
        output = generated_dir / f"{sid}.png"
        record = json.loads((args.run_dir / "samples" / f"{sid}.json").read_text())
        outputs[sid] = sha256(output)
        if record.get("status") != "success" or record["output_sha256"] != outputs[sid]:
            raise ValueError(f"Unverified generated image: {sid}")
        with Image.open(output) as im:
            im.verify()
    sources = json.loads((ROOT / "models" / "sources.json").read_text())
    started = time.perf_counter()
    print(f"Evaluating {len(samples)}/{expected}; CLIP ViT-L/14 fp32", flush=True)
    model = CLIPModel.from_pretrained(ROOT / "models" / "clip", use_safetensors=True, local_files_only=True).to(args.device).eval().requires_grad_(False)
    processor = CLIPProcessor.from_pretrained(ROOT / "models" / "clip", local_files_only=True)
    rows = clip_scores(model, processor, samples, generated_dir, args.device)
    del model, processor
    gc.collect()
    torch.cuda.empty_cache()
    perceptual = load_lpips(args.device)
    for index, sample in enumerate(samples):
        sid = sample["id"]
        with Image.open(generated_dir / f"{sid}.png") as im:
            generated = im.convert("RGB")
        original, mask = load_pair(sample)
        original = original.resize(generated.size, Image.Resampling.BILINEAR)
        m = binary_mask(mask.resize(generated.size, Image.Resampling.NEAREST))
        pixel, band = pixel_metrics(original, generated, m, args.band_width)
        a, b = to_tensor(generated, args.device), to_tensor(original, args.device)
        with torch.inference_mode():
            perceptual.spatial = False
            full = float(perceptual(a, b).item())
            perceptual.spatial = True
            distance = perceptual(a, b).squeeze().cpu().numpy()
        if distance.shape != m.shape:
            raise ValueError("LPIPS spatial map must match image dimensions")
        rows[sid].update(pixel)
        rows[sid].update(lpips=full, lpips_outside=float(distance[~m].mean()) if (~m).any() else None, lpips_inside_vs_original=float(distance[m].mean()) if m.any() else None, boundary_lpips_vs_original=float(distance[band].mean()) if band.any() else None)
        if index % 50 == 0 or index == len(samples) - 1:
            print(f"LPIPS/pixel {index+1}/{len(samples)}", flush=True)
    del perceptual
    gc.collect()
    torch.cuda.empty_cache()
    keys = [k for k in next(iter(rows.values())) if k not in ("id", "background_exact_match")]
    metrics = {key: metric_stats([row[key] for row in rows.values()]) for key in keys}
    report = {
        "evaluated_at": now(), "status": "partial" if missing else "complete", "expected": expected, "evaluated": len(samples), "missing_ids": missing,
        "sample_ids": [s["id"] for s in samples], "sample_set_sha256": json_hash([s["id"] for s in samples]),
        "outputs_sha256": json_hash(outputs), "dataset_manifest_sha256": manifest["manifest_sha256"],
        "clip": sources["clip"], "versions": runtime_versions(), "evaluation_script_sha256": sha256(__file__),
        "definitions": {
            "clip": "Normalized image/text feature dot product times 100; no clamping; CLIPModel fp32.",
            "local_clip": "Generated image with pixels outside mask set to black, NOT bounding-box crop.",
            "lpips": "AlexNet LPIPS v0.1 full-image scalar; generated vs ORIGINAL resized BILINEAR.",
            "lpips_regions": "Average of spatial=True distance map in pixel masks. Feature receptive fields cross region boundaries.",
            "background_ssim": "Mean of skimage SSIM map over outside pixels; windows may cross the boundary.",
            "boundary": f"Two-sided dilation minus erosion band, width={args.band_width} output pixels. Error vs ORIGINAL, not a seam realism or editing-success metric.",
            "reference": "Original RGB image, not edited ground truth. LPIPS/FID/FWD cannot independently establish semantic edit success.",
            "psnr": "Null when no outside pixels or MSE=0; exact-match count separately provided.",
        },
        "background_exact_matches": sum(r["background_exact_match"] for r in rows.values()), "metrics": metrics,
    }
    output = args.run_dir / args.output_name
    atomic_json(output.with_name(output.stem + "_per_sample.json"), list(rows.values()))
    report["evaluation_seconds"] = time.perf_counter() - started
    final_status = report["status"]
    report["status"] = "running_distribution_metrics"
    atomic_json(output, report)
    distributions, errors = distribution_metrics(samples, generated_dir, config["parameters"]["output_size"], not args.skip_fwd, args.fid, args.device)
    report["metrics"].update(distributions)
    report["errors"] = errors
    report["evaluation_seconds"] = time.perf_counter() - started
    report["status"] = "failed_metrics" if errors else final_status
    atomic_json(output, report)
    text = [f"Status: {report['status']}", f"Samples: {len(samples)}/{expected}", "Reference: original image, not edited ground truth", ""]
    text.extend(f"{key}: {value['value']} (n={value['num_images']})" for key, value in report["metrics"].items())
    if errors:
        text.append("Errors: " + json.dumps(errors))
    output.with_suffix(".txt").write_text("\n".join(text) + "\n")
    print("\n".join(text), flush=True)
    if errors:
        raise RuntimeError("Some requested metrics failed; see evaluation report")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", type=Path, default=DEFAULT_DATA)
    p.add_argument("--run-dir", type=Path, default=ROOT / "runs" / "brushnet_sd15_random_s42")
    p.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    p.add_argument("--allow-partial", action="store_true")
    p.add_argument("--skip-fwd", action="store_true")
    p.add_argument("--fid", action="store_true", help="Optional legacy FID; disabled in latest aligned protocol")
    p.add_argument("--band-width", type=int, default=3)
    p.add_argument("--output-name", default="eval_results.json")
    args = p.parse_args()
    reserved = {"config.json", "scheduler_config.json", "generation_summary.json", "job_status.json", "progress.json"}
    if Path(args.output_name).name != args.output_name or not args.output_name.endswith(".json") or args.output_name in reserved or args.band_width < 1:
        p.error("Invalid/reserved report filename or band width")
    with exclusive_lock(args.run_dir / ".inference.lock"), exclusive_lock(args.run_dir / ".evaluation.lock"):
        main(args)
