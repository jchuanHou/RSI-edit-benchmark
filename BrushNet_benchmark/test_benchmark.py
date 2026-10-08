import json
import tempfile
import unittest
import zipfile
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from common import atomic_json, binary_mask, json_hash, load_pair, load_samples, safe_file
from evaluate import load_lpips, metric_stats, pixel_metrics
from prepare_dataset import extract_checked


class BenchmarkTests(unittest.TestCase):
    def test_archive_path_traversal_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with zipfile.ZipFile(root / "bad.zip", "w") as z:
                z.writestr("../escaped.txt", "bad")
            with self.assertRaises(ValueError):
                extract_checked(root / "bad.zip", root / "out")
            self.assertFalse((root / "escaped.txt").exists())
            self.assertFalse((root / "out").exists())

    def test_dataset_pairing_and_scaled_masks(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "reference_images").mkdir()
            (root / "masks").mkdir()
            Image.new("RGB", (224, 224)).save(root / "reference_images" / "00001.jpg")
            Image.new("L", (256, 256), 255).save(root / "masks" / "00001.png")
            atomic_json(root / "prompts.json", {"data": {"00001": {"image_file": "00001.jpg", "mask_file": "00001.png", "prompt": "A green park"}}})
            samples = load_samples(root)
            self.assertEqual(samples[0]["id"], "00001")
            image, mask = load_pair(samples[0])
            self.assertEqual(image.size, (224, 224))
            self.assertEqual(mask.size, (256, 256))
            self.assertTrue(binary_mask(mask).all())
            with self.assertRaises(ValueError):
                safe_file(root / "masks", "../prompts.json")

    def test_stats_and_json(self):
        stats = metric_stats([1, 3, None])
        self.assertEqual(stats["value"], 2)
        self.assertEqual(stats["num_images"], 2)
        with self.assertRaises(ValueError):
            metric_stats([float("nan")])
        self.assertEqual(json_hash({"a": 1, "b": 2}), json_hash({"b": 2, "a": 1}))

    def test_pixel_metrics_identity(self):
        image = Image.fromarray(np.random.default_rng(0).integers(0, 255, (32, 32, 3), dtype=np.uint8))
        m = np.zeros((32, 32), dtype=bool)
        m[8:24, 8:24] = True
        metrics, band = pixel_metrics(image, image, m, 3)
        self.assertTrue(metrics["background_exact_match"])
        self.assertIsNone(metrics["background_psnr"])
        self.assertEqual(metrics["background_mae"], 0)
        self.assertAlmostEqual(metrics["background_ssim"], 1)
        self.assertTrue(band[m].any() and band[~m].any())

    def test_lpips_identity_and_spatial_shape(self):
        model = load_lpips("cpu")
        x = torch.rand(1, 3, 64, 64) * 2 - 1
        with torch.inference_mode():
            self.assertAlmostEqual(float(model(x, x)), 0, places=6)
            model.spatial = True
            distance = model(x, x)
        self.assertEqual(tuple(distance.shape), (1, 1, 64, 64))
        self.assertLess(float(distance.abs().max()), 1e-6)


if __name__ == "__main__":
    torch.set_num_threads(2)
    unittest.main()
