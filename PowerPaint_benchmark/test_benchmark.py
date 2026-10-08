import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
from PIL import Image
from transformers import CLIPTextConfig, CLIPTextModel

from common import binary_mask, exclusive_lock, safe_file
from evaluate import metric_stats, pixel_metrics
from gpu_guard import gpu_state, wait_for_memory
from powerpaint_adapter import QUALITY_NEGATIVE, check_prompt, load_text_weights, prompts_for, vendor_modules


class BenchmarkTests(unittest.TestCase):
    def test_mask_threshold(self):
        mask = Image.fromarray(np.array([[0, 127, 128, 255]], dtype=np.uint8))
        self.assertEqual(binary_mask(mask).tolist(), [[False, False, True, True]])

    def test_safe_paths(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "valid.png").touch()
            self.assertEqual(safe_file(root, "valid.png"), root / "valid.png")
            for value in ("../outside.png", "/etc/passwd", "a\\b.png", "missing.png"):
                with self.assertRaises(ValueError):
                    safe_file(root, value)

    def test_duplicate_run_lock(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "lock"
            with exclusive_lock(path):
                with self.assertRaises(RuntimeError):
                    with exclusive_lock(path):
                        pass

    def test_prompts(self):
        positive, negative = prompts_for("a satellite image of farmland")
        self.assertEqual(positive, "a satellite image of farmland P_obj")
        self.assertEqual(negative, QUALITY_NEGATIVE + "P_obj")
        with self.assertRaises(ValueError):
            prompts_for("P_ctxt")

    def test_no_task_token_truncation(self):
        class Tokenizer:
            model_max_length = 77

            def __call__(self, text, truncation):
                return {"input_ids": list(range(int(text)))}

        self.assertEqual(check_prompt(Tokenizer(), "77"), 77)
        with self.assertRaises(ValueError):
            check_prompt(Tokenizer(), "78")

    def test_legacy_text_weights_remain_strict(self):
        config = CLIPTextConfig(vocab_size=20, hidden_size=8, intermediate_size=16, num_hidden_layers=1, num_attention_heads=2, max_position_embeddings=8)
        model = CLIPTextModel(config)
        position_key = "text_model.embeddings.position_ids"
        state = dict(model.state_dict())
        state[position_key] = model.text_model.embeddings.position_ids.clone()
        with patch("powerpaint_adapter.load_file", return_value=dict(state)):
            load_text_weights(model, "unused.safetensors")
        bad = dict(state)
        bad[position_key] = torch.zeros_like(state[position_key])
        with patch("powerpaint_adapter.load_file", return_value=bad):
            with self.assertRaises(ValueError):
                load_text_weights(model, "unused.safetensors")
        del state["text_model.embeddings.token_embedding.weight"]
        with patch("powerpaint_adapter.load_file", return_value=state):
            with self.assertRaises(RuntimeError):
                load_text_weights(model, "unused.safetensors")

    def test_official_masking(self):
        pipeline, _ = vendor_modules()
        image = Image.fromarray(np.full((16, 16, 3), 255, dtype=np.uint8))
        arr = np.zeros((16, 16), dtype=np.uint8)
        arr[4:12, 4:12] = 255
        mask, masked = pipeline.prepare_mask_and_masked_image(image, Image.fromarray(arr), 16, 16)
        self.assertEqual(float(mask[0, 0, 6, 6]), 1.0)
        self.assertEqual(float(masked[0, 0, 6, 6]), 0.0)
        self.assertEqual(float(masked[0, 0, 0, 0]), 1.0)

    def test_pixel_metrics_and_null_psnr(self):
        arr = np.full((16, 16, 3), 128, dtype=np.uint8)
        mask = np.zeros((16, 16), dtype=bool)
        mask[4:12, 4:12] = True
        metrics, band = pixel_metrics(Image.fromarray(arr), Image.fromarray(arr), mask, 3)
        self.assertEqual(metrics["background_mae"], 0)
        self.assertIsNone(metrics["background_psnr"])
        self.assertTrue(metrics["background_exact_match"])
        self.assertAlmostEqual(metrics["background_ssim"], 1.0)
        self.assertTrue(band.any())
        self.assertEqual(metric_stats([None, 2, 4])["value"], 3.0)
        with self.assertRaises(ValueError):
            metric_stats([float("nan")])

    def test_memory_guard(self):
        with patch("gpu_guard.gpu_state", return_value={"free_mib": 9000}):
            self.assertEqual(wait_for_memory(8192)["free_mib"], 9000)
        with patch("gpu_guard.gpu_state", return_value={"free_mib": 100}), patch("gpu_guard.time.monotonic", side_effect=[0, 2]):
            with self.assertRaises(TimeoutError):
                wait_for_memory(8192, timeout=1)
        with patch("gpu_guard.subprocess.run") as run:
            run.return_value.stdout = "0, Tesla T4, 15360, 4703, 10227, 99\n"
            self.assertEqual(gpu_state()["free_mib"], 10227)


if __name__ == "__main__":
    unittest.main()
