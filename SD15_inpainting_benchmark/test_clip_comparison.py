import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
from PIL import Image

from audit_clip_comparison import validated_rows
from common import ROOT
from evaluate import clip_scores, metric_stats


class ClipComparisonTests(unittest.TestCase):
    def setUp(self):
        (ROOT / "cache").mkdir(exist_ok=True)
        self.temporary = tempfile.TemporaryDirectory(dir=ROOT / "cache", prefix="clip-test-")
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)

    def write_rows(self, rows):
        path = self.root / "rows.json"
        path.write_text(json.dumps(rows))
        return path

    def test_duplicate_ids_rejected(self):
        row = {"id": "a", "global_clip_score": 1.0, "local_clip_score": 2.0}
        with self.assertRaises(ValueError):
            validated_rows(self.write_rows([row, row]), ["a", "b"])

    def test_nonfinite_old_scores_rejected(self):
        for invalid in (float("nan"), float("inf"), float("-inf"), True, "1.0"):
            rows = [
                {"id": "a", "global_clip_score": 1.0, "local_clip_score": 2.0},
                {"id": "b", "global_clip_score": invalid, "local_clip_score": 2.0},
            ]
            with self.subTest(value=invalid), self.assertRaises(ValueError):
                validated_rows(self.write_rows(rows), ["a", "b"])

    def test_complete_finite_rows_accepted(self):
        rows = [{"id": "a", "global_clip_score": -1.0, "local_clip_score": 2.0}]
        self.assertEqual(validated_rows(self.write_rows(rows), ["a"])["a"], rows[0])

    def test_population_statistics(self):
        result = metric_stats([1.0, 3.0])
        self.assertEqual(result, {"value": 2.0, "std": 1.0, "num_images": 2})

    def test_local_mask_formula_and_unclamped_score(self):
        image = Image.fromarray(np.full((2, 2, 3), 200, dtype=np.uint8))
        mask = Image.fromarray(np.asarray([[127, 128], [0, 255]], dtype=np.uint8))
        image.save(self.root / "a.png")
        captured = {}

        def processor(**kwargs):
            captured.update(kwargs)
            return {
                "pixel_values": torch.zeros((2, 3, 2, 2)),
                "input_ids": torch.ones((2, 2), dtype=torch.long),
                "attention_mask": torch.ones((2, 2), dtype=torch.long),
            }

        class Model:
            def get_image_features(self, **kwargs):
                return torch.tensor([[3.0, 4.0], [-3.0, -4.0]])

            def get_text_features(self, **kwargs):
                return torch.tensor([[3.0, 4.0], [3.0, 4.0]])

        with patch("evaluate.load_pair", return_value=(image, mask)):
            rows = clip_scores(Model(), processor, [{"id": "a", "prompt": "test prompt"}], self.root, "cpu")
        expected = np.asarray(image) * (np.asarray(mask) > 127)[..., None]
        np.testing.assert_array_equal(np.asarray(captured["images"][1]), expected)
        self.assertEqual(captured["text"], ["test prompt", "test prompt"])
        self.assertTrue(captured["truncation"])
        self.assertAlmostEqual(rows["a"]["global_clip_score"], 100.0, places=4)
        self.assertAlmostEqual(rows["a"]["local_clip_score"], -100.0, places=4)


if __name__ == "__main__":
    unittest.main()
