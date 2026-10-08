import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
from PIL import Image

from common import ROOT, atomic_json, binary_mask, sha256
from infer import load_pipeline, preprocess, verified_output
from model_spec import BASE_REPOSITORY, BASE_REVISION, ORIGINAL_REPOSITORY, WEIGHTS_SHA256, validate_architecture


class SD20Tests(unittest.TestCase):
    def test_original_checkpoint_identity(self):
        sources = json.loads((ROOT / "models" / "sources.json").read_text())
        self.assertEqual(set(sources), {"base", "clip", "lpips_backbone"})
        self.assertEqual(sources["base"]["repo_id"], BASE_REPOSITORY)
        self.assertEqual(sources["base"]["revision"], BASE_REVISION)
        config = json.loads((ROOT / "models" / "base" / "unet" / "config.json").read_text())
        self.assertEqual(config["in_channels"], 9)
        self.assertEqual(config["cross_attention_dim"], 1024)
        self.assertEqual(sources["base"]["original_repo_id"], ORIGINAL_REPOSITORY)
        for name, digest in WEIGHTS_SHA256.items():
            self.assertEqual(sources["base"]["files_sha256"][name], digest)
            self.assertEqual(sha256(ROOT / "models" / "base" / name), digest)

    def test_architecture_rejects_sd15_and_v_prediction(self):
        unet = {"in_channels": 9, "out_channels": 4, "cross_attention_dim": 1024, "use_linear_projection": True}
        text = {"hidden_size": 1024, "num_hidden_layers": 23}
        index = {"_class_name": "StableDiffusionInpaintPipeline"}
        validate_architecture(unet, text, {}, index)
        with self.assertRaises(ValueError):
            validate_architecture(dict(unet, cross_attention_dim=768), text, {}, index)
        with self.assertRaises(ValueError):
            validate_architecture(unet, text, {"prediction_type": "v_prediction"}, index)
        with self.assertRaises(ValueError):
            validate_architecture(unet, text, {}, {"_class_name": "CustomPipeline"})

    def test_preprocess_preserves_rgb_and_white_means_edit(self):
        with tempfile.TemporaryDirectory(dir=ROOT / "cache") as directory:
            root = Path(directory)
            Image.new("RGB", (224, 224), (31, 63, 127)).save(root / "image.png")
            values = np.zeros((256, 256), dtype=np.uint8)
            values[:, 128:] = 255
            Image.fromarray(values).save(root / "mask.png")
            sample = {"id": "test", "image": str(root / "image.png"), "mask": str(root / "mask.png")}
            image, mask = preprocess(sample, 512)
            self.assertEqual(image.size, (512, 512))
            self.assertEqual(image.getpixel((400, 128)), (31, 63, 127))
            self.assertEqual(set(np.unique(mask)), {0, 255})
            self.assertFalse(binary_mask(mask)[:, :256].any())
            self.assertTrue(binary_mask(mask)[:, 256:].all())
            threshold = Image.fromarray(np.array([[0, 127, 128, 255]], dtype=np.uint8))
            self.assertEqual(binary_mask(threshold).tolist(), [[False, False, True, True]])

    def test_resume_requires_output_and_configuration_integrity(self):
        with tempfile.TemporaryDirectory(dir=ROOT / "cache") as directory:
            root = Path(directory)
            output, record = root / "image.png", root / "record.json"
            Image.new("RGB", (64, 64), "green").save(output)
            atomic_json(record, {"status": "success", "config_sha256": "config-a", "output_sha256": sha256(output)})
            self.assertIsNotNone(verified_output(output, record, "config-a"))
            self.assertIsNone(verified_output(output, record, "config-b"))
            Image.new("RGB", (64, 64), "blue").save(output)
            self.assertIsNone(verified_output(output, record, "config-a"))

    def test_loader_refuses_four_channel_base(self):
        model = MagicMock()
        model.unet.config.in_channels = 4
        with patch("infer.StableDiffusionInpaintPipeline.from_pretrained", return_value=model) as loader:
            with self.assertRaises(ValueError):
                load_pipeline("unipc")
            options = loader.call_args.kwargs
            self.assertTrue(options["use_safetensors"])
            self.assertTrue(options["local_files_only"])
            self.assertEqual(options["variant"], "fp16")
            model.to.assert_not_called()


if __name__ == "__main__":
    unittest.main()
