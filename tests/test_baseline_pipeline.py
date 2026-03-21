import tempfile
import unittest
from pathlib import Path

import torch
from PIL import Image

from preprocess.image_preprocessor import ImagePreprocessor
from utils.inference_service import InferenceService


class StubModelWrapper:
    id2label = {
        0: "real",
        1: "fake",
    }

    def __init__(self, logits):
        self.logits = logits

    def predict(self, image_tensor):
        self.last_shape = tuple(image_tensor.shape)
        return self.logits


class BaselinePipelineTests(unittest.TestCase):
    def test_image_preprocessor_loads_rgb_image(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            image_path = Path(tmp_dir) / "sample.png"
            Image.new("L", (32, 32), color=128).save(image_path)

            image = ImagePreprocessor().load_image(str(image_path))

        self.assertEqual(image.mode, "RGB")
        self.assertEqual(image.size, (32, 32))

    def test_inference_service_transforms_pil_image(self):
        model = StubModelWrapper(torch.tensor([[0.2, 1.8]]))
        service = InferenceService(model)
        image = Image.new("RGB", (80, 120), color=(64, 128, 192))

        result = service.predict(image)

        self.assertEqual(model.last_shape, (1, 3, 224, 224))
        self.assertEqual(result["prediction"], "fake")
        self.assertAlmostEqual(
            result["probabilities"]["real"] + result["probabilities"]["fake"],
            1.0,
            places=6,
        )


if __name__ == "__main__":
    unittest.main()
