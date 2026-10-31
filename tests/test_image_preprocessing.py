import numpy as np
from dataclasses import asdict
import json
from PIL import Image
import pytest
import torch

from awbc_data import ImagePreprocessConfig, ImagePreprocessor
from awbc_data.images import IMAGENET_MEAN, IMAGENET_STD


@pytest.mark.parametrize("mode", ["resize", "letterbox"])
def test_rgb_normalization_and_inference_parity(mode):
    rgb = np.zeros((20, 40, 3), dtype=np.uint8)
    rgb[:] = [255, 100, 20]
    config = ImagePreprocessConfig(size=(32, 32), mode=mode, padding_rgb=(10, 20, 30))
    preprocess = ImagePreprocessor(config)
    # Dataset passes PIL; inference typically passes a uint8 RGB array.
    actual = preprocess(Image.fromarray(rgb))
    torch.testing.assert_close(actual, preprocess(rgb), rtol=0, atol=0)
    assert actual.shape == (3, 32, 32) and actual.dtype == torch.float32
    restored = actual * IMAGENET_STD + IMAGENET_MEAN
    torch.testing.assert_close(restored[:, 16, 16], torch.tensor([255, 100, 20])/255)
    assert torch.isfinite(actual).all()
    if mode == "letterbox":
        torch.testing.assert_close(restored[:, 0, 0], torch.tensor([10, 20, 30])/255)
        torch.testing.assert_close(restored[:, 8, 0], torch.tensor([255, 100, 20])/255)


def test_letterbox_preserves_entire_image_without_cropping():
    rgb = np.zeros((20, 40, 3), dtype=np.uint8)
    rgb[:, :10, 0] = 255
    rgb[:, -10:, 2] = 255
    preprocess = ImagePreprocessor(ImagePreprocessConfig(size=(40, 40), mode="letterbox"))
    restored = preprocess(rgb) * IMAGENET_STD + IMAGENET_MEAN
    assert restored[0, 20, 0] > 0.99
    assert restored[2, 20, -1] > 0.99


def test_saved_config_reproduces_inference_preprocessing():
    config = ImagePreprocessConfig(size=(32, 48), mode="letterbox", padding_rgb=(10, 20, 30))
    loaded = ImagePreprocessConfig(**json.loads(json.dumps(asdict(config))))
    rgb = np.random.default_rng(0).integers(0, 256, (20, 40, 3), dtype=np.uint8)
    torch.testing.assert_close(ImagePreprocessor(config)(rgb), ImagePreprocessor(loaded)(rgb), rtol=0, atol=0)
