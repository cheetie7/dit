"""One RGB preprocessing implementation for Dataset and inference."""
from dataclasses import dataclass
from typing import Literal

import numpy as np
from PIL import Image, ImageOps
import torch

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406])[:, None, None]
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225])[:, None, None]


@dataclass
class ImagePreprocessConfig:
    size: tuple[int, int] = (240, 320)  # H, W
    mode: Literal["resize", "letterbox"] = "resize"
    padding_rgb: tuple[int, int, int] = (0, 0, 0)

    def __post_init__(self):
        # JSON stores tuples as lists; restore the types expected by Pillow.
        self.size = tuple(self.size)
        self.padding_rgb = tuple(self.padding_rgb)


class ImagePreprocessor:
    """PIL RGB or uint8 HWC RGB -> normalized float32 CHW, without cropping."""

    def __init__(self, config: ImagePreprocessConfig | None = None):
        self.config = config or ImagePreprocessConfig()

    def __call__(self, image):
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        image = image.convert("RGB")
        h, w = self.config.size
        if self.config.mode == "resize":
            image = image.resize((w, h), Image.Resampling.BILINEAR)
        elif self.config.mode == "letterbox":
            content = ImageOps.contain(image, (w, h), Image.Resampling.BILINEAR)
            image = Image.new("RGB", (w, h), self.config.padding_rgb)
            image.paste(content, ((w - content.width) // 2, (h - content.height) // 2))
        else:
            raise ValueError(f"Unknown resize mode: {self.config.mode}")
        rgb = torch.from_numpy(np.array(image, copy=True)).permute(2, 0, 1).float() / 255
        return (rgb - IMAGENET_MEAN) / IMAGENET_STD
