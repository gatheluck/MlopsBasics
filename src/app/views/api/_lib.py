import logging
import pathlib
from io import BytesIO
from typing import Final, List

import albumentations as albu
import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)


def check_suffix(filepath: pathlib.Path) -> None:
    supported_suffixes: Final[List[str]] = [".png", ".jpg", ".jpeg"]

    suffix: Final = filepath.suffix
    if suffix not in supported_suffixes:
        logger.error(f"input {str(filepath)} is not acceptable suffix.")
        raise ValueError


def read_imagefile(file) -> Image.Image:
    image = Image.open(BytesIO(file))
    return image.convert("RGB")


def apply_transform(
    image: np.ndarray,
    transform: albu.Compose,
    add_batch_dim: bool = True,
) -> np.ndarray:
    num_channel: Final = len(image.shape)
    if num_channel != 3:
        logger.error(f"input image should have 3 channels instead of {num_channel}")
        raise ValueError

    transformed: Final[np.ndarray] = transform(image=image)["image"]  # (h, w, c)
    transformed_chw: Final = transformed.transpose(2, 0, 1)
    if not add_batch_dim:
        return transformed_chw
    return np.expand_dims(transformed_chw, 0)
