from __future__ import annotations

from pathlib import Path
from typing import TypeAlias

import numpy as np
from PIL import Image, ImageOps

from .types import ImageArray

ImageSource: TypeAlias = str | Path | Image.Image | np.ndarray


def preprocess_image(
    source: ImageSource,
    *,
    color_space: str = "rgb",
    alpha_background: tuple[int, int, int] = (255, 255, 255),
) -> ImageArray:
    """Convert a path, PIL image, or NumPy array to contiguous RGB uint8."""
    _validate_color_space(color_space)
    _validate_background(alpha_background)

    if isinstance(source, (str, Path)):
        with Image.open(source) as image:
            return _from_pil(image, alpha_background)

    if isinstance(source, Image.Image):
        return _from_pil(source, alpha_background)

    if isinstance(source, np.ndarray):
        return _from_numpy(source, color_space, alpha_background)

    raise TypeError("Image source must be a path, PIL image, or NumPy array.")


def validate_image(image: ImageArray) -> None:
    """Validate the canonical image format used inside the core package."""
    if not isinstance(image, np.ndarray):
        raise TypeError("Image must be a NumPy array.")
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(f"Image must have shape (H, W, 3), got {image.shape}.")
    if image.dtype != np.uint8:
        raise ValueError(f"Image must have uint8 dtype, got {image.dtype}.")
    if image.shape[0] == 0 or image.shape[1] == 0:
        raise ValueError("Image dimensions must be non-zero.")


def _from_pil(
    source: Image.Image,
    alpha_background: tuple[int, int, int],
) -> ImageArray:
    image = ImageOps.exif_transpose(source)
    if image.mode in ("RGBA", "LA") or "transparency" in image.info:
        rgba = image.convert("RGBA")
        background = Image.new("RGBA", rgba.size, (*alpha_background, 255))
        image = Image.alpha_composite(background, rgba).convert("RGB")
    else:
        image = image.convert("RGB")
    return np.ascontiguousarray(np.asarray(image, dtype=np.uint8))


def _from_numpy(
    source: np.ndarray,
    color_space: str,
    alpha_background: tuple[int, int, int],
) -> ImageArray:
    image = np.asarray(source)
    if image.size == 0:
        raise ValueError("Image cannot be empty.")
    if image.ndim == 2:
        image = image[..., None]
    if image.ndim != 3 or image.shape[2] not in (1, 3, 4):
        raise ValueError(
            f"NumPy image must have 1, 3, or 4 channels, got {image.shape}."
        )

    image = _to_uint8(image)
    if image.shape[2] == 1:
        rgb = np.repeat(image, 3, axis=2)
    else:
        rgb = image[..., :3]
        if color_space == "bgr":
            rgb = rgb[..., ::-1]
        if image.shape[2] == 4:
            alpha = image[..., 3:4].astype(np.float32) / 255.0
            background = np.asarray(alpha_background, dtype=np.float32)
            rgb = rgb.astype(np.float32) * alpha + background * (1.0 - alpha)
            rgb = np.clip(rgb, 0, 255).astype(np.uint8)

    result = np.ascontiguousarray(rgb, dtype=np.uint8)
    validate_image(result)
    return result


def _to_uint8(image: np.ndarray) -> np.ndarray:
    if np.issubdtype(image.dtype, np.floating):
        if not np.all(np.isfinite(image)):
            raise ValueError("Floating-point image contains NaN or infinity.")
        if float(image.min()) < 0:
            raise ValueError("Floating-point image values cannot be negative.")
        if float(image.max()) <= 1.0:
            image = image * 255.0
    return np.clip(image, 0, 255).astype(np.uint8)


def _validate_color_space(color_space: str) -> None:
    if color_space not in {"rgb", "bgr"}:
        raise ValueError("color_space must be 'rgb' or 'bgr'.")


def _validate_background(background: tuple[int, int, int]) -> None:
    if len(background) != 3 or any(not 0 <= value <= 255 for value in background):
        raise ValueError("alpha_background must contain three values in [0, 255].")

