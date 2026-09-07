from __future__ import annotations

from pathlib import Path
from typing import TypeAlias

import cv2
import numpy as np
from PIL import Image, ImageOps

from .types import BoxPrompt, ImageArray, PointPrompt, SelectionPrompt

ImageSource: TypeAlias = str | Path | Image.Image | np.ndarray


def preprocess_image_only(
    source: ImageSource,
    *,
    color_space: str = "rgb",
    alpha_background: tuple[int, int, int] = (255, 255, 255),
    max_side: int | None = None,
) -> ImageArray:
    """Normalize an image for modes that do not use a point or box selection."""
    _validate_color_space(color_space)
    _validate_background(alpha_background)
    if max_side is not None and max_side <= 0:
        raise ValueError("max_side must be greater than zero.")
    if isinstance(source, (str, Path)):
        with Image.open(source) as image:
            result = _from_pil(image, alpha_background)
    elif isinstance(source, Image.Image):
        result = _from_pil(source, alpha_background)
    elif isinstance(source, np.ndarray):
        result = _from_numpy(source, color_space, alpha_background)
    else:
        raise TypeError("Image source must be a path, PIL image, or NumPy array.")
    return _resize_to_max_side(result, max_side)


def preprocess_image(
    source: ImageSource,
    selection: SelectionPrompt,
    *,
    color_space: str = "rgb",
    alpha_background: tuple[int, int, int] = (255, 255, 255),
    max_side: int | None = None,
) -> tuple[ImageArray, SelectionPrompt]:
    """Normalize an image and scale its required selection with the image."""
    _validate_color_space(color_space)
    _validate_background(alpha_background)
    if max_side is not None and max_side <= 0:
        raise ValueError("max_side must be greater than zero.")

    result = preprocess_image_only(
        source,
        color_space=color_space,
        alpha_background=alpha_background,
        max_side=None,
    )

    original_size = result.shape[:2]
    result = _resize_to_max_side(result, max_side)
    resized_selection = _resize_selection(
        selection,
        original_size=original_size,
        resized_size=result.shape[:2],
    )
    return result, resized_selection


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


def _resize_to_max_side(image: ImageArray, max_side: int | None) -> ImageArray:
    if max_side is None:
        return image

    height, width = image.shape[:2]
    longest_side = max(height, width)
    if longest_side <= max_side:
        return image

    scale = max_side / longest_side
    resized_height = max(1, round(height * scale))
    resized_width = max(1, round(width * scale))
    resized = cv2.resize(
        image,
        (resized_width, resized_height),
        interpolation=cv2.INTER_AREA,
    )
    return np.ascontiguousarray(resized, dtype=np.uint8)


def _resize_selection(
    selection: SelectionPrompt,
    *,
    original_size: tuple[int, int],
    resized_size: tuple[int, int],
) -> SelectionPrompt:
    original_height, original_width = original_size
    resized_height, resized_width = resized_size
    scale_x = resized_width / original_width
    scale_y = resized_height / original_height

    if isinstance(selection, PointPrompt):
        return PointPrompt(
            points=[(x * scale_x, y * scale_y) for x, y in selection.points],
            labels=selection.labels.copy(),
        )
    if isinstance(selection, BoxPrompt):
        return BoxPrompt(
            x1=selection.x1 * scale_x,
            y1=selection.y1 * scale_y,
            x2=selection.x2 * scale_x,
            y2=selection.y2 * scale_y,
        )
    raise TypeError("selection must be a PointPrompt or BoxPrompt.")


def _validate_color_space(color_space: str) -> None:
    if color_space not in {"rgb", "bgr"}:
        raise ValueError("color_space must be 'rgb' or 'bgr'.")


def _validate_background(background: tuple[int, int, int]) -> None:
    if len(background) != 3 or any(not 0 <= value <= 255 for value in background):
        raise ValueError("alpha_background must contain three values in [0, 255].")
