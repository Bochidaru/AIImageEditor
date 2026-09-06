from __future__ import annotations

import cv2
import numpy as np

from ..types import MaskArray


def validate_mask(mask: MaskArray, image_shape: tuple[int, ...] | None = None) -> None:
    if mask.ndim != 2:
        raise ValueError(f"Mask must have shape (H, W), got {mask.shape}.")
    if image_shape is not None and mask.shape != image_shape[:2]:
        raise ValueError(
            f"Mask shape {mask.shape} does not match image shape {image_shape[:2]}."
        )
    if not np.any(mask):
        raise ValueError("Mask is empty.")


def threshold(mask: MaskArray, value: int = 127) -> MaskArray:
    return np.where(mask > value, 255, 0).astype(np.uint8)


def invert(mask: MaskArray) -> MaskArray:
    return (255 - threshold(mask)).astype(np.uint8)


def dilate(mask: MaskArray, radius: int) -> MaskArray:
    binary = threshold(mask)
    if radius <= 0:
        return binary
    kernel_size = 2 * radius + 1
    kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8)
    return cv2.dilate(binary, kernel, iterations=1)


def erode(mask: MaskArray, radius: int) -> MaskArray:
    binary = threshold(mask)
    if radius <= 0:
        return binary
    kernel_size = 2 * radius + 1
    kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8)
    return cv2.erode(binary, kernel, iterations=1)


def feather(mask: MaskArray, radius: int) -> np.ndarray:
    binary = threshold(mask).astype(np.float32) / 255.0
    if radius <= 0:
        return binary
    kernel_size = 2 * radius + 1
    return cv2.GaussianBlur(binary, (kernel_size, kernel_size), sigmaX=0)

