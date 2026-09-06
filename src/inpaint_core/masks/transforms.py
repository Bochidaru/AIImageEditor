from __future__ import annotations

import cv2
import numpy as np

from ..types import CropTransform, ImageArray, MaskArray, PreparedCrop
from .operations import threshold, validate_mask


def prepare_masked_crop(
    image: ImageArray,
    mask: MaskArray,
    *,
    padding: int = 128,
    max_side: int = 1024,
    size_multiple: int = 16,
) -> PreparedCrop:
    validate_mask(mask, image.shape)
    binary = threshold(mask)
    ys, xs = np.nonzero(binary)
    height, width = image.shape[:2]

    x1 = max(int(xs.min()) - padding, 0)
    y1 = max(int(ys.min()) - padding, 0)
    x2 = min(int(xs.max()) + padding + 1, width)
    y2 = min(int(ys.max()) + padding + 1, height)

    image_crop = image[y1:y2, x1:x2]
    mask_crop = binary[y1:y2, x1:x2]
    crop_height, crop_width = image_crop.shape[:2]

    scale = min(1.0, max_side / max(crop_height, crop_width))
    process_height = max(1, round(crop_height * scale))
    process_width = max(1, round(crop_width * scale))

    if (process_height, process_width) != (crop_height, crop_width):
        image_crop = cv2.resize(
            image_crop,
            (process_width, process_height),
            interpolation=cv2.INTER_AREA,
        )
        mask_crop = cv2.resize(
            mask_crop,
            (process_width, process_height),
            interpolation=cv2.INTER_NEAREST,
        )

    pad_bottom = (-process_height) % size_multiple
    pad_right = (-process_width) % size_multiple
    image_prepared = np.pad(
        image_crop,
        ((0, pad_bottom), (0, pad_right), (0, 0)),
        mode="reflect",
    )
    mask_prepared = np.pad(
        mask_crop,
        ((0, pad_bottom), (0, pad_right)),
        mode="constant",
    )

    transform = CropTransform(
        crop_box=(x1, y1, x2, y2),
        crop_size=(crop_height, crop_width),
        processing_size=(process_height, process_width),
        padding=(0, pad_bottom, 0, pad_right),
        original_size=(height, width),
    )
    return PreparedCrop(image_prepared, threshold(mask_prepared), transform)


def restore_crop(
    original: ImageArray,
    edited_crop: ImageArray,
    transform: CropTransform,
) -> ImageArray:
    x1, y1, x2, y2 = transform.crop_box
    process_height, process_width = transform.processing_size
    crop_height, crop_width = transform.crop_size

    unpadded = edited_crop[:process_height, :process_width]
    if unpadded.shape[:2] != (crop_height, crop_width):
        unpadded = cv2.resize(
            unpadded,
            (crop_width, crop_height),
            interpolation=cv2.INTER_LINEAR,
        )

    restored = original.copy()
    restored[y1:y2, x1:x2] = unpadded
    return restored

