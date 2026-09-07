from __future__ import annotations

import numpy as np

from ..types import ImageArray, ImageTransform, MaskArray, PreparedImage
from .operations import threshold, validate_mask


def prepare_full_image(
    image: ImageArray,
    mask: MaskArray,
    *,
    size_multiple: int = 16,
) -> PreparedImage:
    """Pad a full image and its mask to dimensions accepted by the editor."""
    validate_mask(mask, image.shape)
    if size_multiple <= 0:
        raise ValueError("size_multiple must be greater than zero.")

    binary = threshold(mask)
    height, width = image.shape[:2]
    pad_bottom = (-height) % size_multiple
    pad_right = (-width) % size_multiple

    image_prepared = np.pad(
        image,
        ((0, pad_bottom), (0, pad_right), (0, 0)),
        mode="reflect",
    )
    mask_prepared = np.pad(
        binary,
        ((0, pad_bottom), (0, pad_right)),
        mode="constant",
    )

    transform = ImageTransform(
        image_size=(height, width),
        padding=(0, pad_bottom, 0, pad_right),
    )
    return PreparedImage(
        image=np.ascontiguousarray(image_prepared),
        mask=np.ascontiguousarray(threshold(mask_prepared)),
        transform=transform,
    )


def restore_full_image(
    edited_image: ImageArray,
    transform: ImageTransform,
) -> ImageArray:
    """Remove padding added by prepare_full_image."""
    height, width = transform.image_size
    if edited_image.shape[0] < height or edited_image.shape[1] < width:
        raise ValueError("Edited image is smaller than the unpadded image size.")
    return np.ascontiguousarray(edited_image[:height, :width])
