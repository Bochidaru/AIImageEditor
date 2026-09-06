from __future__ import annotations

import numpy as np

from ..masks.operations import feather, validate_mask
from ..types import ImageArray, MaskArray


class Compositor:
    def compose(
        self,
        original: ImageArray,
        generated: ImageArray,
        mask: MaskArray,
        *,
        feather_radius: int = 0,
    ) -> ImageArray:
        if original.shape != generated.shape:
            raise ValueError("Original and generated images must have equal shapes.")
        validate_mask(mask, original.shape)

        alpha = feather(mask, feather_radius)[..., None]
        result = (
            generated.astype(np.float32) * alpha
            + original.astype(np.float32) * (1.0 - alpha)
        )
        return np.clip(result, 0, 255).astype(np.uint8)

