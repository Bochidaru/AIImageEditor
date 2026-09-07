from __future__ import annotations

import numpy as np

from ..backends.protocols import MaskedEditor
from ..types import EditResult, GenerationOptions, ImageArray, OutpaintMargins
from .common import OperationContext


class OutpaintingOperation:
    def __init__(self, context: OperationContext, backend: MaskedEditor):
        self.context, self.backend = context, backend

    def run(
        self, image: ImageArray, prompt: str, margins: OutpaintMargins, *,
        generation_options: GenerationOptions | None = None,
    ) -> EditResult:
        if not prompt.strip():
            raise ValueError("An outpaint prompt is required.")
        self.context.validate_image(image)
        height, width = image.shape[:2]
        new_height = height + margins.top + margins.bottom
        new_width = width + margins.left + margins.right
        if max(new_height, new_width) > self.context.config.processing.max_image_side:
            raise ValueError("Outpaint canvas exceeds processing.max_image_side.")
        canvas = np.pad(
            image,
            ((margins.top, margins.bottom), (margins.left, margins.right), (0, 0)),
            mode="edge",
        )
        mask = np.full((new_height, new_width), 255, dtype=np.uint8)
        mask[margins.top:margins.top + height, margins.left:margins.left + width] = 0
        options = self.context.generation_options(generation_options)
        return self.context.run_masked(
            mode="outpainting", image=canvas, mask=mask, options=options,
            invoke=lambda prepared_image, prepared_mask, opts:
                self.backend.edit(prepared_image, prepared_mask, prompt, opts),
        )
