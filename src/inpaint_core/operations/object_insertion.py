from __future__ import annotations

import numpy as np

from ..backends.protocols import PromptEditor, ReferenceInserter
from ..image import validate_image
from ..masks.operations import threshold, validate_mask
from ..types import (
    BoxPrompt,
    EditResult,
    GenerationOptions,
    ImageArray,
    MaskArray,
    MaskOptions,
)
from .common import OperationContext, mask_from_box


class ObjectInsertionOperation:
    def __init__(
        self,
        context: OperationContext,
        prompt_backend: PromptEditor,
        reference_backend: ReferenceInserter,
    ):
        self.context = context
        self.prompt_backend = prompt_backend
        self.reference_backend = reference_backend

    def by_prompt(
        self, image: ImageArray, prompt: str, *, placement: BoxPrompt | None = None,
        generation_options: GenerationOptions | None = None,
    ) -> GenerationResult:
        if not prompt.strip():
            raise ValueError("An object prompt is required.")
        self.context.validate_image(image)
        options = self.context.generation_options(generation_options)
        location = self._placement_description(image, placement)
        instruction = (
            f"Add {prompt.strip()}{location}. Keep all existing subjects and "
            "background content unchanged. Integrate the new object naturally "
            "with realistic scale, perspective, lighting, contact shadow, and occlusion."
        )
        with self.context.memory.measure("object_insertion_prompt"):
            result = self.prompt_backend.edit_image(image, instruction, options)
        result.metadata.update(
            {
                "mode": "object_insertion_prompt",
                "stages": self.context.memory.results.copy(),
            }
        )
        return result

    def by_reference(
        self, image: ImageArray, reference: ImageArray, *,
        placement: BoxPrompt | None = None, mask: MaskArray | None = None,
        reference_mask: MaskArray | None = None,
        mask_options: MaskOptions | None = None,
        generation_options: GenerationOptions | None = None,
    ) -> EditResult:
        self.context.validate_image(image)
        validate_image(reference)
        resolved = self._placement_mask(image, placement, mask, mask_options)
        subject = reference
        if reference_mask is not None:
            validate_mask(reference_mask, reference.shape)
            alpha = threshold(reference_mask)[..., None].astype(np.float32) / 255.0
            subject = np.clip(
                reference.astype(np.float32) * alpha + 255.0 * (1.0 - alpha),
                0,
                255,
            ).astype(np.uint8)
        go = self.context.generation_options(generation_options)
        return self.context.run_masked(
            mode="object_insertion_reference", image=image, mask=resolved, options=go,
            invoke=lambda prepared_image, prepared_mask, options:
                self.reference_backend.insert(
                    prepared_image, prepared_mask, subject, options
                ),
        )

    def _placement_mask(
        self, image: ImageArray, placement: BoxPrompt | None,
        mask: MaskArray | None, options: MaskOptions | None,
    ) -> MaskArray:
        if (placement is None) == (mask is None):
            raise ValueError("Provide exactly one of placement or mask.")
        mo = self.context.mask_options(options)
        raw = mask_from_box(image, placement) if placement is not None else mask
        return self.context.resolve_mask(image, None, raw, mo)

    @staticmethod
    def _placement_description(
        image: ImageArray, placement: BoxPrompt | None,
    ) -> str:
        if placement is None:
            return " in a natural, context-appropriate location"
        height, width = image.shape[:2]
        center_x = (placement.x1 + placement.x2) / 2 / width
        center_y = (placement.y1 + placement.y2) / 2 / height
        horizontal = "left" if center_x < 1 / 3 else "right" if center_x > 2 / 3 else "center"
        vertical = "upper" if center_y < 1 / 3 else "lower" if center_y > 2 / 3 else "middle"
        box_fraction = max(
            (placement.x2 - placement.x1) / width,
            (placement.y2 - placement.y1) / height,
        )
        size = "small" if box_fraction < 0.25 else "medium-sized" if box_fraction < 0.5 else "large"
        return f" as a {size} object in the {vertical}-{horizontal} area"
