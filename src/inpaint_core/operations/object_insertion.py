from __future__ import annotations

import numpy as np

from ..backends.protocols import MaskedEditor, ReferenceInserter
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
from ..validation import require_exactly_one
from .common import OperationContext, mask_from_box


class ObjectInsertionOperation:
    def __init__(
        self,
        context: OperationContext,
        prompt_backend: MaskedEditor,
        reference_backend: ReferenceInserter,
    ):
        self.context = context
        self.prompt_backend = prompt_backend
        self.reference_backend = reference_backend

    def by_prompt(
        self, image: ImageArray, prompt: str, *, placement: BoxPrompt | None = None,
        mask: MaskArray | None = None, mask_options: MaskOptions | None = None,
        generation_options: GenerationOptions | None = None,
    ) -> EditResult:
        if not prompt.strip():
            raise ValueError("An object prompt is required.")
        self.context.validate_image(image)
        resolved = self._placement_mask(image, placement, mask, mask_options)
        go = self.context.generation_options(generation_options)
        return self.context.run_masked(
            mode="object_insertion_prompt", image=image, mask=resolved, options=go,
            invoke=lambda prepared_image, prepared_mask, options:
                self.prompt_backend.edit(prepared_image, prepared_mask, prompt, options),
        )

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
        require_exactly_one(placement, mask, "placement", "mask")
        mo = self.context.mask_options(options)
        raw = mask_from_box(image, placement) if placement is not None else mask
        return self.context.resolve_mask(image, None, raw, mo)
