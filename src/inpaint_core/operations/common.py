from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

from ..backends.protocols import Segmenter
from ..config import AppConfig
from ..image import validate_image
from ..masks.operations import dilate, erode, threshold, validate_mask
from ..masks.transforms import prepare_full_image, restore_full_image
from ..models.memory import MemoryTracker
from ..types import (
    BoxPrompt,
    EditResult,
    GenerationOptions,
    GenerationResult,
    ImageArray,
    MaskArray,
    MaskOptions,
    SelectionPrompt,
)
from ..validation import require_exactly_one


@dataclass(slots=True)
class OperationContext:
    config: AppConfig
    segmenter: Segmenter
    memory: MemoryTracker

    def validate_image(self, image: ImageArray) -> None:
        validate_image(image)
        max_side = self.config.processing.max_image_side
        if max(image.shape[:2]) > max_side:
            raise ValueError(
                f"Image side exceeds processing.max_image_side={max_side}. "
                "Call preprocess_image first."
            )

    def mask_options(self, value: MaskOptions | None) -> MaskOptions:
        if value is not None:
            return value
        fields = MaskOptions.__dataclass_fields__
        return MaskOptions(**{
            key: item for key, item in self.config.mask_defaults.items() if key in fields
        })

    def generation_options(self, value: GenerationOptions | None) -> GenerationOptions:
        if value is not None:
            return value
        fields = GenerationOptions.__dataclass_fields__
        return GenerationOptions(**{
            key: item
            for key, item in self.config.generation_defaults.items()
            if key in fields
        })

    def resolve_mask(
        self,
        image: ImageArray,
        selection: SelectionPrompt | None,
        mask: MaskArray | None,
        options: MaskOptions,
    ) -> MaskArray:
        require_exactly_one(selection, mask, "selection", "mask")
        if mask is not None:
            validate_mask(mask, image.shape)
            raw = mask
        else:
            with self.memory.measure("segmentation"):
                result = self.segmenter.segment(image, selection)
            index = result.best_index if options.candidate_index is None else options.candidate_index
            if not 0 <= index < len(result.masks):
                raise IndexError("candidate_index is outside the returned mask list.")
            raw = result.masks[index]
        result = threshold(raw, options.threshold)
        result = dilate(result, options.dilate)
        result = erode(result, options.erode)
        validate_mask(result, image.shape)
        return result

    def run_masked(
        self,
        *,
        mode: str,
        image: ImageArray,
        mask: MaskArray,
        options: GenerationOptions,
        invoke: Callable[[ImageArray, MaskArray, GenerationOptions], GenerationResult],
    ) -> EditResult:
        prepared = prepare_full_image(
            image, mask, size_multiple=self.config.processing.size_multiple
        )
        with self.memory.measure(mode):
            generated = invoke(prepared.image, prepared.mask, options)
        restored = restore_full_image(generated.image, prepared.transform)
        return EditResult(
            image=restored,
            mask=mask,
            generated_image=restored,
            metadata={
                "mode": mode,
                "seed": generated.seed,
                "compositing": False,
                "stages": self.memory.results.copy(),
                **generated.metadata,
            },
        )


def mask_from_box(image: ImageArray, box: BoxPrompt) -> MaskArray:
    height, width = image.shape[:2]
    x1 = max(0, min(width, int(round(box.x1))))
    y1 = max(0, min(height, int(round(box.y1))))
    x2 = max(0, min(width, int(round(box.x2))))
    y2 = max(0, min(height, int(round(box.y2))))
    if x2 <= x1 or y2 <= y1:
        raise ValueError("Placement box is outside the image.")
    mask = np.zeros((height, width), dtype=np.uint8)
    mask[y1:y2, x1:x2] = 255
    return mask
