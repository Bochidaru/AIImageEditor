"""Small backend contracts used by operations.

These are structural Protocols, not parent classes. A backend only needs to
provide the matching method; it does not inherit from anything here.
"""

from __future__ import annotations

from typing import Protocol

from ..types import (
    GenerationOptions,
    GenerationResult,
    ImageArray,
    MaskArray,
    SegmentationResult,
    SelectionPrompt,
    UpscaleOptions,
)


class Segmenter(Protocol):
    def segment(self, image: ImageArray, selection: SelectionPrompt) -> SegmentationResult: ...


class MaskedEditor(Protocol):
    def edit(
        self, image: ImageArray, mask: MaskArray, prompt: str,
        options: GenerationOptions,
    ) -> GenerationResult: ...


class ObjectRemover(Protocol):
    def remove(
        self, image: ImageArray, mask: MaskArray, options: GenerationOptions,
    ) -> GenerationResult: ...


class ReferenceInserter(Protocol):
    def insert(
        self, image: ImageArray, mask: MaskArray, reference: ImageArray,
        options: GenerationOptions,
    ) -> GenerationResult: ...


class PromptEditor(Protocol):
    def edit_image(
        self, image: ImageArray, prompt: str, options: GenerationOptions,
    ) -> GenerationResult: ...


class ImageGenerator(Protocol):
    def generate(
        self, prompt: str, width: int, height: int, options: GenerationOptions,
    ) -> GenerationResult: ...


class Upscaler(Protocol):
    def upscale(self, image: ImageArray, options: UpscaleOptions) -> GenerationResult: ...
