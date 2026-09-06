from __future__ import annotations

from typing import Protocol

from ..types import GenerationOptions, GenerationResult, ImageArray, MaskArray


class MaskedImageEditor(Protocol):
    def edit(
        self,
        image: ImageArray,
        mask: MaskArray,
        prompt: str,
        options: GenerationOptions,
    ) -> GenerationResult:
        ...

