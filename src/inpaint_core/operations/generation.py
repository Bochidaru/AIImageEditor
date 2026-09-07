from __future__ import annotations

from ..backends.protocols import ImageGenerator
from ..types import GenerationOptions, GenerationResult
from .common import OperationContext


class ImageGenerationOperation:
    def __init__(self, context: OperationContext, backend: ImageGenerator):
        self.context, self.backend = context, backend

    def run(
        self, prompt: str, *, width: int = 1024, height: int = 1024,
        generation_options: GenerationOptions | None = None,
    ) -> GenerationResult:
        if not prompt.strip():
            raise ValueError("A generation prompt is required.")
        multiple = self.context.config.processing.size_multiple
        if width <= 0 or height <= 0 or width % multiple or height % multiple:
            raise ValueError(f"width and height must be positive multiples of {multiple}.")
        if max(width, height) > self.context.config.processing.max_image_side:
            raise ValueError("Generated dimensions exceed processing.max_image_side.")
        options = self.context.generation_options(generation_options)
        with self.context.memory.measure("image_generation"):
            result = self.backend.generate(prompt, width, height, options)
        result.metadata.update({"mode": "image_generation", "stages": self.context.memory.results.copy()})
        return result
