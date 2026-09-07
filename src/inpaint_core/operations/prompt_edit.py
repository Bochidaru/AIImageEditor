from __future__ import annotations

from ..backends.protocols import PromptEditor
from ..types import GenerationOptions, GenerationResult, ImageArray
from .common import OperationContext


class PromptEditOperation:
    def __init__(self, context: OperationContext, backend: PromptEditor):
        self.context, self.backend = context, backend

    def run(
        self, image: ImageArray, prompt: str,
        generation_options: GenerationOptions | None = None,
    ) -> GenerationResult:
        if not prompt.strip():
            raise ValueError("An edit prompt is required.")
        self.context.validate_image(image)
        options = self.context.generation_options(generation_options)
        with self.context.memory.measure("prompt_edit"):
            result = self.backend.edit_image(image, prompt, options)
        result.metadata.update({"mode": "prompt_edit", "stages": self.context.memory.results.copy()})
        return result
