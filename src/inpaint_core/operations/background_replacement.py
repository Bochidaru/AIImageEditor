from __future__ import annotations

from ..backends.protocols import PromptEditor
from ..types import GenerationOptions, GenerationResult, ImageArray
from .common import OperationContext


class BackgroundReplacementOperation:
    def __init__(self, context: OperationContext, backend: PromptEditor):
        self.context, self.backend = context, backend

    def run(
        self,
        image: ImageArray,
        prompt: str,
        *,
        generation_options: GenerationOptions | None = None,
    ) -> GenerationResult:
        if not prompt.strip():
            raise ValueError("A background prompt is required.")
        self.context.validate_image(image)
        options = self.context.generation_options(generation_options)
        instruction = (
            f"Replace only the background with {prompt.strip()}. "
            "Preserve the main foreground subject exactly: keep its identity, "
            "appearance, pose, size, position, and camera framing unchanged. "
            "Make the new background photorealistic with coherent perspective, "
            "lighting, shadows, and depth of field."
        )
        with self.context.memory.measure("background_replacement"):
            result = self.backend.edit_image(image, instruction, options)
        result.metadata.update(
            {
                "mode": "background_replacement",
                "stages": self.context.memory.results.copy(),
            }
        )
        return result
