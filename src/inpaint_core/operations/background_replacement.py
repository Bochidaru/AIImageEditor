from __future__ import annotations

from ..backends.protocols import MaskedEditor
from ..masks.operations import invert
from ..types import EditResult, GenerationOptions, ImageArray, MaskArray, MaskOptions, SelectionPrompt
from .common import OperationContext


class BackgroundReplacementOperation:
    def __init__(self, context: OperationContext, backend: MaskedEditor):
        self.context, self.backend = context, backend

    def run(self, image: ImageArray, prompt: str, *,
            foreground_selection: SelectionPrompt | None = None,
            foreground_mask: MaskArray | None = None,
            mask_options: MaskOptions | None = None,
            generation_options: GenerationOptions | None = None) -> EditResult:
        if not prompt.strip():
            raise ValueError("A background prompt is required.")
        self.context.validate_image(image)
        mo = self.context.mask_options(mask_options)
        go = self.context.generation_options(generation_options)
        foreground = self.context.resolve_mask(
            image, foreground_selection, foreground_mask, mo
        )
        background = invert(foreground)
        return self.context.run_masked(
            mode="background_replacement", image=image, mask=background, options=go,
            invoke=lambda prepared_image, prepared_mask, options:
                self.backend.edit(prepared_image, prepared_mask, prompt, options),
        )
