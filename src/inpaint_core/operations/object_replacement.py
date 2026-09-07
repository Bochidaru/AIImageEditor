from __future__ import annotations

from ..backends.protocols import MaskedEditor
from ..types import EditResult, GenerationOptions, ImageArray, MaskArray, MaskOptions, SelectionPrompt
from .common import OperationContext


class ObjectReplacementOperation:
    def __init__(self, context: OperationContext, backend: MaskedEditor):
        self.context, self.backend = context, backend

    def run(self, image: ImageArray, prompt: str, *, selection: SelectionPrompt | None = None,
            mask: MaskArray | None = None, mask_options: MaskOptions | None = None,
            generation_options: GenerationOptions | None = None) -> EditResult:
        if not prompt.strip():
            raise ValueError("A replacement prompt is required.")
        self.context.validate_image(image)
        mo = self.context.mask_options(mask_options)
        go = self.context.generation_options(generation_options)
        resolved = self.context.resolve_mask(image, selection, mask, mo)
        return self.context.run_masked(
            mode="object_replacement", image=image, mask=resolved, options=go,
            invoke=lambda prepared_image, prepared_mask, options:
                self.backend.edit(prepared_image, prepared_mask, prompt, options),
        )
