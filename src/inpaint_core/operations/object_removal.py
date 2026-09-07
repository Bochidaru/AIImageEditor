from __future__ import annotations

from ..backends.protocols import ObjectRemover
from ..types import EditResult, GenerationOptions, ImageArray, MaskArray, MaskOptions, SelectionPrompt
from .common import OperationContext


class ObjectRemovalOperation:
    def __init__(self, context: OperationContext, backend: ObjectRemover):
        self.context, self.backend = context, backend

    def run(self, image: ImageArray, *, selection: SelectionPrompt | None = None,
            mask: MaskArray | None = None, mask_options: MaskOptions | None = None,
            generation_options: GenerationOptions | None = None) -> EditResult:
        self.context.validate_image(image)
        mo = self.context.mask_options(mask_options)
        go = self.context.generation_options(generation_options)
        resolved = self.context.resolve_mask(image, selection, mask, mo)
        return self.context.run_masked(
            mode="object_removal", image=image, mask=resolved, options=go,
            invoke=lambda prepared_image, prepared_mask, options:
                self.backend.remove(prepared_image, prepared_mask, options),
        )
