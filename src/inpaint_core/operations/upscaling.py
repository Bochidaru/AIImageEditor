from __future__ import annotations

from ..backends.protocols import Upscaler
from ..image import validate_image
from ..types import GenerationResult, ImageArray, UpscaleOptions
from .common import OperationContext


class UpscalingOperation:
    def __init__(self, context: OperationContext, backend: Upscaler):
        self.context, self.backend = context, backend

    def run(
        self, image: ImageArray, options: UpscaleOptions | None = None,
    ) -> GenerationResult:
        validate_image(image)
        options = options or UpscaleOptions()
        with self.context.memory.measure("upscaling"):
            result = self.backend.upscale(image, options)
        result.metadata.update({"mode": "upscaling", "stages": self.context.memory.results.copy()})
        return result
