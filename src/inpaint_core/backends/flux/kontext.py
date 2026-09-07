from __future__ import annotations

from typing import Any

import numpy as np
from PIL import Image

from ...config import DeviceConfig, ModelConfig
from ...models.manager import ModelManager
from ...types import GenerationOptions, GenerationResult, ImageArray
from .common import generator_for, pipeline_load_options, place_pipeline


class FluxKontextBackend:
    RESOURCE_KEY = "flux_kontext"

    def __init__(self, config: ModelConfig, device: DeviceConfig, manager: ModelManager):
        self.config = config
        self.device = device
        self.manager = manager

    def edit_image(
        self, image: ImageArray, prompt: str, options: GenerationOptions,
    ) -> GenerationResult:
        pipe = self.manager.get(self.RESOURCE_KEY, self._load_pipeline)
        kwargs: dict[str, Any] = {
            "image": Image.fromarray(image, mode="RGB"),
            "prompt": prompt,
            "num_inference_steps": options.num_inference_steps,
            "generator": generator_for(options.seed),
        }
        if options.guidance_scale is not None:
            kwargs["guidance_scale"] = options.guidance_scale
        output = pipe(**kwargs).images[0].convert("RGB")
        if output.size != (image.shape[1], image.shape[0]):
            output = output.resize((image.shape[1], image.shape[0]), Image.Resampling.LANCZOS)
        return GenerationResult(
            np.ascontiguousarray(np.asarray(output, dtype=np.uint8)),
            options.seed,
            {"backend": "flux_kontext", "model_id": self.config.model_id},
        )

    def _load_pipeline(self):
        if not self.config.model_id:
            raise ValueError("models.flux_kontext.model_id is not configured.")
        from diffusers import FluxKontextPipeline

        pipe = FluxKontextPipeline.from_pretrained(
            self.config.model_id,
            **pipeline_load_options(self.config, self.device),
        )
        return place_pipeline(pipe, self.config, self.device)
