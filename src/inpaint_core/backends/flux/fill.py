from __future__ import annotations

from typing import Any

import numpy as np
from PIL import Image

from ...config import DeviceConfig, ModelConfig
from ...models.manager import ModelManager
from ...types import GenerationOptions, GenerationResult, ImageArray, MaskArray
from .common import generator_for, pipeline_load_options, place_pipeline


class FluxFillBackend:
    RESOURCE_KEY = "flux_fill"

    def __init__(self, config: ModelConfig, device: DeviceConfig, manager: ModelManager):
        self.config = config
        self.device = device
        self.manager = manager

    def edit(
        self, image: ImageArray, mask: MaskArray, prompt: str,
        options: GenerationOptions,
    ) -> GenerationResult:
        pipe = self.manager.get(self.RESOURCE_KEY, self._load_pipeline)
        kwargs: dict[str, Any] = {
            "prompt": prompt,
            "image": Image.fromarray(image, mode="RGB"),
            "mask_image": Image.fromarray(mask, mode="L"),
            "height": image.shape[0],
            "width": image.shape[1],
            "num_inference_steps": options.num_inference_steps or int(
                self.config.options.get("num_inference_steps", 28)
            ),
            "generator": generator_for(options.seed),
        }
        if options.guidance_scale is not None:
            kwargs["guidance_scale"] = options.guidance_scale
        output = pipe(**kwargs).images[0].convert("RGB")
        return GenerationResult(
            np.ascontiguousarray(np.asarray(output, dtype=np.uint8)),
            options.seed,
            {"backend": "flux_fill", "model_id": self.config.model_id},
        )

    def _load_pipeline(self):
        if not self.config.model_id:
            raise ValueError("models.flux_fill.model_id is not configured.")
        from diffusers import FluxFillPipeline

        pipe = FluxFillPipeline.from_pretrained(
            self.config.model_id,
            **pipeline_load_options(self.config, self.device),
        )
        return place_pipeline(pipe, self.config, self.device)
