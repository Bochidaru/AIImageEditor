from __future__ import annotations

from typing import Any

import numpy as np

from ...config import DeviceConfig, ModelConfig
from ...models.manager import ModelManager
from ...types import GenerationOptions, GenerationResult
from .common import generator_for, pipeline_load_options, place_pipeline


class FluxGeneratorBackend:
    """Text-to-image FLUX backend; this deliberately does not use FLUX Fill."""

    RESOURCE_KEY = "flux_generation"

    def __init__(self, config: ModelConfig, device: DeviceConfig, manager: ModelManager):
        self.config = config
        self.device = device
        self.manager = manager

    def generate(
        self, prompt: str, width: int, height: int, options: GenerationOptions,
    ) -> GenerationResult:
        pipe = self.manager.get(self.RESOURCE_KEY, self._load_pipeline)
        kwargs: dict[str, Any] = {
            "prompt": prompt,
            "width": width,
            "height": height,
            "num_inference_steps": options.num_inference_steps,
            "generator": generator_for(options.seed),
        }
        if options.guidance_scale is not None:
            kwargs["guidance_scale"] = options.guidance_scale
        output = pipe(**kwargs).images[0].convert("RGB")
        return GenerationResult(
            np.ascontiguousarray(np.asarray(output, dtype=np.uint8)),
            options.seed,
            {"backend": "flux_generation", "model_id": self.config.model_id},
        )

    def _load_pipeline(self):
        if not self.config.model_id:
            raise ValueError("models.flux_generation.model_id is not configured.")
        from diffusers import FluxPipeline

        pipe = FluxPipeline.from_pretrained(
            self.config.model_id,
            **pipeline_load_options(self.config, self.device),
        )
        return place_pipeline(pipe, self.config, self.device)
