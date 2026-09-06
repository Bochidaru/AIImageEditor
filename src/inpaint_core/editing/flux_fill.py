from __future__ import annotations

from typing import Any

import numpy as np
from PIL import Image

from ..config import DeviceConfig, ModelConfig
from ..models.manager import ModelManager
from ..types import GenerationOptions, GenerationResult, ImageArray, MaskArray


class FluxFillEditor:
    def __init__(
        self,
        config: ModelConfig,
        device: DeviceConfig,
        model_manager: ModelManager,
    ) -> None:
        self.config = config
        self.device = device
        self.model_manager = model_manager

    def edit(
        self,
        image: ImageArray,
        mask: MaskArray,
        prompt: str,
        options: GenerationOptions,
    ) -> GenerationResult:
        pipeline = self.model_manager.get("flux_fill", self._load_pipeline)

        import torch

        generator = torch.Generator("cpu").manual_seed(options.seed)
        kwargs: dict[str, Any] = {
            "prompt": prompt,
            "image": Image.fromarray(image, mode="RGB"),
            "mask_image": Image.fromarray(mask, mode="L"),
            "num_inference_steps": options.num_inference_steps,
            "generator": generator,
        }
        if options.guidance_scale is not None:
            kwargs["guidance_scale"] = options.guidance_scale

        output = pipeline(**kwargs).images[0].convert("RGB")
        return GenerationResult(
            image=np.asarray(output, dtype=np.uint8),
            seed=options.seed,
            metadata={"backend": "flux_fill"},
        )

    def _load_pipeline(self):
        if not self.config.model_id:
            raise ValueError("models.editing.model_id is not configured.")

        import torch
        from diffusers import FluxFillPipeline

        dtype = getattr(torch, self.device.dtype)
        pipeline = FluxFillPipeline.from_pretrained(
            self.config.model_id,
            torch_dtype=dtype,
        )

        if self.config.options.get("cpu_offload", False):
            pipeline.enable_model_cpu_offload()
        else:
            pipeline.to(self.device.name)
        return pipeline

