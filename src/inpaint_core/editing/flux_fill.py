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
            metadata={
                "backend": "flux_fill",
                "quantization": self.config.options.get("quantization"),
            },
        )

    def _load_pipeline(self):
        if not self.config.model_id:
            raise ValueError("models.editing.model_id is not configured.")

        import torch
        from diffusers import FluxFillPipeline

        dtype = getattr(torch, self.device.dtype)
        load_options: dict[str, Any] = {"torch_dtype": dtype}
        quantization = self.config.options.get("quantization")

        if quantization == "bitsandbytes_8bit":
            from diffusers.quantizers import PipelineQuantizationConfig

            components = self.config.options.get(
                "quantization_components", ["transformer"]
            )
            if not isinstance(components, list) or not components:
                raise ValueError(
                    "quantization_components must be a non-empty list."
                )

            load_options["quantization_config"] = PipelineQuantizationConfig(
                quant_backend="bitsandbytes_8bit",
                quant_kwargs={"load_in_8bit": True},
                components_to_quantize=components,
            )
        elif quantization is not None:
            raise ValueError(
                f"Unsupported Flux quantization mode: {quantization!r}."
            )

        pipeline = FluxFillPipeline.from_pretrained(
            self.config.model_id,
            **load_options,
        )

        if self.config.options.get("cpu_offload", False):
            pipeline.enable_model_cpu_offload()
        else:
            pipeline.to(self.device.name)
        return pipeline
