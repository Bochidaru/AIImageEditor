"""FLUX.2 Klein 4B backend for direct image editing and text-to-image."""

from __future__ import annotations

from typing import Any

import numpy as np
from PIL import Image

from ...config import DeviceConfig, ModelConfig
from ...models.manager import ModelManager
from ...types import GenerationOptions, GenerationResult, ImageArray


class Flux2KleinBackend:
    """One shared Klein pipeline implementing both backend protocols."""

    RESOURCE_KEY = "flux2_klein"

    def __init__(
        self, config: ModelConfig, device: DeviceConfig, manager: ModelManager,
    ) -> None:
        self.config = config
        self.device = device
        self.manager = manager

    def edit_image(
        self, image: ImageArray, prompt: str, options: GenerationOptions,
    ) -> GenerationResult:
        pipe = self.manager.get(self.RESOURCE_KEY, self._load_pipeline)
        prepared, original_size = self._pad_image(image)
        kwargs = self._generation_kwargs(options)
        kwargs.update(
            {
                "image": Image.fromarray(prepared, mode="RGB"),
                "prompt": prompt,
                "height": prepared.shape[0],
                "width": prepared.shape[1],
            }
        )
        output = pipe(**kwargs).images[0].convert("RGB")
        output = output.crop((0, 0, original_size[0], original_size[1]))
        return self._result(output, options, task="image_edit")

    def generate(
        self,
        prompt: str,
        width: int,
        height: int,
        options: GenerationOptions,
    ) -> GenerationResult:
        pipe = self.manager.get(self.RESOURCE_KEY, self._load_pipeline)
        kwargs = self._generation_kwargs(options)
        kwargs.update({"prompt": prompt, "width": width, "height": height})
        output = pipe(**kwargs).images[0].convert("RGB")
        return self._result(output, options, task="text_to_image")

    def _generation_kwargs(self, options: GenerationOptions) -> dict[str, Any]:
        import torch

        steps = options.num_inference_steps or int(
            self.config.options.get("num_inference_steps", 4)
        )
        guidance = (
            float(self.config.options.get("guidance_scale", 1.0))
            if options.guidance_scale is None
            else options.guidance_scale
        )
        generator_device = self.device.name if self.device.name.startswith("cuda") else "cpu"
        return {
            "num_inference_steps": steps,
            "guidance_scale": guidance,
            "generator": torch.Generator(generator_device).manual_seed(options.seed),
        }

    def _load_pipeline(self):
        if not self.config.model_id:
            raise ValueError("models.flux2_klein.model_id is not configured.")
        import torch
        from diffusers import Flux2KleinPipeline

        try:
            dtype = getattr(torch, self.device.dtype)
        except AttributeError as exc:
            raise ValueError(f"Unsupported torch dtype: {self.device.dtype!r}.") from exc
        pipeline = Flux2KleinPipeline.from_pretrained(
            self.config.model_id,
            torch_dtype=dtype,
        )
        if self.config.options.get("vae_tiling", True):
            pipeline.enable_vae_tiling()
        if self.config.options.get("vae_slicing", True):
            pipeline.enable_vae_slicing()
        if self.config.options.get("cpu_offload", True):
            pipeline.enable_model_cpu_offload()
        else:
            pipeline.to(self.device.name)
        return pipeline

    def _pad_image(self, image: ImageArray) -> tuple[ImageArray, tuple[int, int]]:
        multiple = int(self.config.options.get("size_multiple", 16))
        height, width = image.shape[:2]
        pad_bottom = (-height) % multiple
        pad_right = (-width) % multiple
        if not pad_bottom and not pad_right:
            return image, (width, height)
        prepared = np.pad(
            image,
            ((0, pad_bottom), (0, pad_right), (0, 0)),
            mode="reflect",
        )
        return np.ascontiguousarray(prepared), (width, height)

    def _result(
        self, output: Image.Image, options: GenerationOptions, *, task: str,
    ) -> GenerationResult:
        return GenerationResult(
            np.ascontiguousarray(np.asarray(output, dtype=np.uint8)),
            options.seed,
            {
                "backend": "flux2_klein",
                "model_id": self.config.model_id,
                "task": task,
                "num_inference_steps": options.num_inference_steps
                or int(self.config.options.get("num_inference_steps", 4)),
            },
        )
