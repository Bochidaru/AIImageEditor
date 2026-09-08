"""Adapter for the official yeates/OmniPaint inference runtime.

OmniPaint changes FLUX's transformer forward pass, so loading its LoRA with a
plain Diffusers pipeline is insufficient. This adapter intentionally imports
the official repository's Condition and generate implementation.
"""

from __future__ import annotations

import importlib
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from ...config import DeviceConfig, ModelConfig
from ...models.manager import ModelManager
from ...types import GenerationOptions, GenerationResult, ImageArray, MaskArray


@dataclass(slots=True)
class _Runtime:
    pipeline: Any
    Condition: type
    generate: Any
    embeddings: dict[str, tuple[Any, Any, Any]]


class OmniPaintBackend:
    RESOURCE_KEY = "omnipaint"

    def __init__(self, config: ModelConfig, device: DeviceConfig, manager: ModelManager):
        self.config = config
        self.device = device
        self.manager = manager

    def remove(
        self, image: ImageArray, mask: MaskArray, options: GenerationOptions,
    ) -> GenerationResult:
        runtime = self.manager.get(self.RESOURCE_KEY, self._load_runtime)
        source = Image.fromarray(image, mode="RGB")
        binary_mask = Image.fromarray(mask, mode="L")
        conditioned = Image.composite(
            Image.new("RGB", source.size, "black"), source, binary_mask
        )
        return self._generate(runtime, "removal", [runtime.Condition("removal", conditioned)], image, options)

    def insert(
        self, image: ImageArray, mask: MaskArray, reference: ImageArray,
        options: GenerationOptions,
    ) -> GenerationResult:
        runtime = self.manager.get(self.RESOURCE_KEY, self._load_runtime)
        source = Image.fromarray(image, mode="RGB")
        binary_mask = Image.fromarray(mask, mode="L")
        conditioned = Image.composite(
            Image.new("RGB", source.size, "black"), source, binary_mask
        )
        reference_size = int(self.config.options.get("reference_size", 512))
        subject = Image.fromarray(reference, mode="RGB").resize(
            (reference_size, reference_size), Image.Resampling.LANCZOS
        )
        conditions = [
            runtime.Condition("insertion", conditioned),
            runtime.Condition(
                "insertion",
                subject,
                position_delta=tuple(
                    self.config.options.get("reference_position_delta", [0, -32])
                ),
            ),
        ]
        return self._generate(runtime, "insertion", conditions, image, options)

    def _generate(
        self, runtime: _Runtime, task: str, conditions: list[Any],
        image: ImageArray, options: GenerationOptions,
    ) -> GenerationResult:
        import torch

        self._prepare_custom_transformer(runtime.pipeline, torch)
        prompt_embeds, pooled_prompt_embeds, text_ids = runtime.embeddings[task]
        guidance = 3.5 if options.guidance_scale is None else options.guidance_scale
        output = runtime.generate(
            runtime.pipeline,
            conditions=conditions,
            width=image.shape[1],
            height=image.shape[0],
            num_inference_steps=options.num_inference_steps,
            guidance_scale=guidance,
            generator=torch.Generator("cpu").manual_seed(options.seed),
            prompt_embeds=prompt_embeds,
            pooled_prompt_embeds=pooled_prompt_embeds,
            text_ids=text_ids,
        ).images[0].convert("RGB")
        return GenerationResult(
            np.ascontiguousarray(np.asarray(output, dtype=np.uint8)),
            options.seed,
            {"backend": "omnipaint", "task": task},
        )

    def _prepare_custom_transformer(self, pipeline: Any, torch: Any) -> None:
        """Place FLUX transformer explicitly for OmniPaint's custom forward.

        OmniPaint calls ``tranformer_forward`` directly, bypassing the normal
        Diffusers module-forward hook that would trigger CPU offload.
        """
        if not self.config.options.get("cpu_offload", True):
            return
        execution_device = getattr(
            pipeline, "_execution_device", torch.device(self.device.name)
        )
        pipeline.transformer.to(execution_device)

    def _load_runtime(self) -> _Runtime:
        if not self.config.model_id:
            raise ValueError("models.omnipaint.model_id is not configured.")
        source_path = Path(
            self.config.options.get("source_path", "third_party/OmniPaint")
        ).expanduser().resolve()
        if not (source_path / "src" / "generate.py").is_file():
            raise FileNotFoundError(
                "Official OmniPaint source was not found at "
                f"{source_path}. Run scripts/install_omnipaint.py first or set "
                "models.omnipaint.source_path."
            )

        source_string = str(source_path)
        if source_string not in sys.path:
            sys.path.insert(0, source_string)
        try:
            self._install_diffusers_compat()
            condition_module = importlib.import_module("src.condition")
            flux_core_module = importlib.import_module("src.flux_core")
            self._install_device_compat(condition_module, flux_core_module)
            Condition = condition_module.Condition
            generate = importlib.import_module("src.generate").generate
        except (ImportError, AttributeError) as exc:
            raise ImportError(
                "Could not import the official OmniPaint runtime from "
                f"{source_path}. Root cause: {type(exc).__name__}: {exc}"
            ) from exc

        import torch
        from diffusers import FluxPipeline

        dtype = getattr(torch, self.device.dtype)
        pipeline = FluxPipeline.from_pretrained(
            self.config.options.get("base_model_id", "black-forest-labs/FLUX.1-dev"),
            torch_dtype=dtype,
        )
        files = {
            "removal": (
                self.config.options.get("removal_lora", "weights/omnipaint_remove.safetensors"),
                self.config.options.get("removal_embedding", "embeddings/remove.npz"),
            ),
            "insertion": (
                self.config.options.get("insertion_lora", "weights/omnipaint_insert.safetensors"),
                self.config.options.get("insertion_embedding", "embeddings/insert.npz"),
            ),
        }
        embeddings: dict[str, tuple[Any, Any, Any]] = {}
        for task, (lora_name, embedding_name) in files.items():
            pipeline.load_lora_weights(
                self.config.model_id,
                weight_name=lora_name,
                adapter_name=task,
            )
            model_path = Path(self.config.model_id).expanduser()
            if model_path.is_dir():
                embedding_path = model_path / embedding_name
                if not embedding_path.is_file():
                    raise FileNotFoundError(
                        f"OmniPaint embedding not found: {embedding_path}"
                    )
            else:
                from huggingface_hub import hf_hub_download

                embedding_path = hf_hub_download(
                    self.config.model_id,
                    embedding_name,
                )
            embeddings[task] = self._load_embeddings(embedding_path, torch, dtype)

        if self.config.options.get("cpu_offload", True):
            pipeline.enable_model_cpu_offload()
        else:
            pipeline.to(self.device.name)
        return _Runtime(pipeline, Condition, generate, embeddings)

    @staticmethod
    def _install_diffusers_compat() -> None:
        """Bridge a private Diffusers symbol moved after OmniPaint was released."""
        from diffusers.models.transformers import transformer_flux
        from diffusers.utils.torch_utils import is_torch_version

        if not hasattr(transformer_flux, "is_torch_version"):
            transformer_flux.is_torch_version = is_torch_version

    @staticmethod
    def _install_device_compat(condition_module: Any, flux_core_module: Any) -> None:
        """Use Diffusers' execution device when CPU offload is enabled."""
        def encode_images(pipeline: Any, images: Any):
            execution_device = getattr(
                pipeline, "_execution_device", pipeline.device
            )
            images = pipeline.image_processor.preprocess(images)
            images = images.to(execution_device).to(pipeline.dtype)
            images = pipeline.vae.encode(images).latent_dist.sample()
            images = (
                images - pipeline.vae.config.shift_factor
            ) * pipeline.vae.config.scaling_factor
            images_tokens = pipeline._pack_latents(images, *images.shape)
            images_ids = pipeline._prepare_latent_image_ids(
                images.shape[0],
                images.shape[2],
                images.shape[3],
                execution_device,
                pipeline.dtype,
            )
            if images_tokens.shape[1] != images_ids.shape[0]:
                images_ids = pipeline._prepare_latent_image_ids(
                    images.shape[0],
                    images.shape[2] // 2,
                    images.shape[3] // 2,
                    execution_device,
                    pipeline.dtype,
                )
            return images_tokens, images_ids

        # condition.py imported encode_images into its own module namespace,
        # so both references must be replaced.
        condition_module.encode_images = encode_images
        flux_core_module.encode_images = encode_images

    def _load_embeddings(self, path: str, torch: Any, dtype: Any):
        data = np.load(path)
        required = ("prompt_embeds", "pooled_prompt_embeds", "text_ids")
        missing = [key for key in required if key not in data]
        if missing:
            raise KeyError(f"OmniPaint embedding is missing keys: {missing}.")
        device = self.device.name
        prompt = torch.from_numpy(data["prompt_embeds"]).to(device=device, dtype=dtype)
        pooled = torch.from_numpy(data["pooled_prompt_embeds"]).to(device=device, dtype=dtype)
        ids = data["text_ids"]
        if ids.ndim == 3 and ids.shape[0] == 1:
            ids = ids[0]
        text_ids = torch.as_tensor(ids, device=device, dtype=torch.long)
        return prompt, pooled, text_ids
