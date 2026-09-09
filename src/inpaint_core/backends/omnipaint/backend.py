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


@dataclass(slots=True)
class _EncodedCondition:
    """Condition tensors cached before OmniPaint's custom denoising loop."""

    condition_type: str
    encoded: tuple[Any, Any, Any]

    def encode(self, pipeline: Any) -> tuple[Any, Any, Any]:
        del pipeline
        return self.encoded


class OmniPaintBackend:
    RESOURCE_KEY = "omnipaint"

    def __init__(self, config: ModelConfig, device: DeviceConfig, manager: ModelManager):
        self.config = config
        self.device = device
        self.manager = manager

    def remove(
        self, image: ImageArray, mask: MaskArray, options: GenerationOptions,
    ) -> GenerationResult:
        self._validate_target_size(image)
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
        self._validate_target_size(image)
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

        # Encoding a condition invokes the VAE. Diffusers' sequential offload
        # hook consequently moves the transformer back to CPU. OmniPaint then
        # bypasses the transformer's normal forward hook, so encode everything
        # first and only afterwards put the transformer on its execution device.
        encoded_conditions = [
            _EncodedCondition(
                condition_type=condition.condition_type,
                encoded=condition.encode(runtime.pipeline),
            )
            for condition in conditions
        ]
        execution_device = self._prepare_custom_transformer(runtime.pipeline, torch)
        prompt_embeds, pooled_prompt_embeds, text_ids = (
            tensor.to(execution_device) for tensor in runtime.embeddings[task]
        )
        guidance = (
            float(self.config.options.get("guidance_scale", 3.5))
            if options.guidance_scale is None
            else options.guidance_scale
        )
        steps = options.num_inference_steps or int(
            self.config.options.get("num_inference_steps", 28)
        )
        try:
            output = runtime.generate(
                runtime.pipeline,
                conditions=encoded_conditions,
                width=image.shape[1],
                height=image.shape[0],
                num_inference_steps=steps,
                guidance_scale=guidance,
                generator=torch.Generator("cpu").manual_seed(options.seed),
                prompt_embeds=prompt_embeds,
                pooled_prompt_embeds=pooled_prompt_embeds,
                text_ids=text_ids,
            ).images[0].convert("RGB")
        finally:
            if self.config.options.get("release_cuda_after_run", True):
                del encoded_conditions, prompt_embeds, pooled_prompt_embeds, text_ids
                self._release_cuda(runtime.pipeline, torch)
        return GenerationResult(
            np.ascontiguousarray(np.asarray(output, dtype=np.uint8)),
            options.seed,
            {
                "backend": "omnipaint",
                "task": task,
                "quantization": self.config.options.get("quantization"),
                "num_inference_steps": steps,
            },
        )

    def _prepare_custom_transformer(self, pipeline: Any, torch: Any) -> Any:
        """Place FLUX transformer explicitly for OmniPaint's custom forward.

        OmniPaint calls ``tranformer_forward`` directly, bypassing the normal
        Diffusers module-forward hook that would trigger CPU offload.
        """
        execution_device = getattr(
            pipeline, "_execution_device", torch.device(self.device.name)
        )
        if self.config.options.get("cpu_offload", True):
            pipeline.transformer.to(execution_device)
        return execution_device

    @staticmethod
    def _release_cuda(pipeline: Any, torch: Any) -> None:
        """Explicitly offload after OmniPaint's hook-bypassing custom forward."""
        try:
            pipeline.maybe_free_model_hooks()
        finally:
            hook = getattr(pipeline.transformer, "_hf_hook", None)
            offload = getattr(hook, "offload", None)
            if callable(offload):
                offload()
            else:
                try:
                    pipeline.transformer.to("cpu")
                except (RuntimeError, ValueError):
                    # Older bitsandbytes releases may reject explicit device
                    # migration; the manager will still delete this resource
                    # before another backend is loaded in sequential mode.
                    pass
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    def _load_runtime(self) -> _Runtime:
        if not self.config.model_id:
            raise ValueError("models.omnipaint.model_id is not configured.")
        source_path = Path(
            self.config.options.get("source_path", "third_party/OmniPaint")
        ).expanduser().resolve()
        if not (source_path / "src" / "generate.py").is_file():
            raise FileNotFoundError(
                "Official OmniPaint source was not found at "
                f"{source_path}. Clone the official repository there or set "
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
        load_options = self._pipeline_load_options(dtype)
        pipeline = FluxPipeline.from_pretrained(
            self.config.options.get("base_model_id", "black-forest-labs/FLUX.1-dev"),
            **load_options,
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
        if self.config.options.get("vae_tiling", True):
            pipeline.enable_vae_tiling()
        if self.config.options.get("vae_slicing", True):
            pipeline.enable_vae_slicing()
        return _Runtime(pipeline, Condition, generate, embeddings)

    def _pipeline_load_options(self, dtype: Any) -> dict[str, Any]:
        load_options: dict[str, Any] = {"torch_dtype": dtype}
        if not self.config.options.get("load_text_encoders", False):
            load_options.update(
                {
                    "text_encoder": None,
                    "text_encoder_2": None,
                    "tokenizer": None,
                    "tokenizer_2": None,
                }
            )
        quantization = self.config.options.get("quantization")
        if quantization is not None:
            if quantization != "bitsandbytes_8bit":
                raise ValueError(
                    f"Unsupported OmniPaint quantization mode: {quantization!r}."
                )
            from diffusers.quantizers import PipelineQuantizationConfig

            components = self.config.options.get(
                "quantization_components", ["transformer"]
            )
            if components != ["transformer"]:
                raise ValueError(
                    "OmniPaint INT8 currently supports only "
                    "quantization_components: [transformer]."
                )
            load_options["quantization_config"] = PipelineQuantizationConfig(
                quant_backend="bitsandbytes_8bit",
                quant_kwargs={"load_in_8bit": True},
                components_to_quantize=components,
            )
        return load_options

    @staticmethod
    def _install_diffusers_compat() -> None:
        """Bridge private Diffusers symbols moved after OmniPaint was released."""
        from diffusers.models.transformers import transformer_flux
        from diffusers.utils import (
            USE_PEFT_BACKEND,
            scale_lora_layers,
            unscale_lora_layers,
        )
        from diffusers.utils.torch_utils import is_torch_version

        compatibility_symbols = {
            "USE_PEFT_BACKEND": USE_PEFT_BACKEND,
            "scale_lora_layers": scale_lora_layers,
            "unscale_lora_layers": unscale_lora_layers,
            "is_torch_version": is_torch_version,
        }
        for name, value in compatibility_symbols.items():
            if not hasattr(transformer_flux, name):
                setattr(transformer_flux, name, value)

    @staticmethod
    def _install_device_compat(condition_module: Any, flux_core_module: Any) -> None:
        """Use Diffusers' execution device when CPU offload is enabled."""
        def encode_images(pipeline: Any, images: Any):
            execution_device = getattr(
                pipeline, "_execution_device", pipeline.device
            )
            images = pipeline.image_processor.preprocess(images)
            images = images.to(
                device=execution_device,
                dtype=pipeline.vae.dtype,
            )
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
                images.dtype,
            )
            if images_tokens.shape[1] != images_ids.shape[0]:
                images_ids = pipeline._prepare_latent_image_ids(
                    images.shape[0],
                    images.shape[2] // 2,
                    images.shape[3] // 2,
                    execution_device,
                    images.dtype,
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
        # Keep static embeddings on CPU while idle. They are moved to the
        # current execution device only for one generation call.
        prompt = torch.from_numpy(data["prompt_embeds"]).to(dtype=dtype)
        pooled = torch.from_numpy(data["pooled_prompt_embeds"]).to(dtype=dtype)
        ids = data["text_ids"]
        if ids.ndim == 3 and ids.shape[0] == 1:
            ids = ids[0]
        text_ids = torch.as_tensor(ids, dtype=torch.long)
        return prompt, pooled, text_ids

    def _validate_target_size(self, image: ImageArray) -> None:
        max_side = int(self.config.options.get("max_image_side", 1024))
        actual = max(image.shape[:2])
        if actual > max_side:
            raise ValueError(
                f"OmniPaint input side is {actual}px, exceeding its configured "
                f"max_image_side={max_side}. Preprocess the target image with "
                f"max_side={max_side} before removal or reference insertion."
            )
