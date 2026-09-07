from __future__ import annotations

from typing import Any

from ...config import DeviceConfig, ModelConfig


def pipeline_load_options(config: ModelConfig, device: DeviceConfig) -> dict[str, Any]:
    import torch

    try:
        dtype = getattr(torch, device.dtype)
    except AttributeError as exc:
        raise ValueError(f"Unsupported torch dtype: {device.dtype!r}.") from exc

    result: dict[str, Any] = {"torch_dtype": dtype}
    quantization = config.options.get("quantization")
    if quantization is None:
        return result
    if quantization != "bitsandbytes_8bit":
        raise ValueError(f"Unsupported Flux quantization mode: {quantization!r}.")

    from diffusers.quantizers import PipelineQuantizationConfig

    components = config.options.get("quantization_components", ["transformer"])
    if not isinstance(components, list) or not components:
        raise ValueError("quantization_components must be a non-empty list.")
    result["quantization_config"] = PipelineQuantizationConfig(
        quant_backend="bitsandbytes_8bit",
        quant_kwargs={"load_in_8bit": True},
        components_to_quantize=components,
    )
    return result


def place_pipeline(pipeline: Any, config: ModelConfig, device: DeviceConfig) -> Any:
    if config.options.get("cpu_offload", False):
        pipeline.enable_model_cpu_offload()
    else:
        pipeline.to(device.name)
    return pipeline


def generator_for(seed: int):
    import torch

    return torch.Generator("cpu").manual_seed(seed)
