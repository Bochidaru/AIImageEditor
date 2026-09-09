import sys
from types import ModuleType

import numpy as np
import pytest

from inpaint_core.backends.omnipaint import OmniPaintBackend
from inpaint_core.config import DeviceConfig, ModelConfig
from inpaint_core.models.manager import ModelManager


def make_backend(max_side=1024):
    return OmniPaintBackend(
        ModelConfig(
            backend="omnipaint",
            model_id="yeates/OmniPaint",
            options={
                "max_image_side": max_side,
                "quantization": "bitsandbytes_8bit",
                "quantization_components": ["transformer"],
                "load_text_encoders": False,
            },
        ),
        DeviceConfig(name="cuda", dtype="bfloat16"),
        ModelManager(),
    )


def test_omnipaint_has_its_own_image_limit():
    backend = make_backend(1024)
    backend._validate_target_size(np.zeros((768, 1024, 3), dtype=np.uint8))

    with pytest.raises(ValueError, match="max_image_side=1024"):
        backend._validate_target_size(np.zeros((1025, 768, 3), dtype=np.uint8))


class FakePipelineQuantizationConfig:
    def __init__(self, **kwargs):
        self.kwargs = kwargs


def test_omnipaint_int8_load_options_skip_text_encoders(monkeypatch):
    quantizers = ModuleType("diffusers.quantizers")
    quantizers.PipelineQuantizationConfig = FakePipelineQuantizationConfig
    monkeypatch.setitem(sys.modules, "diffusers.quantizers", quantizers)
    backend = make_backend()
    dtype = object()

    options = backend._pipeline_load_options(dtype)

    assert options["torch_dtype"] is dtype
    assert options["text_encoder"] is None
    assert options["text_encoder_2"] is None
    assert options["quantization_config"].kwargs == {
        "quant_backend": "bitsandbytes_8bit",
        "quant_kwargs": {"load_in_8bit": True},
        "components_to_quantize": ["transformer"],
    }


def test_omnipaint_rejects_quantizing_non_transformer_components(monkeypatch):
    quantizers = ModuleType("diffusers.quantizers")
    quantizers.PipelineQuantizationConfig = FakePipelineQuantizationConfig
    monkeypatch.setitem(sys.modules, "diffusers.quantizers", quantizers)
    backend = make_backend()
    backend.config.options["quantization_components"] = ["transformer", "vae"]

    with pytest.raises(ValueError, match="only quantization_components"):
        backend._pipeline_load_options(object())


def test_omnipaint_installs_moved_diffusers_symbols(monkeypatch):
    transformer_flux = ModuleType(
        "diffusers.models.transformers.transformer_flux"
    )
    transformers = ModuleType("diffusers.models.transformers")
    transformers.transformer_flux = transformer_flux
    models = ModuleType("diffusers.models")
    models.transformers = transformers
    diffusers = ModuleType("diffusers")
    diffusers.models = models

    utils = ModuleType("diffusers.utils")
    utils.USE_PEFT_BACKEND = object()
    utils.scale_lora_layers = lambda *args: None
    utils.unscale_lora_layers = lambda *args: None
    torch_utils = ModuleType("diffusers.utils.torch_utils")
    torch_utils.is_torch_version = lambda *args: True
    utils.torch_utils = torch_utils

    monkeypatch.setitem(sys.modules, "diffusers", diffusers)
    monkeypatch.setitem(sys.modules, "diffusers.models", models)
    monkeypatch.setitem(sys.modules, "diffusers.models.transformers", transformers)
    monkeypatch.setitem(
        sys.modules,
        "diffusers.models.transformers.transformer_flux",
        transformer_flux,
    )
    monkeypatch.setitem(sys.modules, "diffusers.utils", utils)
    monkeypatch.setitem(sys.modules, "diffusers.utils.torch_utils", torch_utils)

    OmniPaintBackend._install_diffusers_compat()

    assert transformer_flux.USE_PEFT_BACKEND is utils.USE_PEFT_BACKEND
    assert transformer_flux.scale_lora_layers is utils.scale_lora_layers
    assert transformer_flux.unscale_lora_layers is utils.unscale_lora_layers
    assert transformer_flux.is_torch_version is torch_utils.is_torch_version
