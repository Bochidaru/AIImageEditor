import sys
from types import ModuleType, SimpleNamespace

import pytest

from inpaint_core.config import DeviceConfig, ModelConfig
from inpaint_core.editing.flux_fill import FluxFillEditor
from inpaint_core.models.manager import ModelManager


class FakePipeline:
    model_id = None
    load_options = None

    def __init__(self):
        self.cpu_offload_enabled = False

    @classmethod
    def from_pretrained(cls, model_id, **load_options):
        cls.model_id = model_id
        cls.load_options = load_options
        return cls()

    def enable_model_cpu_offload(self):
        self.cpu_offload_enabled = True


class FakePipelineQuantizationConfig:
    def __init__(self, **kwargs):
        self.kwargs = kwargs


def install_fake_dependencies(monkeypatch):
    fake_torch = SimpleNamespace(bfloat16=object())
    fake_diffusers = ModuleType("diffusers")
    fake_diffusers.FluxFillPipeline = FakePipeline
    fake_quantizers = ModuleType("diffusers.quantizers")
    fake_quantizers.PipelineQuantizationConfig = FakePipelineQuantizationConfig

    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setitem(sys.modules, "diffusers", fake_diffusers)
    monkeypatch.setitem(sys.modules, "diffusers.quantizers", fake_quantizers)


def make_editor(options):
    return FluxFillEditor(
        ModelConfig(
            backend="flux_fill",
            model_id="black-forest-labs/FLUX.1-Fill-dev",
            options=options,
        ),
        DeviceConfig(name="cuda", dtype="bfloat16"),
        ModelManager(),
    )


def test_load_pipeline_quantizes_transformer_in_8bit(monkeypatch):
    install_fake_dependencies(monkeypatch)
    editor = make_editor(
        {
            "cpu_offload": True,
            "quantization": "bitsandbytes_8bit",
            "quantization_components": ["transformer"],
        }
    )

    pipeline = editor._load_pipeline()

    quantization = FakePipeline.load_options["quantization_config"]
    assert quantization.kwargs == {
        "quant_backend": "bitsandbytes_8bit",
        "quant_kwargs": {"load_in_8bit": True},
        "components_to_quantize": ["transformer"],
    }
    assert pipeline.cpu_offload_enabled is True


def test_load_pipeline_rejects_unknown_quantization(monkeypatch):
    install_fake_dependencies(monkeypatch)
    editor = make_editor({"quantization": "unknown"})

    with pytest.raises(ValueError, match="Unsupported Flux quantization mode"):
        editor._load_pipeline()
