import sys
from types import ModuleType, SimpleNamespace

import numpy as np
from PIL import Image

from inpaint_core.backends.flux2 import Flux2KleinBackend
from inpaint_core.config import DeviceConfig, ModelConfig
from inpaint_core.models.manager import ModelManager
from inpaint_core.types import GenerationOptions


class FakeGenerator:
    def __init__(self, device):
        self.device = device

    def manual_seed(self, seed):
        self.seed = seed
        return self


class FakePipeline:
    load_options = None

    def __init__(self):
        self.cpu_offload_enabled = False
        self.calls = []

    @classmethod
    def from_pretrained(cls, model_id, **kwargs):
        cls.load_options = kwargs
        return cls()

    def enable_model_cpu_offload(self):
        self.cpu_offload_enabled = True

    def enable_vae_tiling(self):
        pass

    def enable_vae_slicing(self):
        pass

    def __call__(self, **kwargs):
        self.calls.append(kwargs)
        size = (kwargs["width"], kwargs["height"])
        return SimpleNamespace(images=[Image.new("RGB", size, "white")])


def make_backend():
    return Flux2KleinBackend(
        ModelConfig(
            backend="flux2_klein",
            model_id="black-forest-labs/FLUX.2-klein-4B",
            options={
                "cpu_offload": True,
                "num_inference_steps": 4,
                "guidance_scale": 1.0,
            },
        ),
        DeviceConfig(name="cuda", dtype="bfloat16"),
        ModelManager(),
    )


def test_klein_loads_once_and_uses_distilled_defaults(monkeypatch):
    fake_torch = SimpleNamespace(bfloat16=object(), Generator=FakeGenerator)
    fake_diffusers = ModuleType("diffusers")
    fake_diffusers.Flux2KleinPipeline = FakePipeline
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setitem(sys.modules, "diffusers", fake_diffusers)

    backend = make_backend()
    image = np.zeros((65, 97, 3), dtype=np.uint8)
    result = backend.edit_image(image, "change the background", GenerationOptions())
    pipeline = backend.manager.get(backend.RESOURCE_KEY, backend._load_pipeline)

    assert result.image.shape == image.shape
    assert pipeline.calls[0]["height"] == 80
    assert pipeline.calls[0]["width"] == 112
    assert pipeline.calls[0]["num_inference_steps"] == 4
    assert pipeline.calls[0]["guidance_scale"] == 1.0
    assert pipeline.cpu_offload_enabled is True


def test_klein_text_to_image_honors_call_overrides(monkeypatch):
    fake_torch = SimpleNamespace(bfloat16=object(), Generator=FakeGenerator)
    fake_diffusers = ModuleType("diffusers")
    fake_diffusers.Flux2KleinPipeline = FakePipeline
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setitem(sys.modules, "diffusers", fake_diffusers)

    backend = make_backend()
    result = backend.generate(
        "a lake",
        128,
        96,
        GenerationOptions(seed=7, num_inference_steps=6, guidance_scale=1.2),
    )
    pipeline = backend.manager.get(backend.RESOURCE_KEY, backend._load_pipeline)

    assert result.image.shape == (96, 128, 3)
    assert pipeline.calls[0]["num_inference_steps"] == 6
    assert pipeline.calls[0]["guidance_scale"] == 1.2
