from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass(slots=True)
class DeviceConfig:
    name: str = "cuda"
    dtype: str = "bfloat16"


@dataclass(slots=True)
class ModelConfig:
    backend: str
    model_id: str | None = None
    checkpoint: str | None = None
    options: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class MemoryConfig:
    policy: str = "sequential"
    track_peak: bool = True


@dataclass(slots=True)
class ProcessingConfig:
    max_image_side: int = 2048
    size_multiple: int = 16


@dataclass(slots=True)
class AppConfig:
    device: DeviceConfig
    segmentation: ModelConfig
    flux_fill: ModelConfig
    flux_generation: ModelConfig
    omnipaint: ModelConfig
    flux_kontext: ModelConfig
    upscaler: ModelConfig
    memory: MemoryConfig = field(default_factory=MemoryConfig)
    processing: ProcessingConfig = field(default_factory=ProcessingConfig)
    mask_defaults: dict[str, Any] = field(default_factory=dict)
    generation_defaults: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_yaml(cls, path: str | Path) -> "AppConfig":
        config_path = Path(path).resolve()
        with config_path.open("r", encoding="utf-8") as stream:
            payload = yaml.safe_load(stream) or {}
        config = cls.from_dict(payload)
        source_path = config.omnipaint.options.get("source_path")
        if source_path and not Path(source_path).expanduser().is_absolute():
            config.omnipaint.options["source_path"] = str(
                (config_path.parent / source_path).resolve()
            )
        return config

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "AppConfig":
        models = payload.get("models", {})
        return cls(
            device=DeviceConfig(**payload.get("device", {})),
            segmentation=_model_config(models.get("segmentation", {})),
            flux_fill=_model_config(
                models.get("flux_fill", models.get("editing", {}))
            ),
            flux_generation=_model_config(models.get("flux_generation", {})),
            omnipaint=_model_config(models.get("omnipaint", {})),
            flux_kontext=_model_config(models.get("flux_kontext", {})),
            upscaler=_model_config(models.get("upscaler", {})),
            memory=MemoryConfig(**payload.get("memory", {})),
            processing=ProcessingConfig(**payload.get("processing", {})),
            mask_defaults=payload.get("mask", {}),
            generation_defaults=payload.get("generation", {}),
        )


def _model_config(payload: dict[str, Any]) -> ModelConfig:
    known = {"backend", "model_id", "checkpoint"}
    options = {key: value for key, value in payload.items() if key not in known}
    return ModelConfig(
        backend=payload.get("backend", "unconfigured"),
        model_id=payload.get("model_id"),
        checkpoint=payload.get("checkpoint"),
        options=options,
    )
