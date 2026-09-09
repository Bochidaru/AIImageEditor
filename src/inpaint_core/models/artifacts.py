"""Download model artifacts without constructing inference pipelines."""

from __future__ import annotations

import logging
import os
import urllib.request
from pathlib import Path
from urllib.parse import urlparse

from ..config import AppConfig, ModelConfig

logger = logging.getLogger(__name__)


def prefetch_model_assets(config: AppConfig) -> dict[str, str]:
    """Download configured checkpoints and replace remote IDs with local paths."""
    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise ImportError(
            "huggingface-hub is required when artifacts.prefetch_on_init is enabled."
        ) from exc

    downloaded: dict[str, str] = {}
    snapshots: dict[tuple[str, tuple[str, ...]], str] = {}
    cache_dir = config.artifacts.cache_dir

    for name, model in {
        "sam2": config.segmentation,
        "flux_fill": config.flux_fill,
        "flux2_klein": config.flux2_klein,
    }.items():
        local = _snapshot(model.model_id, snapshot_download, cache_dir, snapshots)
        if local is not None:
            _use_local_model(model, local)
            downloaded[name] = local

    base_model_id = config.omnipaint.options.get(
        "base_model_id", "black-forest-labs/FLUX.1-dev"
    )
    omni_base_patterns = None
    if not config.omnipaint.options.get("load_text_encoders", False):
        omni_base_patterns = [
            "model_index.json",
            "scheduler/**",
            "transformer/**",
            "vae/**",
        ]
    local_base = _snapshot(
        base_model_id,
        snapshot_download,
        cache_dir,
        snapshots,
        allow_patterns=omni_base_patterns,
    )
    if local_base is not None:
        config.omnipaint.options.setdefault("source_base_model_id", base_model_id)
        config.omnipaint.options["base_model_id"] = local_base
        downloaded["omnipaint_base"] = local_base

    if config.omnipaint.model_id:
        patterns = [
            config.omnipaint.options.get(
                "removal_lora", "weights/omnipaint_remove.safetensors"
            ),
            config.omnipaint.options.get(
                "insertion_lora", "weights/omnipaint_insert.safetensors"
            ),
            config.omnipaint.options.get(
                "removal_embedding", "embeddings/remove.npz"
            ),
            config.omnipaint.options.get(
                "insertion_embedding", "embeddings/insert.npz"
            ),
        ]
        omni_local = snapshot_download(
            repo_id=config.omnipaint.model_id,
            cache_dir=cache_dir,
            allow_patterns=patterns,
        )
        _use_local_model(config.omnipaint, omni_local)
        downloaded["omnipaint_adapters"] = omni_local

    checkpoint = _prefetch_realesrgan(config)
    if checkpoint is not None:
        downloaded["realesrgan"] = checkpoint
    return downloaded


def _snapshot(
    model_id,
    snapshot_download,
    cache_dir,
    snapshots,
    *,
    allow_patterns=None,
):
    if not model_id:
        return None
    local_candidate = Path(model_id).expanduser()
    if local_candidate.exists():
        return str(local_candidate.resolve())
    key = (model_id, tuple(allow_patterns or ()))
    if key not in snapshots:
        logger.info("Downloading model artifacts: %s", model_id)
        snapshots[key] = snapshot_download(
            repo_id=model_id,
            cache_dir=cache_dir,
            allow_patterns=allow_patterns,
        )
    return snapshots[key]


def _use_local_model(model: ModelConfig, local_path: str) -> None:
    model.options.setdefault("source_model_id", model.model_id)
    model.model_id = local_path


def _prefetch_realesrgan(config: AppConfig) -> str | None:
    if config.upscaler.checkpoint:
        checkpoint = Path(config.upscaler.checkpoint).expanduser().resolve()
        if not checkpoint.is_file():
            raise FileNotFoundError(f"Real-ESRGAN checkpoint not found: {checkpoint}")
        config.upscaler.checkpoint = str(checkpoint)
        return str(checkpoint)

    url = config.upscaler.options.get(
        "model_url",
        "https://github.com/xinntao/Real-ESRGAN/releases/download/"
        "v0.1.0/RealESRGAN_x4plus.pth",
    )
    weights_dir = Path(config.artifacts.weights_dir).expanduser().resolve()
    weights_dir.mkdir(parents=True, exist_ok=True)
    destination = weights_dir / Path(urlparse(url).path).name
    if not destination.is_file():
        temporary = destination.with_suffix(destination.suffix + ".download")
        logger.info("Downloading Real-ESRGAN checkpoint: %s", url)
        try:
            urllib.request.urlretrieve(url, temporary)
            os.replace(temporary, destination)
        finally:
            if temporary.exists():
                temporary.unlink()
    config.upscaler.checkpoint = str(destination)
    return str(destination)
