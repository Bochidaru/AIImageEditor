from __future__ import annotations

from fastapi import Request

from ..processor import ImageProcessor
from .image_cache import ImageCache


def get_processor(request: Request) -> ImageProcessor:
    """FastAPI dependency returning the ImageProcessor stored on app.state.

    The processor is constructed once at app-startup time (see app.py /
    scripts/run_api.py) — either a real GPU-backed one via
    ImageProcessor.from_config(), or a lightweight fake for local frontend
    development. Building it lazily per-request would repeatedly reload
    multi-gigabyte model weights.
    """
    processor: ImageProcessor | None = getattr(request.app.state, "processor", None)
    if processor is None:
        raise RuntimeError(
            "No ImageProcessor configured on app.state. Build the app via "
            "create_app(processor=...)."
        )
    return processor


def get_image_cache(request: Request) -> ImageCache:
    """FastAPI dependency returning the shared ImageCache stored on app.state."""
    cache: ImageCache | None = getattr(request.app.state, "image_cache", None)
    if cache is None:
        raise RuntimeError("No ImageCache configured on app.state. Build the app via create_app(...).")
    return cache
