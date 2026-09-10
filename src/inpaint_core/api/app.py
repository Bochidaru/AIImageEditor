from __future__ import annotations

import logging
import os

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from ..processor import ImageProcessor
from .image_cache import ImageCache
from .routes import router

logger = logging.getLogger(__name__)

# No explicit allowlist configured: default to any localhost port rather
# than a single hardcoded one. Next.js silently picks 3001/3002/3005/... when
# its default port is already taken, and a single hardcoded default just
# turns into a CORS failure with no indication that the port is the problem.
DEFAULT_CORS_ORIGIN_REGEX = r"^http://localhost:\d+$"


def _cors_kwargs() -> dict:
    raw = os.environ.get("INPAINT_API_CORS_ORIGINS")
    if raw:
        origins = [origin.strip() for origin in raw.split(",") if origin.strip()]
        return {"allow_origins": origins}
    return {"allow_origins": [], "allow_origin_regex": DEFAULT_CORS_ORIGIN_REGEX}


def _warn_if_multi_worker() -> None:
    """ImageCache (see image_cache.py) lives in this process's memory only.
    Behind multiple worker processes, an image_id registered on one worker
    is invisible to the others, and a request that lands on a different
    worker fails with a confusing "Unknown or expired image_id" instead of
    an obvious deployment error — warn loudly at startup instead of letting
    that surprise whoever deploys this beyond a single process.
    """
    for var in ("WEB_CONCURRENCY", "UVICORN_WORKERS", "GUNICORN_WORKERS"):
        raw = os.environ.get(var)
        if raw and raw.strip().isdigit() and int(raw) > 1:
            logger.warning(
                "%s=%s: this API is running with more than one worker process, but "
                "ImageCache (api/image_cache.py) is in-process memory only. A client "
                "that registers an image via POST /api/images on one worker will get "
                "'Unknown or expired image_id' errors from requests that land on a "
                "different worker. Run with a single worker, or replace ImageCache "
                "with a shared store (e.g. Redis) before scaling horizontally.",
                var,
                raw,
            )
            return


def create_app(processor: ImageProcessor) -> FastAPI:
    """Build the FastAPI app around an already-constructed ImageProcessor.

    Callers choose the processor: a real one from ImageProcessor.from_config()
    for GPU inference, or a lightweight fake for local frontend development
    (see scripts/run_api.py --fake).
    """
    _warn_if_multi_worker()

    app = FastAPI(title="Inpaint Core API", version="0.1.0")
    app.state.processor = processor
    app.state.image_cache = ImageCache()

    app.add_middleware(
        CORSMiddleware,
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
        **_cors_kwargs(),
    )

    # ValueError/IndexError (DecodeError is a ValueError subclass) are how
    # this codebase signals bad *user input* — see operations/common.py,
    # masks/operations.py, codec.py. TypeError is deliberately NOT mapped
    # here: nothing in the request-handling path raises it for user input:
    # it only ever means a real internal bug, so it should propagate as an
    # unhandled 500 (visible to error monitoring) instead of masquerading
    # as a 400 that gets dismissed as the client's fault.
    @app.exception_handler(ValueError)
    @app.exception_handler(IndexError)
    async def _domain_error_handler(_: Request, exc: Exception) -> JSONResponse:
        return JSONResponse(status_code=400, content={"detail": str(exc)})

    app.include_router(router)
    return app
