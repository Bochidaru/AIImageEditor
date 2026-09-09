"""Run the inpaint_core HTTP API.

Two modes:
  --fake            In-memory fake backends (no GPU, no checkpoints, no
                     network). This is what the Next.js frontend talks to
                     during local development — see frontend/README or
                     NEXT_PUBLIC_API_URL in frontend/.env.local.
  --config PATH      Real ImageProcessor.from_config(PATH) (defaults to
                     configs/default.yaml). Requires CUDA, downloaded
                     checkpoints, and (for gated models) HF_TOKEN.

Examples:
    python scripts/run_api.py --fake
    python scripts/run_api.py --config configs/default.yaml
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from inpaint_core.api import create_app  # noqa: E402
from inpaint_core.processor import ImageProcessor  # noqa: E402
from inpaint_core.testing import build_fake_processor  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--fake", action="store_true", help="Use in-memory fake backends instead of real models.")
    parser.add_argument(
        "--config",
        default=str(ROOT / "configs" / "default.yaml"),
        help="Path to the AppConfig YAML (ignored when --fake is set).",
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.fake:
        print("Starting inpaint_core API with FAKE backends (no GPU, no checkpoints).")
        processor: ImageProcessor = build_fake_processor()
    else:
        print(f"Starting inpaint_core API from config: {args.config}")
        processor = ImageProcessor.from_config(args.config)

    app = create_app(processor)

    import uvicorn

    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
