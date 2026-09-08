"""Validate runtime dependencies without downloading or loading model weights."""

from __future__ import annotations

import argparse
import importlib
import subprocess
import sys
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path


EXPECTED_VERSIONS = {
    "numpy": "1.26.4",
    "scipy": "1.16.3",
    "torch": "2.14.0",
    "torchvision": "0.29.0",
    "diffusers": "0.35.1",
    "transformers": "4.55.4",
    "tokenizers": "0.21.4",
    "peft": "0.17.1",
    "accelerate": "1.10.1",
    "huggingface-hub": "0.34.4",
    "safetensors": "0.6.2",
    "sentencepiece": "0.2.1",
    "bitsandbytes": "0.48.0",
    "realesrgan": "0.3.0",
    "basicsr": "1.4.2",
    "facexlib": "0.3.0",
    "gfpgan": "1.3.8",
}
EXPECTED_PYTHON = (3, 12)
EXPECTED_CUDA = "13.0"


def record(results: list[tuple[bool, str]], ok: bool, message: str) -> None:
    results.append((ok, message))
    print(f"[{'OK' if ok else 'FAIL'}] {message}")


def check_versions(results: list[tuple[bool, str]]) -> None:
    for package, expected in EXPECTED_VERSIONS.items():
        try:
            installed = version(package)
        except PackageNotFoundError:
            record(results, False, f"{package} is not installed")
            continue
        # CUDA wheels may append a local suffix, for example 2.14.0+cu130.
        matches = installed == expected or installed.startswith(f"{expected}+")
        record(
            results,
            matches,
            f"{package}={installed} (expected {expected})",
        )


def check_import(
    results: list[tuple[bool, str]], label: str, callback,
) -> None:
    try:
        callback()
    except Exception as exc:  # A preflight should report every failing import.
        record(results, False, f"{label}: {type(exc).__name__}: {exc}")
    else:
        record(results, True, label)


def check_diffusers_api() -> None:
    from diffusers import FluxFillPipeline, FluxKontextPipeline, FluxPipeline
    from diffusers.models.transformers.transformer_flux import (
        FluxTransformer2DModel,
        Transformer2DModelOutput,
        USE_PEFT_BACKEND,
        scale_lora_layers,
        unscale_lora_layers,
    )
    from diffusers.utils.torch_utils import is_torch_version
    from diffusers.pipelines.flux.pipeline_flux import (
        FluxPipelineOutput,
        calculate_shift,
        retrieve_timesteps,
    )
    from diffusers.quantizers import PipelineQuantizationConfig

    # Keep imported names live so linters and future refactors cannot silently
    # remove a compatibility check used by OmniPaint or the project backends.
    assert all(
        item is not None
        for item in (
            FluxFillPipeline,
            FluxKontextPipeline,
            FluxPipeline,
            FluxTransformer2DModel,
            Transformer2DModelOutput,
            USE_PEFT_BACKEND,
            is_torch_version,
            scale_lora_layers,
            unscale_lora_layers,
            FluxPipelineOutput,
            calculate_shift,
            retrieve_timesteps,
            PipelineQuantizationConfig,
        )
    )


def check_omnipaint_import(project_root: Path) -> None:
    source_root = project_root / "third_party" / "OmniPaint"
    if not (source_root / "src" / "generate.py").is_file():
        raise FileNotFoundError(f"OmniPaint source not found: {source_root}")
    source_string = str(source_root)
    if source_string not in sys.path:
        sys.path.insert(0, source_string)
    from inpaint_core.backends.omnipaint.backend import OmniPaintBackend

    OmniPaintBackend._install_diffusers_compat()
    condition = importlib.import_module("src.condition")
    generate = importlib.import_module("src.generate")
    flux_core = importlib.import_module("src.flux_core")
    assert condition.Condition is not None
    assert generate.generate is not None
    assert flux_core.tranformer_forward is not None


def check_sam2_import() -> None:
    from sam2.sam2_image_predictor import SAM2ImagePredictor

    assert SAM2ImagePredictor is not None


def check_realesrgan_import() -> None:
    from inpaint_core.backends.upscaling.realesrgan import RealESRGANBackend

    RealESRGANBackend._install_torchvision_compat()
    from basicsr.archs.rrdbnet_arch import RRDBNet
    from realesrgan import RealESRGANer

    assert RRDBNet is not None and RealESRGANer is not None


def check_cuda(results: list[tuple[bool, str]], require_cuda: bool) -> None:
    try:
        import torch
    except Exception as exc:
        record(results, False, f"PyTorch import: {type(exc).__name__}: {exc}")
        return

    available = torch.cuda.is_available()
    message = (
        f"CUDA available={available}, torch CUDA={torch.version.cuda}, "
        f"device={torch.cuda.get_device_name(0) if available else 'none'}"
    )
    record(results, available or not require_cuda, message)
    if available:
        cuda_matches = torch.version.cuda == EXPECTED_CUDA
        record(
            results,
            cuda_matches or not require_cuda,
            f"PyTorch CUDA runtime={torch.version.cuda} (expected {EXPECTED_CUDA})",
        )
        bf16_supported = torch.cuda.is_bf16_supported()
        record(results, bf16_supported, f"CUDA bfloat16 support={bf16_supported}")


def check_pip(results: list[tuple[bool, str]]) -> None:
    completed = subprocess.run(
        [sys.executable, "-m", "pip", "check"],
        capture_output=True,
        text=True,
        check=False,
    )
    output = (completed.stdout or completed.stderr).strip()
    record(results, completed.returncode == 0, f"pip check: {output}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--require-cuda",
        action="store_true",
        help="Fail when CUDA or bfloat16 support is unavailable.",
    )
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[1]
    results: list[tuple[bool, str]] = []
    print(f"Python: {sys.version.split()[0]}")
    print(f"Project: {project_root}")

    python_matches = sys.version_info[:2] == EXPECTED_PYTHON
    record(
        results,
        python_matches,
        f"Python {sys.version_info.major}.{sys.version_info.minor} "
        f"(expected {EXPECTED_PYTHON[0]}.{EXPECTED_PYTHON[1]})",
    )
    check_versions(results)
    check_import(results, "NumPy/SciPy/OpenCV imports", lambda: (
        importlib.import_module("numpy"),
        importlib.import_module("scipy"),
        importlib.import_module("cv2"),
    ))
    check_import(results, "FLUX/Diffusers API", check_diffusers_api)
    check_import(
        results,
        "OmniPaint private Diffusers API",
        lambda: check_omnipaint_import(project_root),
    )
    check_import(results, "SAM2 API", check_sam2_import)
    check_import(results, "Real-ESRGAN/BasicSR API", check_realesrgan_import)
    check_cuda(results, args.require_cuda)
    check_pip(results)

    failures = sum(not ok for ok, _ in results)
    print(f"\nResult: {len(results) - failures} passed, {failures} failed")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
