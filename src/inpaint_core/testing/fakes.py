"""Fake segmenter/backend + a fully-wired fake ImageProcessor.

The single source of truth for fake ImageProcessor backends — used by both
tests/test_processor.py (library-level tests) and tests/test_api.py /
`scripts/run_api.py --fake` (HTTP-level tests and local frontend dev), so the
two layers can't silently drift into non-equivalent fake behavior.
"""

from __future__ import annotations

import numpy as np

from ..config import AppConfig
from ..processor import ImageProcessor
from ..types import GenerationResult, ImageArray, SegmentationResult


class FakeSegmenter:
    """Returns a caller-supplied mask (or an auto-generated centered blob)."""

    def __init__(self, mask: np.ndarray | None = None, *, score: float = 1.0) -> None:
        self.mask = mask
        self.score = score

    def segment(self, image, selection) -> SegmentationResult:  # noqa: ANN001 - protocol match
        mask = self.mask
        if mask is None:
            height, width = image.shape[:2]
            mask = np.zeros((height, width), dtype=np.uint8)
            y0, y1 = int(height * 0.25), int(height * 0.75)
            x0, x1 = int(width * 0.25), int(width * 0.75)
            mask[y0:y1, x0:x1] = 255
        return SegmentationResult([mask], [self.score], 0, {"backend": "fake"})


class FakeBackend:
    """Implements every backend protocol with a solid-color stand-in image.

    Tracks the last call's mask/prompt/image-shape/reference so tests can
    assert on what each operation forwarded to its backend.
    """

    def __init__(self, *, tint: int = 200) -> None:
        self.tint = tint
        self.last_mask = None
        self.last_prompt = None
        self.last_image_shape = None
        self.last_reference = None

    def _result(self, image: ImageArray, options) -> GenerationResult:  # noqa: ANN001
        self.last_image_shape = image.shape
        return GenerationResult(np.full_like(image, self.tint), options.seed, {"backend": "fake"})

    def edit(self, image, mask, prompt, options):  # noqa: ANN001, D401
        self.last_mask, self.last_prompt = mask, prompt
        return self._result(image, options)

    def remove(self, image, mask, options):  # noqa: ANN001
        self.last_mask = mask
        return self._result(image, options)

    def insert(self, image, mask, reference, options):  # noqa: ANN001
        self.last_mask, self.last_reference = mask, reference
        return self._result(image, options)

    def edit_image(self, image, prompt, options):  # noqa: ANN001
        self.last_prompt = prompt
        return self._result(image, options)

    def generate(self, prompt, width, height, options):  # noqa: ANN001
        self.last_prompt = prompt
        image = np.full((height, width, 3), self.tint, dtype=np.uint8)
        return GenerationResult(image, options.seed, {"backend": "fake"})

    def upscale(self, image, options):  # noqa: ANN001
        output = np.repeat(np.repeat(image, options.scale, axis=0), options.scale, axis=1)
        return GenerationResult(output, 0, {"backend": "fake"})


def build_fake_processor(*, max_image_side: int = 2048) -> ImageProcessor:
    """A fully-wired ImageProcessor backed entirely by in-memory fakes."""
    config = AppConfig.from_dict(
        {
            "device": {"name": "cpu", "dtype": "float32"},
            "models": {"segmentation": {"backend": "fake"}},
            "processing": {"max_image_side": max_image_side, "size_multiple": 16},
        }
    )
    segmenter = FakeSegmenter(score=0.97)
    backend = FakeBackend()
    return ImageProcessor(
        config=config,
        segmenter=segmenter,
        flux_fill=backend,
        omnipaint=backend,
        flux2_klein=backend,
        upscaler=backend,
    )
