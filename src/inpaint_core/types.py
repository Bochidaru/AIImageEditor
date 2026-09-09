from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, TypeAlias

import numpy as np
from numpy.typing import NDArray

ImageArray: TypeAlias = NDArray[np.uint8]
MaskArray: TypeAlias = NDArray[np.uint8]


@dataclass(frozen=True, slots=True)
class PointPrompt:
    points: list[tuple[float, float]]
    labels: list[int]

    def __post_init__(self) -> None:
        if not self.points:
            raise ValueError("At least one point is required.")
        if len(self.points) != len(self.labels):
            raise ValueError("points and labels must have equal lengths.")
        if any(label not in (0, 1) for label in self.labels):
            raise ValueError("Point labels must be 0 (negative) or 1 (positive).")


@dataclass(frozen=True, slots=True)
class BoxPrompt:
    x1: float
    y1: float
    x2: float
    y2: float

    def __post_init__(self) -> None:
        if self.x2 <= self.x1 or self.y2 <= self.y1:
            raise ValueError("A box must have positive width and height.")


SelectionPrompt: TypeAlias = PointPrompt | BoxPrompt


@dataclass(slots=True)
class MaskOptions:
    threshold: int = 127
    dilate: int = 0
    erode: int = 0
    candidate_index: int | None = None


@dataclass(slots=True)
class GenerationOptions:
    seed: int = 42
    # None lets each backend use its own appropriate default: Klein uses 4,
    # whereas FLUX Fill and OmniPaint normally use 28.
    num_inference_steps: int | None = None
    guidance_scale: float | None = None

    def __post_init__(self) -> None:
        if self.num_inference_steps is not None and self.num_inference_steps <= 0:
            raise ValueError("num_inference_steps must be greater than zero.")


@dataclass(frozen=True, slots=True)
class OutpaintMargins:
    left: int = 0
    top: int = 0
    right: int = 0
    bottom: int = 0

    def __post_init__(self) -> None:
        if min(self.left, self.top, self.right, self.bottom) < 0:
            raise ValueError("Outpaint margins cannot be negative.")
        if self.left + self.top + self.right + self.bottom == 0:
            raise ValueError("At least one outpaint margin must be positive.")


@dataclass(slots=True)
class UpscaleOptions:
    scale: int = 4
    face_enhance: bool = False
    tile: int = 0
    tile_pad: int = 10
    pre_pad: int = 0

    def __post_init__(self) -> None:
        if self.scale <= 0:
            raise ValueError("Upscale scale must be greater than zero.")


@dataclass(slots=True)
class SegmentationResult:
    masks: list[MaskArray]
    scores: list[float]
    best_index: int
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def best_mask(self) -> MaskArray:
        if not self.masks:
            raise ValueError("Segmentation returned no masks.")
        return self.masks[self.best_index]


@dataclass(slots=True)
class GenerationResult:
    image: ImageArray
    seed: int
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class EditResult:
    image: ImageArray
    mask: MaskArray
    generated_image: ImageArray | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class ImageTransform:
    image_size: tuple[int, int]
    padding: tuple[int, int, int, int]


@dataclass(slots=True)
class PreparedImage:
    image: ImageArray
    mask: MaskArray
    transform: ImageTransform
