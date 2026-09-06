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
    feather: int = 6
    crop_padding: int = 128
    candidate_index: int | None = None


@dataclass(slots=True)
class GenerationOptions:
    seed: int = 42
    num_inference_steps: int = 28
    guidance_scale: float | None = None


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
class CropTransform:
    crop_box: tuple[int, int, int, int]
    crop_size: tuple[int, int]
    processing_size: tuple[int, int]
    padding: tuple[int, int, int, int]
    original_size: tuple[int, int]


@dataclass(slots=True)
class PreparedCrop:
    image: ImageArray
    mask: MaskArray
    transform: CropTransform

