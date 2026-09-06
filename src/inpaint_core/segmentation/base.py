from __future__ import annotations

from typing import Protocol

from ..types import ImageArray, SegmentationResult, SelectionPrompt


class Segmenter(Protocol):
    def segment(
        self,
        image: ImageArray,
        selection: SelectionPrompt,
    ) -> SegmentationResult:
        ...

