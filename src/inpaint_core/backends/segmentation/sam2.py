from __future__ import annotations

from contextlib import ExitStack
from typing import Any

import numpy as np

from ...config import DeviceConfig, ModelConfig
from ...models.manager import ModelManager
from ...types import BoxPrompt, ImageArray, PointPrompt, SegmentationResult, SelectionPrompt


class SAM2Segmenter:
    RESOURCE_KEY = "sam2"

    def __init__(self, config: ModelConfig, device: DeviceConfig, manager: ModelManager):
        self.config = config
        self.device = device
        self.manager = manager

    def segment(self, image: ImageArray, selection: SelectionPrompt) -> SegmentationResult:
        self._validate_image(image)
        predictor = self.manager.get(self.RESOURCE_KEY, self._load_predictor)
        try:
            import torch
        except ImportError as exc:
            raise ImportError("PyTorch is required to run SAM2.") from exc

        with ExitStack() as stack:
            stack.enter_context(torch.inference_mode())
            if self.device.name.split(":", 1)[0] == "cuda":
                if not torch.cuda.is_available():
                    raise RuntimeError("SAM2 is configured for CUDA, but CUDA is unavailable.")
                stack.enter_context(torch.autocast("cuda", dtype=self._torch_dtype(torch)))
            predictor.set_image(image)
            masks, scores, logits = predictor.predict(
                **self._prediction_args(selection),
                multimask_output=bool(self.config.options.get("multimask_output", True)),
            )

        masks = np.asarray(masks)
        if masks.ndim == 2:
            masks = masks[None, ...]
        scores = np.asarray(scores, dtype=np.float32).reshape(-1)
        if masks.ndim != 3 or masks.shape[0] == 0 or scores.size != masks.shape[0]:
            raise RuntimeError("SAM2 returned incompatible masks and scores.")
        normalized = [np.ascontiguousarray(item.astype(bool).astype(np.uint8) * 255) for item in masks]
        return SegmentationResult(
            normalized,
            scores.astype(float).tolist(),
            int(np.argmax(scores)),
            {
                "backend": "sam2",
                "model_id": self.config.model_id,
                "multimask_output": bool(self.config.options.get("multimask_output", True)),
                "low_res_logits_shape": list(np.asarray(logits).shape),
            },
        )

    def _load_predictor(self) -> Any:
        if not self.config.model_id:
            raise ValueError("models.segmentation.model_id is required.")
        try:
            from sam2.sam2_image_predictor import SAM2ImagePredictor
        except ImportError as exc:
            raise ImportError("Install Meta's `sam2` package before using SAM2.") from exc
        return SAM2ImagePredictor.from_pretrained(self.config.model_id, device=self.device.name)

    @staticmethod
    def _prediction_args(selection: SelectionPrompt) -> dict[str, np.ndarray | None]:
        if isinstance(selection, PointPrompt):
            return {
                "point_coords": np.asarray(selection.points, dtype=np.float32),
                "point_labels": np.asarray(selection.labels, dtype=np.int32),
                "box": None,
            }
        if isinstance(selection, BoxPrompt):
            return {
                "point_coords": None,
                "point_labels": None,
                "box": np.asarray([selection.x1, selection.y1, selection.x2, selection.y2], dtype=np.float32),
            }
        raise TypeError(f"Unsupported SAM2 selection: {type(selection).__name__}.")

    def _torch_dtype(self, torch: Any) -> Any:
        try:
            return {"float16": torch.float16, "bfloat16": torch.bfloat16}[self.device.dtype]
        except KeyError as exc:
            raise ValueError("CUDA autocast dtype must be float16 or bfloat16.") from exc

    @staticmethod
    def _validate_image(image: ImageArray) -> None:
        if not isinstance(image, np.ndarray):
            raise TypeError("SAM2 expects a NumPy image.")
        if image.ndim != 3 or image.shape[2] != 3 or image.dtype != np.uint8:
            raise ValueError("SAM2 expects RGB uint8 with shape (H, W, 3).")
