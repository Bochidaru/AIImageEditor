from __future__ import annotations

from contextlib import ExitStack
from typing import Any

import numpy as np

from ..config import DeviceConfig, ModelConfig
from ..models.manager import ModelManager
from ..types import (
    BoxPrompt,
    ImageArray,
    PointPrompt,
    SegmentationResult,
    SelectionPrompt,
)


class SAM2Segmenter:
    """Segment a still image with SAM2 loaded from Hugging Face."""

    RESOURCE_KEY = "sam2"

    def __init__(
        self,
        config: ModelConfig,
        device: DeviceConfig,
        model_manager: ModelManager,
    ) -> None:
        self.config = config
        self.device = device
        self.model_manager = model_manager
        
    def _load_predictor(self) -> Any:
        """Download/load the configured checkpoint and place it on the device."""
        if not self.config.model_id:
            raise ValueError(
                "models.segmentation.model_id is required for the SAM2 backend."
            )

        try:
            from sam2.sam2_image_predictor import SAM2ImagePredictor
        except ImportError as exc:
            raise ImportError(
                "SAM2 is not installed. Install Meta's `sam2` package before "
                "using the SAM2 segmentation backend."
            ) from exc

        return SAM2ImagePredictor.from_pretrained(
            self.config.model_id,
            device=self.device.name,
        )

    def segment(
        self,
        image: ImageArray,
        selection: SelectionPrompt,
    ) -> SegmentationResult:
        """Embed ``image`` and predict masks from a point or box prompt."""
        self._validate_image(image)
        predictor = self.model_manager.get(self.RESOURCE_KEY, self._load_predictor)
        prediction_args = self._prediction_args(selection)
        multimask_output = bool(
            self.config.options.get("multimask_output", True)
        )

        try:
            import torch
        except ImportError as exc:
            raise ImportError("PyTorch is required to run SAM2.") from exc

        with ExitStack() as stack:
            stack.enter_context(torch.inference_mode())
            if self.device.name.split(":", maxsplit=1)[0] == "cuda":
                if not torch.cuda.is_available():
                    raise RuntimeError(
                        "SAM2 is configured for CUDA, but CUDA is not available."
                    )
                stack.enter_context(
                    torch.autocast(
                        device_type="cuda",
                        dtype=self._torch_dtype(torch),
                    )
                )

            predictor.set_image(image)
            masks, scores, low_res_logits = predictor.predict(
                **prediction_args,
                multimask_output=multimask_output,
            )

        mask_array = np.asarray(masks)
        if mask_array.ndim == 2:
            mask_array = mask_array[np.newaxis, ...]
        if mask_array.ndim != 3 or mask_array.shape[0] == 0:
            raise RuntimeError(
                f"SAM2 returned an invalid mask shape: {mask_array.shape}."
            )

        score_array = np.atleast_1d(np.asarray(scores, dtype=np.float32)).reshape(-1)
        if score_array.size != mask_array.shape[0]:
            raise RuntimeError(
                "SAM2 returned a different number of scores and masks."
            )

        normalized_masks = [
            np.ascontiguousarray(mask.astype(bool).astype(np.uint8) * 255)
            for mask in mask_array
        ]
        best_index = int(np.argmax(score_array))

        return SegmentationResult(
            masks=normalized_masks,
            scores=score_array.astype(float).tolist(),
            best_index=best_index,
            metadata={
                "backend": "sam2",
                "model_id": self.config.model_id,
                "multimask_output": multimask_output,
                "low_res_logits_shape": list(np.asarray(low_res_logits).shape),
            },
        )

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
                "box": np.asarray(
                    [selection.x1, selection.y1, selection.x2, selection.y2],
                    dtype=np.float32,
                ),
            }
        raise TypeError(f"Unsupported SAM2 selection type: {type(selection).__name__}.")

    def _torch_dtype(self, torch: Any) -> Any:
        supported = {
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
        }
        try:
            return supported[self.device.dtype]
        except KeyError as exc:
            raise ValueError(
                "CUDA autocast dtype must be 'float16' or 'bfloat16', "
                f"not {self.device.dtype!r}."
            ) from exc

    @staticmethod
    def _validate_image(image: ImageArray) -> None:
        if not isinstance(image, np.ndarray):
            raise TypeError("SAM2 expects a NumPy image. Run preprocess_image first.")
        if image.ndim != 3 or image.shape[2] != 3 or image.dtype != np.uint8:
            raise ValueError(
                "SAM2 expects an RGB uint8 image with shape (height, width, 3)."
            )
