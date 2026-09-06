import sys
from contextlib import nullcontext
from types import SimpleNamespace

import numpy as np

from inpaint_core.config import DeviceConfig, ModelConfig
from inpaint_core.models.manager import ModelManager
from inpaint_core.segmentation.sam2 import SAM2Segmenter
from inpaint_core.types import BoxPrompt, PointPrompt


class FakeSAM2Predictor:
    def __init__(self):
        self.image = None
        self.arguments = None

    def set_image(self, image):
        self.image = image

    def predict(self, **kwargs):
        self.arguments = kwargs
        height, width = self.image.shape[:2]
        masks = np.zeros((3, height, width), dtype=bool)
        masks[1, 2:6, 3:7] = True
        scores = np.asarray([0.2, 0.9, 0.4], dtype=np.float32)
        logits = np.zeros((3, 256, 256), dtype=np.float32)
        return masks, scores, logits


def make_segmenter(predictor):
    manager = ModelManager(memory_policy="resident")
    manager.get("sam2", lambda: predictor)
    return SAM2Segmenter(
        ModelConfig(
            backend="sam2",
            model_id="facebook/sam2.1-hiera-base-plus",
            options={"multimask_output": True},
        ),
        DeviceConfig(name="cpu", dtype="float32"),
        manager,
    )


def install_fake_torch(monkeypatch):
    fake_torch = SimpleNamespace(inference_mode=lambda: nullcontext())
    monkeypatch.setitem(sys.modules, "torch", fake_torch)


def test_segment_with_point_prompt_normalizes_sam2_output(monkeypatch):
    install_fake_torch(monkeypatch)
    predictor = FakeSAM2Predictor()
    segmenter = make_segmenter(predictor)
    image = np.zeros((10, 12, 3), dtype=np.uint8)

    result = segmenter.segment(
        image,
        PointPrompt(points=[(4.0, 5.0), (8.0, 2.0)], labels=[1, 0]),
    )

    np.testing.assert_array_equal(
        predictor.arguments["point_coords"],
        np.asarray([[4.0, 5.0], [8.0, 2.0]], dtype=np.float32),
    )
    np.testing.assert_array_equal(
        predictor.arguments["point_labels"], np.asarray([1, 0], dtype=np.int32)
    )
    assert predictor.arguments["box"] is None
    assert predictor.arguments["multimask_output"] is True
    assert result.best_index == 1
    assert result.best_mask.dtype == np.uint8
    assert set(np.unique(result.best_mask)) == {0, 255}


def test_segment_with_box_prompt_uses_xyxy_coordinates(monkeypatch):
    install_fake_torch(monkeypatch)
    predictor = FakeSAM2Predictor()
    segmenter = make_segmenter(predictor)
    image = np.zeros((10, 12, 3), dtype=np.uint8)

    segmenter.segment(image, BoxPrompt(1.0, 2.0, 9.0, 8.0))

    np.testing.assert_array_equal(
        predictor.arguments["box"],
        np.asarray([1.0, 2.0, 9.0, 8.0], dtype=np.float32),
    )
    assert predictor.arguments["point_coords"] is None
    assert predictor.arguments["point_labels"] is None
