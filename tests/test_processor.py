import numpy as np
import pytest

from inpaint_core.config import AppConfig
from inpaint_core.processor import ImageProcessor
from inpaint_core.testing import FakeBackend, FakeSegmenter
from inpaint_core.types import BoxPrompt, MaskOptions, OutpaintMargins, PointPrompt, UpscaleOptions


def make_processor(mask):
    config = AppConfig.from_dict({
        "device": {"name": "cpu", "dtype": "float32"},
        "models": {"segmentation": {"backend": "fake"}},
        "processing": {"max_image_side": 128, "size_multiple": 16},
    })
    segmenter = FakeSegmenter(mask)
    backend = FakeBackend()
    processor = ImageProcessor(
        config=config, segmenter=segmenter, flux_fill=backend,
        omnipaint=backend, flux2_klein=backend,
        upscaler=backend,
    )
    return processor, backend


def test_remove_uses_omnipaint_and_returns_generated_full_image():
    image = np.zeros((64, 64, 3), np.uint8)
    mask = np.zeros((64, 64), np.uint8)
    mask[20:40, 20:40] = 255
    processor, backend = make_processor(mask)
    result = processor.remove_object(image, mask=mask, mask_options=MaskOptions(dilate=0))
    assert np.all(result.image == 200)
    assert result.metadata["compositing"] is False
    assert backend.last_mask[30, 30] == 255


def test_background_replacement_uses_direct_klein_edit():
    image = np.zeros((64, 64, 3), np.uint8)
    foreground = np.zeros((64, 64), np.uint8)
    foreground[20:40, 20:40] = 255
    processor, backend = make_processor(foreground)
    result = processor.replace_background(image, "a beach")
    assert result.image.shape == image.shape
    assert "Replace only the background with a beach" in backend.last_prompt
    assert "Preserve the main foreground subject exactly" in backend.last_prompt


def test_replacement_forwards_prompt_and_pads_full_image():
    image = np.zeros((77, 113, 3), np.uint8)
    mask = np.zeros((77, 113), np.uint8)
    mask[30:40, 50:60] = 255
    processor, backend = make_processor(mask)
    result = processor.replace_object(image, "a vase", mask=mask, mask_options=MaskOptions())
    assert backend.last_prompt == "a vase"
    assert backend.last_image_shape == (80, 128, 3)
    assert result.image.shape == image.shape


def test_both_object_insertion_modes():
    image = np.zeros((64, 64, 3), np.uint8)
    reference = np.full((16, 16, 3), 100, np.uint8)
    processor, backend = make_processor(np.zeros((64, 64), np.uint8))
    box = BoxPrompt(10, 10, 30, 30)
    processor.add_object_by_prompt(image, "a cup", placement=box)
    assert "Add a cup" in backend.last_prompt
    assert "upper-left" in backend.last_prompt
    processor.add_object_by_reference(image, reference, placement=box, mask_options=MaskOptions())
    np.testing.assert_array_equal(backend.last_reference, reference)


def test_prompt_edit_generation_outpaint_and_upscale():
    image = np.zeros((32, 32, 3), np.uint8)
    processor, _ = make_processor(np.zeros((32, 32), np.uint8))
    assert processor.prompt_edit(image, "make it night").image.shape == image.shape
    assert processor.generate_image("a lake", width=64, height=48).image.shape == (48, 64, 3)
    assert processor.outpaint(image, "continue scene", OutpaintMargins(right=16)).image.shape == (32, 48, 3)
    assert processor.upscale(image, options=UpscaleOptions(scale=2)).image.shape == (64, 64, 3)


def test_processor_rejects_image_above_configured_max_side():
    processor, _ = make_processor(np.zeros((64, 128), np.uint8))
    with pytest.raises(ValueError, match="preprocess_image"):
        processor.segment(np.zeros((100, 200, 3), np.uint8), PointPrompt([(100, 50)], [1]))
