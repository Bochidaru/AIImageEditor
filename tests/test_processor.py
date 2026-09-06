import numpy as np

from inpaint_core.compositing import Compositor
from inpaint_core.config import AppConfig
from inpaint_core.processor import ImageProcessor
from inpaint_core.types import GenerationResult, MaskOptions, SegmentationResult


class FakeSegmenter:
    def __init__(self, mask):
        self.mask = mask
        self.calls = 0

    def segment(self, image, selection):
        self.calls += 1
        return SegmentationResult([self.mask], [1.0], 0)


class FakeEditor:
    def __init__(self):
        self.last_mask = None
        self.last_prompt = None

    def edit(self, image, mask, prompt, options):
        self.last_mask = mask
        self.last_prompt = prompt
        generated = np.full_like(image, 200)
        return GenerationResult(generated, options.seed, {"backend": "fake"})


def make_processor(mask):
    config = AppConfig.from_dict(
        {
            "device": {"name": "cpu", "dtype": "float32"},
            "models": {
                "segmentation": {"backend": "fake"},
                "editing": {"backend": "fake"},
            },
            "processing": {
                "generation_max_side": 128,
                "crop_padding": 0,
                "size_multiple": 16,
            },
        }
    )
    segmenter = FakeSegmenter(mask)
    editor = FakeEditor()
    processor = ImageProcessor(
        config=config,
        segmenter=segmenter,
        editor=editor,
        compositor=Compositor(),
    )
    return processor, segmenter, editor


def test_removal_preserves_pixels_outside_mask():
    image = np.zeros((64, 64, 3), dtype=np.uint8)
    mask = np.zeros((64, 64), dtype=np.uint8)
    mask[20:40, 20:40] = 255
    processor, _, _ = make_processor(mask)

    result = processor.remove_object(
        image,
        mask=mask,
        mask_options=MaskOptions(dilate=0, feather=0, crop_padding=4),
    )

    assert np.all(result.image[0, 0] == 0)
    assert np.all(result.image[30, 30] == 200)


def test_background_replacement_inverts_foreground_mask():
    image = np.zeros((64, 64, 3), dtype=np.uint8)
    foreground = np.zeros((64, 64), dtype=np.uint8)
    foreground[20:40, 20:40] = 255
    processor, _, _ = make_processor(foreground)

    result = processor.replace_background(
        image,
        "a beach",
        foreground_mask=foreground,
        mask_options=MaskOptions(feather=0, crop_padding=0),
    )

    assert np.all(result.image[30, 30] == 0)
    assert np.all(result.image[0, 0] == 200)


def test_replacement_forwards_user_prompt():
    image = np.zeros((64, 64, 3), dtype=np.uint8)
    mask = np.zeros((64, 64), dtype=np.uint8)
    mask[20:40, 20:40] = 255
    processor, _, editor = make_processor(mask)

    processor.replace_object(
        image,
        "a ceramic vase",
        mask=mask,
        mask_options=MaskOptions(feather=0, crop_padding=0),
    )

    assert editor.last_prompt == "a ceramic vase"

