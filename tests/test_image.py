import numpy as np
from PIL import Image

from inpaint_core import BoxPrompt, PointPrompt, preprocess_image, validate_image


def test_float_zero_to_one_is_scaled_to_uint8():
    source = np.array([[[0.0, 0.5, 1.0]]], dtype=np.float32)
    result, _ = preprocess_image(
        source,
        PointPrompt(points=[(0, 0)], labels=[1]),
    )
    assert result.dtype == np.uint8
    assert result.tolist() == [[[0, 127, 255]]]


def test_bgr_numpy_is_converted_to_rgb():
    source = np.array([[[255, 0, 0]]], dtype=np.uint8)
    result, _ = preprocess_image(
        source,
        PointPrompt(points=[(0, 0)], labels=[1]),
        color_space="bgr",
    )
    assert result.tolist() == [[[0, 0, 255]]]


def test_grayscale_is_expanded_to_three_channels():
    source = np.array([[0, 255]], dtype=np.uint8)
    result, _ = preprocess_image(
        source,
        PointPrompt(points=[(0, 0)], labels=[1]),
    )
    assert result.shape == (1, 2, 3)
    validate_image(result)


def test_pil_input_is_converted_to_rgb():
    source = Image.new("L", (3, 2), color=128)
    result, _ = preprocess_image(
        source,
        PointPrompt(points=[(0, 0)], labels=[1]),
    )
    assert result.shape == (2, 3, 3)
    assert result.dtype == np.uint8


def test_max_side_downscales_without_changing_aspect_ratio():
    source = np.zeros((100, 200, 3), dtype=np.uint8)
    result, _ = preprocess_image(
        source,
        PointPrompt(points=[(100, 50)], labels=[1]),
        max_side=80,
    )
    assert result.shape == (40, 80, 3)


def test_max_side_does_not_upscale_small_images():
    source = np.zeros((20, 40, 3), dtype=np.uint8)
    selection = PointPrompt(points=[(20, 10)], labels=[1])
    result, resized_selection = preprocess_image(
        source,
        selection,
        max_side=80,
    )
    assert result.shape == source.shape
    assert resized_selection == selection


def test_point_prompt_is_scaled_with_image():
    source = np.zeros((100, 200, 3), dtype=np.uint8)
    selection = PointPrompt(points=[(100, 50)], labels=[1])

    image, resized_selection = preprocess_image(
        source,
        selection,
        max_side=80,
    )

    assert image.shape == (40, 80, 3)
    assert resized_selection.points == [(40.0, 20.0)]
    assert resized_selection.labels == [1]


def test_box_prompt_is_scaled_with_image():
    source = np.zeros((100, 200, 3), dtype=np.uint8)
    selection = BoxPrompt(x1=20, y1=10, x2=180, y2=90)

    image, resized_selection = preprocess_image(
        source,
        selection,
        max_side=100,
    )

    assert image.shape == (50, 100, 3)
    assert resized_selection == BoxPrompt(x1=10, y1=5, x2=90, y2=45)
