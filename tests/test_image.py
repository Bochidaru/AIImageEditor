import numpy as np
from PIL import Image

from inpaint_core import preprocess_image, validate_image


def test_float_zero_to_one_is_scaled_to_uint8():
    source = np.array([[[0.0, 0.5, 1.0]]], dtype=np.float32)
    result = preprocess_image(source)
    assert result.dtype == np.uint8
    assert result.tolist() == [[[0, 127, 255]]]


def test_bgr_numpy_is_converted_to_rgb():
    source = np.array([[[255, 0, 0]]], dtype=np.uint8)
    result = preprocess_image(source, color_space="bgr")
    assert result.tolist() == [[[0, 0, 255]]]


def test_grayscale_is_expanded_to_three_channels():
    source = np.array([[0, 255]], dtype=np.uint8)
    result = preprocess_image(source)
    assert result.shape == (1, 2, 3)
    validate_image(result)


def test_pil_input_is_converted_to_rgb():
    source = Image.new("L", (3, 2), color=128)
    result = preprocess_image(source)
    assert result.shape == (2, 3, 3)
    assert result.dtype == np.uint8

