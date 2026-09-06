import numpy as np

from inpaint_core.masks.operations import dilate, erode, feather, invert, threshold


def sample_mask():
    mask = np.zeros((21, 21), dtype=np.uint8)
    mask[8:13, 8:13] = 255
    return mask


def test_threshold_returns_binary_uint8():
    mask = np.array([[0, 126, 128, 255]], dtype=np.uint8)
    result = threshold(mask, 127)
    assert result.dtype == np.uint8
    assert result.tolist() == [[0, 0, 255, 255]]


def test_invert_twice_restores_binary_mask():
    mask = sample_mask()
    assert np.array_equal(invert(invert(mask)), mask)


def test_dilate_and_erode_change_area_in_expected_direction():
    mask = sample_mask()
    assert np.count_nonzero(dilate(mask, 2)) > np.count_nonzero(mask)
    assert np.count_nonzero(erode(mask, 1)) < np.count_nonzero(mask)


def test_feather_returns_normalized_alpha():
    alpha = feather(sample_mask(), 2)
    assert alpha.dtype == np.float32
    assert 0.0 <= float(alpha.min()) <= float(alpha.max()) <= 1.0

