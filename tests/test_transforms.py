import numpy as np

from inpaint_core.masks.transforms import prepare_masked_crop, restore_crop


def test_prepare_and_restore_preserve_original_shape():
    image = np.zeros((80, 120, 3), dtype=np.uint8)
    mask = np.zeros((80, 120), dtype=np.uint8)
    mask[30:50, 50:70] = 255

    prepared = prepare_masked_crop(
        image,
        mask,
        padding=10,
        max_side=64,
        size_multiple=16,
    )
    edited = np.full_like(prepared.image, 200)
    restored = restore_crop(image, edited, prepared.transform)

    assert restored.shape == image.shape
    x1, y1, x2, y2 = prepared.transform.crop_box
    assert np.all(restored[y1:y2, x1:x2] == 200)
    assert np.all(restored[:y1] == 0)


def test_prepared_dimensions_are_multiples():
    image = np.zeros((77, 113, 3), dtype=np.uint8)
    mask = np.zeros((77, 113), dtype=np.uint8)
    mask[10:50, 20:80] = 255
    prepared = prepare_masked_crop(image, mask, padding=3, size_multiple=16)
    assert prepared.image.shape[0] % 16 == 0
    assert prepared.image.shape[1] % 16 == 0

