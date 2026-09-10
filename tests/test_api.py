import base64
from io import BytesIO

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from inpaint_core.api import create_app
from inpaint_core.api.codec import decode_image, decode_mask, encode_image, encode_mask
from inpaint_core.api.schemas import GenerationOptionsIn
from inpaint_core.testing import build_fake_processor


@pytest.fixture()
def client():
    processor = build_fake_processor(max_image_side=256)
    app = create_app(processor)
    return TestClient(app)


@pytest.fixture()
def sample_image_b64():
    image = np.zeros((64, 64, 3), dtype=np.uint8)
    image[:, :, 0] = 40
    return encode_image(image)


@pytest.fixture()
def sample_mask_b64():
    mask = np.zeros((64, 64), dtype=np.uint8)
    mask[16:48, 16:48] = 255
    return encode_mask(mask)


def test_codec_round_trip():
    image = np.arange(64 * 64 * 3, dtype=np.uint8).reshape(64, 64, 3)
    decoded = decode_image(encode_image(image))
    np.testing.assert_array_equal(decoded, image)


def test_generation_options_defaults_steps_to_none():
    """A client that doesn't set num_inference_steps must get None through
    to_domain(), so each backend's own tuned step count applies (see
    backends/flux2/klein.py's `options.num_inference_steps or <tuned>`)
    instead of a single hardcoded value overriding every model."""
    assert GenerationOptionsIn().to_domain().num_inference_steps is None


def test_decode_image_composites_transparent_pixels_onto_white():
    """Regression test: decode_image must go through preprocess_image_only
    so RGBA uploads are alpha-composited (like every other entry point into
    the library) instead of a plain `.convert("RGB")` that would silently
    keep whatever RGB values sit under a transparent pixel."""
    rgba = np.zeros((4, 4, 4), dtype=np.uint8)
    rgba[..., 0] = 255  # fully opaque red channel value
    rgba[..., 3] = 0  # fully transparent everywhere
    buffer = BytesIO()
    Image.fromarray(rgba, mode="RGBA").save(buffer, format="PNG")
    payload = base64.b64encode(buffer.getvalue()).decode("ascii")

    decoded = decode_image(payload)

    # A fully transparent pixel must composite onto the white background,
    # not leak the underlying (255, 0, 0) RGB value.
    np.testing.assert_array_equal(decoded[0, 0], [255, 255, 255])


def test_decode_mask_round_trip():
    mask = np.zeros((32, 32), dtype=np.uint8)
    mask[5:10, 5:10] = 255
    decoded_mask = decode_mask(encode_mask(mask))
    np.testing.assert_array_equal(decoded_mask, mask)


def test_health(client):
    response = client.get("/api/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_segment_returns_masks_and_scores(client, sample_image_b64):
    response = client.post(
        "/api/segment",
        json={
            "image": sample_image_b64,
            "selection": {"kind": "point", "points": [[32, 32]], "labels": [1]},
        },
    )
    assert response.status_code == 200
    body = response.json()
    assert body["best_index"] == 0
    assert len(body["masks"]) == 1
    assert body["scores"][0] == pytest.approx(0.97)
    decoded = decode_mask(body["masks"][0])
    assert decoded.shape == (64, 64)


def test_register_image_then_segment_by_id(client, sample_image_b64):
    """A client should be able to upload an image once via /api/images and
    reuse the returned id across repeated /api/segment calls, instead of
    re-sending the full base64 payload on every request."""
    register = client.post("/api/images", json={"image": sample_image_b64})
    assert register.status_code == 200
    image_id = register.json()["image_id"]

    for point in ([10, 10], [50, 50]):
        response = client.post(
            "/api/segment",
            json={
                "image_id": image_id,
                "selection": {"kind": "point", "points": [point], "labels": [1]},
            },
        )
        assert response.status_code == 200
        assert len(response.json()["masks"]) == 1


def test_segment_rejects_both_image_and_image_id(client, sample_image_b64):
    response = client.post(
        "/api/segment",
        json={
            "image": sample_image_b64,
            "image_id": "irrelevant",
            "selection": {"kind": "point", "points": [[1, 1]], "labels": [1]},
        },
    )
    assert response.status_code == 400
    assert "exactly one" in response.json()["detail"]


def test_segment_rejects_unknown_image_id(client):
    response = client.post(
        "/api/segment",
        json={
            "image_id": "does-not-exist",
            "selection": {"kind": "point", "points": [[1, 1]], "labels": [1]},
        },
    )
    assert response.status_code == 400
    assert "Unknown or expired image_id" in response.json()["detail"]


def test_segment_rejects_neither_image_nor_image_id(client):
    """Symmetric to test_segment_rejects_both_image_and_image_id: a request
    with NEITHER field set must also be rejected (400, from _resolve_image's
    "exactly one of" check), not silently reach the segmenter with no image."""
    response = client.post(
        "/api/segment",
        json={"selection": {"kind": "point", "points": [[1, 1]], "labels": [1]}},
    )
    assert response.status_code == 400
    assert "exactly one" in response.json()["detail"]


def test_remove_object_requires_image_field(client, sample_image_b64):
    """Only /api/segment accepts image_id — the "Run" edit endpoints
    (remove-object and friends) fire once per explicit user action, not once
    per pointer click, so they were never the redundant-re-upload problem
    image_id exists to solve, and they don't accept it. `image` stays a
    required, Pydantic-enforced field with a 422 on omission."""
    image_id = client.post("/api/images", json={"image": sample_image_b64}).json()["image_id"]
    response = client.post(
        "/api/remove-object",
        json={"image_id": image_id, "mask": sample_image_b64},
    )
    assert response.status_code == 422


def test_remove_object_with_explicit_mask(client, sample_image_b64, sample_mask_b64):
    response = client.post(
        "/api/remove-object",
        json={"image": sample_image_b64, "mask": sample_mask_b64},
    )
    assert response.status_code == 200
    body = response.json()
    result_image = decode_image(body["image"])
    assert result_image.shape == (64, 64, 3)
    assert body["metadata"]["mode"] == "object_removal"


def test_remove_object_requires_exactly_one_of_selection_or_mask(client, sample_image_b64):
    response = client.post("/api/remove-object", json={"image": sample_image_b64})
    assert response.status_code == 400
    assert "exactly one" in response.json()["detail"]


def test_replace_object_forwards_prompt(client, sample_image_b64, sample_mask_b64):
    response = client.post(
        "/api/replace-object",
        json={"image": sample_image_b64, "mask": sample_mask_b64, "prompt": "a red ball"},
    )
    assert response.status_code == 200


def test_replace_object_rejects_blank_prompt(client, sample_image_b64, sample_mask_b64):
    response = client.post(
        "/api/replace-object",
        json={"image": sample_image_b64, "mask": sample_mask_b64, "prompt": "   "},
    )
    assert response.status_code == 400


def test_replace_background_direct_prompt_edit(client, sample_image_b64):
    response = client.post(
        "/api/replace-background",
        json={
            "image": sample_image_b64,
            "prompt": "a beach",
        },
    )
    assert response.status_code == 200
    assert "mask" not in response.json()


def test_add_object_by_prompt_with_placement_box(client, sample_image_b64):
    response = client.post(
        "/api/add-object/prompt",
        json={
            "image": sample_image_b64,
            "prompt": "a small vase",
            "placement": {"kind": "box", "x1": 10, "y1": 10, "x2": 30, "y2": 30},
        },
    )
    assert response.status_code == 200
    assert "mask" not in response.json()


def test_replace_background_rejects_legacy_mask_field(client, sample_image_b64, sample_mask_b64):
    """replace-background dropped mask/selection support — a client still on
    the old contract must get a 422, not have the field silently ignored."""
    response = client.post(
        "/api/replace-background",
        json={
            "image": sample_image_b64,
            "prompt": "a beach",
            "foreground_mask": sample_mask_b64,
        },
    )
    assert response.status_code == 422


def test_add_object_by_prompt_rejects_legacy_mask_field(client, sample_image_b64, sample_mask_b64):
    """add-object/prompt dropped mask support — a client still on the old
    contract must get a 422, not have the field silently ignored."""
    response = client.post(
        "/api/add-object/prompt",
        json={
            "image": sample_image_b64,
            "prompt": "a small vase",
            "mask": sample_mask_b64,
        },
    )
    assert response.status_code == 422


def test_add_object_by_reference(client, sample_image_b64):
    reference = np.full((32, 32, 3), 120, dtype=np.uint8)
    response = client.post(
        "/api/add-object/reference",
        json={
            "image": sample_image_b64,
            "reference": encode_image(reference),
            "placement": {"kind": "box", "x1": 5, "y1": 5, "x2": 20, "y2": 20},
        },
    )
    assert response.status_code == 200


def test_prompt_edit(client, sample_image_b64):
    response = client.post(
        "/api/prompt-edit",
        json={"image": sample_image_b64, "prompt": "make it winter"},
    )
    assert response.status_code == 200
    body = response.json()
    assert decode_image(body["image"]).shape == (64, 64, 3)


def test_generate_image_uses_requested_dimensions(client):
    response = client.post(
        "/api/generate",
        json={"prompt": "a lake", "width": 96, "height": 64},
    )
    assert response.status_code == 200
    body = response.json()
    assert decode_image(body["image"]).shape == (64, 96, 3)


def test_outpaint_requires_positive_margin(client, sample_image_b64):
    response = client.post(
        "/api/outpaint",
        json={
            "image": sample_image_b64,
            "prompt": "continue the scene",
            "margins": {"left": 0, "top": 0, "right": 0, "bottom": 0},
        },
    )
    assert response.status_code == 400


def test_outpaint_extends_canvas(client, sample_image_b64):
    response = client.post(
        "/api/outpaint",
        json={
            "image": sample_image_b64,
            "prompt": "continue the scene",
            "margins": {"left": 0, "top": 0, "right": 32, "bottom": 0},
        },
    )
    assert response.status_code == 200
    body = response.json()
    assert decode_image(body["image"]).shape == (64, 96, 3)


def test_upscale_multiplies_dimensions(client, sample_image_b64):
    response = client.post(
        "/api/upscale",
        json={"image": sample_image_b64, "options": {"scale": 2}},
    )
    assert response.status_code == 200
    body = response.json()
    assert decode_image(body["image"]).shape == (128, 128, 3)


def test_missing_required_field_returns_422(client):
    response = client.post("/api/replace-object", json={"image": "x"})
    assert response.status_code == 422


def test_invalid_base64_image_returns_400(client):
    response = client.post(
        "/api/remove-object",
        json={"image": "not-valid-base64!!", "mask": "also-not-valid!!"},
    )
    assert response.status_code == 400
