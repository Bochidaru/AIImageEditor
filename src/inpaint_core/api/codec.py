from __future__ import annotations

import base64
import binascii
from io import BytesIO

import numpy as np
from PIL import Image

from ..image import preprocess_image_only, validate_image
from ..masks.operations import validate_mask
from ..types import ImageArray, MaskArray


class DecodeError(ValueError):
    """Raised when a request body contains an unreadable image/mask payload."""


def _strip_data_url_prefix(value: str) -> bytes:
    try:
        payload = value.split(",", 1)[1] if value.startswith("data:") else value
        return base64.b64decode(payload, validate=False)
    except (binascii.Error, ValueError, IndexError) as exc:
        raise DecodeError("Could not base64-decode the provided image data.") from exc


def decode_image(value: str) -> ImageArray:
    """Decode a base64 (optionally data-URL prefixed) payload into an RGB uint8 array.

    Delegates to preprocess_image_only so uploaded images get the same
    EXIF-orientation correction and alpha-background compositing as every
    other entry point into the library (see image.py's _from_pil) — a
    hand-rolled `.convert("RGB")` here would silently drop both.
    """
    raw = _strip_data_url_prefix(value)
    try:
        with Image.open(BytesIO(raw)) as image:
            array = preprocess_image_only(image)
    except Exception as exc:  # noqa: BLE001 - Pillow raises several unrelated error types
        raise DecodeError("Could not decode image payload.") from exc
    validate_image(array)
    return array


def decode_mask(value: str) -> MaskArray:
    """Decode a base64 (optionally data-URL prefixed) payload into a single-channel uint8 mask.

    A mask with an alpha channel (RGBA/LA) is assumed to encode the selected
    region as transparency, per common mask-authoring convention, and the
    alpha channel is used directly. Plain RGB/L masks fall back to luminance.
    Using `.convert("L")` unconditionally would silently discard alpha and
    compute a semantically wrong mask from RGB brightness instead.
    """
    raw = _strip_data_url_prefix(value)
    try:
        with Image.open(BytesIO(raw)) as image:
            channel = image.getchannel("A") if image.mode in ("RGBA", "LA") else image.convert("L")
            array = np.ascontiguousarray(np.asarray(channel, dtype=np.uint8))
    except Exception as exc:  # noqa: BLE001
        raise DecodeError("Could not decode mask payload.") from exc
    validate_mask(array)
    return array


def encode_image(array: ImageArray) -> str:
    """Encode an RGB uint8 array as a `data:image/png;base64,...` string."""
    image = Image.fromarray(array, mode="RGB")
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def encode_mask(array: MaskArray) -> str:
    """Encode a single-channel uint8 mask as a `data:image/png;base64,...` string."""
    image = Image.fromarray(array, mode="L")
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


__all__ = ["DecodeError", "decode_image", "decode_mask", "encode_image", "encode_mask"]
