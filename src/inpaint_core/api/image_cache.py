from __future__ import annotations

import threading
import uuid
from collections import OrderedDict

from ..types import ImageArray


class ImageCache:
    """Bounded in-memory store so a client can upload an image once (POST
    /api/images) and reference it by id on subsequent requests, instead of
    re-sending the full base64 image on every request — e.g. the frontend
    calls /api/segment once per point/box a user tries while hunting for the
    right selection, all against the same unchanged source image.

    Simple LRU eviction, no TTL: adequate for a single-process dev/demo API.
    A multi-worker/multi-process deployment would need a shared store
    (Redis, etc.) instead of this in-memory dict.
    """

    def __init__(self, max_entries: int = 32) -> None:
        self._max_entries = max_entries
        self._lock = threading.Lock()
        self._entries: "OrderedDict[str, ImageArray]" = OrderedDict()

    def put(self, array: ImageArray) -> str:
        image_id = uuid.uuid4().hex
        with self._lock:
            self._entries[image_id] = array
            self._entries.move_to_end(image_id)
            while len(self._entries) > self._max_entries:
                self._entries.popitem(last=False)
        return image_id

    def get(self, image_id: str) -> ImageArray | None:
        with self._lock:
            array = self._entries.get(image_id)
            if array is not None:
                self._entries.move_to_end(image_id)
            return array
