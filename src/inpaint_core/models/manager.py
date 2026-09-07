from __future__ import annotations

import gc
from collections.abc import Callable
from typing import Any


class ModelManager:
    """Lazy model registry with explicit release hooks.

    Device placement remains inside each backend adapter because Diffusers and
    segmentation libraries expose different offload APIs.
    """

    def __init__(self, memory_policy: str = "sequential") -> None:
        if memory_policy not in ['sequential', 'resident']:
            raise ValueError(
                "memory_policy must be 'sequential' or 'resident', "
                f"got {memory_policy!r}."
            )
        self.memory_policy = memory_policy
        self._models: dict[str, Any] = {}

    def get(self, name: str, loader: Callable[[], Any]) -> Any:
        if name not in self._models:
            if self.memory_policy == "sequential":
                self.release_all()
            self._models[name] = loader()
        return self._models[name]

    def contains(self, name: str) -> bool:
        return name in self._models

    def release(self, name: str) -> None:
        model = self._models.pop(name, None)
        if model is not None:
            del model
        self.cleanup_cuda()

    def release_all(self) -> None:
        names = list(self._models)
        for name in names:
            self.release(name)

    @staticmethod
    def cleanup_cuda() -> None:
        gc.collect()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass
