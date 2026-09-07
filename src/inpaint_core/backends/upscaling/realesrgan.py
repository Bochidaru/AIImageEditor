from __future__ import annotations

from pathlib import Path
import sys
from types import ModuleType

import cv2
import numpy as np

from ...config import DeviceConfig, ModelConfig
from ...models.manager import ModelManager
from ...types import GenerationResult, ImageArray, UpscaleOptions


class RealESRGANBackend:
    RESOURCE_KEY = "realesrgan"

    def __init__(self, config: ModelConfig, device: DeviceConfig, manager: ModelManager):
        self.config = config
        self.device = device
        self.manager = manager

    def upscale(self, image: ImageArray, options: UpscaleOptions) -> GenerationResult:
        resource_key = (
            f"{self.RESOURCE_KEY}:{options.tile}:{options.tile_pad}:{options.pre_pad}"
        )
        upsampler = self.manager.get(
            resource_key, lambda: self._load_upsampler(options)
        )
        bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        if options.face_enhance:
            enhancer = self.manager.get(
                f"gfpgan:{options.scale}:{resource_key}",
                lambda: self._load_face_enhancer(upsampler, options.scale),
            )
            _, _, output = enhancer.enhance(
                bgr, has_aligned=False, only_center_face=False, paste_back=True
            )
        else:
            output, _ = upsampler.enhance(bgr, outscale=options.scale)
        rgb = cv2.cvtColor(output, cv2.COLOR_BGR2RGB)
        return GenerationResult(
            np.ascontiguousarray(rgb, dtype=np.uint8),
            seed=0,
            metadata={"backend": "realesrgan", "scale": options.scale},
        )

    def _load_upsampler(self, options: UpscaleOptions):
        self._install_torchvision_compat()
        try:
            from basicsr.archs.rrdbnet_arch import RRDBNet
            from basicsr.utils.download_util import load_file_from_url
            from realesrgan import RealESRGANer
        except ImportError as exc:
            raise ImportError(
                "Real-ESRGAN is not installed. Install `realesrgan` and `basicsr`."
            ) from exc

        native_scale = int(self.config.options.get("native_scale", 4))
        model = RRDBNet(
            num_in_ch=3,
            num_out_ch=3,
            num_feat=64,
            num_block=23,
            num_grow_ch=32,
            scale=native_scale,
        )
        checkpoint = self.config.checkpoint
        if checkpoint:
            model_path = str(Path(checkpoint).expanduser())
        else:
            url = self.config.options.get(
                "model_url",
                "https://github.com/xinntao/Real-ESRGAN/releases/download/"
                "v0.1.0/RealESRGAN_x4plus.pth",
            )
            model_path = load_file_from_url(
                url=url,
                model_dir=self.config.options.get("weights_dir", "weights/realesrgan"),
                progress=True,
                file_name=None,
            )
        return RealESRGANer(
            scale=native_scale,
            model_path=model_path,
            model=model,
            tile=options.tile,
            tile_pad=options.tile_pad,
            pre_pad=options.pre_pad,
            half=self.device.name.startswith("cuda") and self.device.dtype == "float16",
            gpu_id=self.config.options.get("gpu_id"),
        )

    @staticmethod
    def _install_torchvision_compat() -> None:
        """Bridge the legacy BasicSR import removed by newer torchvision."""
        module_name = "torchvision.transforms.functional_tensor"
        if module_name in sys.modules:
            return
        try:
            __import__(module_name)
        except ModuleNotFoundError:
            from torchvision.transforms import functional

            compatibility = ModuleType(module_name)
            compatibility.rgb_to_grayscale = functional.rgb_to_grayscale
            sys.modules[module_name] = compatibility

    def _load_face_enhancer(self, background_upsampler, scale: int):
        try:
            from gfpgan import GFPGANer
        except ImportError as exc:
            raise ImportError("Install `gfpgan` to use face_enhance=True.") from exc
        return GFPGANer(
            model_path=self.config.options.get(
                "face_model_url",
                "https://github.com/TencentARC/GFPGAN/releases/download/"
                "v1.3.0/GFPGANv1.3.pth",
            ),
            upscale=scale,
            arch="clean",
            channel_multiplier=2,
            bg_upsampler=background_upsampler,
        )
