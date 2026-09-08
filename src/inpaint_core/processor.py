from __future__ import annotations

from pathlib import Path

from .backends.flux import FluxFillBackend, FluxGeneratorBackend, FluxKontextBackend
from .backends.omnipaint import OmniPaintBackend
from .backends.protocols import ImageGenerator, MaskedEditor, ObjectRemover, PromptEditor, ReferenceInserter, Segmenter, Upscaler
from .backends.segmentation import SAM2Segmenter
from .backends.upscaling import RealESRGANBackend
from .config import AppConfig
from .models import MemoryTracker, ModelManager, prefetch_model_assets
from .operations import BackgroundReplacementOperation, ImageGenerationOperation, ObjectInsertionOperation, ObjectRemovalOperation, ObjectReplacementOperation, OutpaintingOperation, PromptEditOperation, UpscalingOperation
from .operations.common import OperationContext
from .types import BoxPrompt, EditResult, GenerationOptions, GenerationResult, ImageArray, MaskArray, MaskOptions, OutpaintMargins, SegmentationResult, SelectionPrompt, UpscaleOptions


class ImageProcessor:
    """Thin public facade over independent image-editing operations."""

    def __init__(
        self, *, config: AppConfig, segmenter: Segmenter,
        flux_fill: MaskedEditor, omnipaint: ObjectRemover | ReferenceInserter,
        flux_kontext: PromptEditor, generator: ImageGenerator, upscaler: Upscaler,
        model_manager: ModelManager | None = None,
        memory_tracker: MemoryTracker | None = None,
    ) -> None:
        self.config = config
        self.model_manager = model_manager or ModelManager(config.memory.policy)
        self.memory_tracker = memory_tracker or MemoryTracker(config.memory.track_peak)
        self.segmenter = segmenter
        context = OperationContext(config, segmenter, self.memory_tracker)
        self._context = context
        self._remove = ObjectRemovalOperation(context, omnipaint)
        self._replace = ObjectReplacementOperation(context, flux_fill)
        self._replace_background = BackgroundReplacementOperation(context, flux_fill)
        self._insert = ObjectInsertionOperation(context, flux_fill, omnipaint)
        self._prompt_edit = PromptEditOperation(context, flux_kontext)
        self._generate = ImageGenerationOperation(context, generator)
        self._outpaint = OutpaintingOperation(context, flux_fill)
        self._upscale = UpscalingOperation(context, upscaler)

    @classmethod
    def from_config(cls, path: str | Path) -> "ImageProcessor":
        config = AppConfig.from_yaml(path)
        cls._validate_backends(config)
        if config.artifacts.prefetch_on_init:
            prefetch_model_assets(config)
        manager = ModelManager(config.memory.policy)
        return cls(
            config=config,
            segmenter=SAM2Segmenter(config.segmentation, config.device, manager),
            flux_fill=FluxFillBackend(config.flux_fill, config.device, manager),
            omnipaint=OmniPaintBackend(config.omnipaint, config.device, manager),
            flux_kontext=FluxKontextBackend(config.flux_kontext, config.device, manager),
            generator=FluxGeneratorBackend(config.flux_generation, config.device, manager),
            upscaler=RealESRGANBackend(config.upscaler, config.device, manager),
            model_manager=manager,
        )

    @staticmethod
    def _validate_backends(config: AppConfig) -> None:
        expected = {
            "models.segmentation.backend": (config.segmentation.backend, "sam2"),
            "models.flux_fill.backend": (config.flux_fill.backend, "flux_fill"),
            "models.flux_generation.backend": (config.flux_generation.backend, "flux"),
            "models.omnipaint.backend": (config.omnipaint.backend, "omnipaint"),
            "models.flux_kontext.backend": (config.flux_kontext.backend, "flux_kontext"),
            "models.upscaler.backend": (config.upscaler.backend, "realesrgan"),
        }
        for field, (actual, wanted) in expected.items():
            if actual != wanted:
                raise ValueError(f"{field} must be {wanted!r}, got {actual!r}.")

    def segment(self, image: ImageArray, selection: SelectionPrompt) -> SegmentationResult:
        self._context.validate_image(image)
        with self.memory_tracker.measure("segmentation"):
            return self.segmenter.segment(image, selection)

    def remove_object(self, image: ImageArray, *, selection: SelectionPrompt | None = None, mask: MaskArray | None = None, mask_options: MaskOptions | None = None, generation_options: GenerationOptions | None = None) -> EditResult:
        return self._remove.run(image, selection=selection, mask=mask, mask_options=mask_options, generation_options=generation_options)

    def replace_object(self, image: ImageArray, prompt: str, *, selection: SelectionPrompt | None = None, mask: MaskArray | None = None, mask_options: MaskOptions | None = None, generation_options: GenerationOptions | None = None) -> EditResult:
        return self._replace.run(image, prompt, selection=selection, mask=mask, mask_options=mask_options, generation_options=generation_options)

    def replace_background(self, image: ImageArray, prompt: str, *, foreground_selection: SelectionPrompt | None = None, foreground_mask: MaskArray | None = None, mask_options: MaskOptions | None = None, generation_options: GenerationOptions | None = None) -> EditResult:
        return self._replace_background.run(image, prompt, foreground_selection=foreground_selection, foreground_mask=foreground_mask, mask_options=mask_options, generation_options=generation_options)

    def add_object_by_prompt(self, image: ImageArray, prompt: str, *, placement: BoxPrompt | None = None, mask: MaskArray | None = None, mask_options: MaskOptions | None = None, generation_options: GenerationOptions | None = None) -> EditResult:
        return self._insert.by_prompt(image, prompt, placement=placement, mask=mask, mask_options=mask_options, generation_options=generation_options)

    def add_object_by_reference(self, image: ImageArray, reference: ImageArray, *, placement: BoxPrompt | None = None, mask: MaskArray | None = None, reference_mask: MaskArray | None = None, mask_options: MaskOptions | None = None, generation_options: GenerationOptions | None = None) -> EditResult:
        return self._insert.by_reference(image, reference, placement=placement, mask=mask, reference_mask=reference_mask, mask_options=mask_options, generation_options=generation_options)

    def prompt_edit(self, image: ImageArray, prompt: str, *, generation_options: GenerationOptions | None = None) -> GenerationResult:
        return self._prompt_edit.run(image, prompt, generation_options)

    def generate_image(self, prompt: str, *, width: int = 1024, height: int = 1024, generation_options: GenerationOptions | None = None) -> GenerationResult:
        return self._generate.run(prompt, width=width, height=height, generation_options=generation_options)

    def outpaint(self, image: ImageArray, prompt: str, margins: OutpaintMargins, *, generation_options: GenerationOptions | None = None) -> EditResult:
        return self._outpaint.run(image, prompt, margins, generation_options=generation_options)

    def upscale(self, image: ImageArray, *, options: UpscaleOptions | None = None) -> GenerationResult:
        return self._upscale.run(image, options)

    def release_models(self) -> None:
        self.model_manager.release_all()

    def save_memory_stats(self, path: str | Path) -> Path:
        return self.memory_tracker.save_json(path)
