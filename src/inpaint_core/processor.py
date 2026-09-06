from __future__ import annotations

from pathlib import Path

from .compositing import Compositor
from .config import AppConfig
from .editing import FluxFillEditor, MaskedImageEditor
from .image import ImageSource, preprocess_image
from .masks.operations import dilate, erode, invert, threshold, validate_mask
from .masks.transforms import prepare_masked_crop, restore_crop
from .models import MemoryTracker, ModelManager
from .segmentation import SAM2Segmenter, Segmenter
from .types import (
    EditResult,
    GenerationOptions,
    ImageArray,
    MaskArray,
    MaskOptions,
    SegmentationResult,
    SelectionPrompt,
)


class ImageProcessor:
    """Framework-independent facade for segmentation-guided image editing."""

    def __init__(
        self,
        *,
        config: AppConfig,
        segmenter: Segmenter,
        editor: MaskedImageEditor,
        model_manager: ModelManager | None = None,
        compositor: Compositor | None = None,
        memory_tracker: MemoryTracker | None = None,
    ) -> None:
        self.config = config
        self.segmenter = segmenter
        self.editor = editor
        self.model_manager = model_manager or ModelManager(config.memory.policy)
        self.compositor = compositor or Compositor()
        self.memory_tracker = memory_tracker or MemoryTracker(
            config.memory.track_peak
        )

    @classmethod
    def from_config(cls, path: str | Path) -> "ImageProcessor":
        config = AppConfig.from_yaml(path)
        manager = ModelManager(config.memory.policy)
        segmenter = SAM2Segmenter(config.segmentation, config.device, manager)
        editor = FluxFillEditor(config.editing, config.device, manager)
        return cls(
            config=config,
            segmenter=segmenter,
            editor=editor,
            model_manager=manager,
        )

    def segment(
        self,
        image: ImageSource,
        selection: SelectionPrompt,
    ) -> SegmentationResult:
        image = preprocess_image(image)
        if self.config.memory.policy == "sequential":
            self.model_manager.release("flux_fill")
        with self.memory_tracker.measure("segmentation"):
            return self.segmenter.segment(image, selection)

    def remove_object(
        self,
        image: ImageSource,
        *,
        selection: SelectionPrompt | None = None,
        mask: MaskArray | None = None,
        mask_options: MaskOptions | None = None,
        generation_options: GenerationOptions | None = None,
    ) -> EditResult:
        return self._run_masked_edit(
            mode="object_removal",
            image=image,
            selection=selection,
            mask=mask,
            prompt=(
                "Remove the selected object and reconstruct the natural "
                "background, matching surrounding texture, lighting, and perspective."
            ),
            invert_selection=False,
            mask_options=mask_options,
            generation_options=generation_options,
        )

    def replace_object(
        self,
        image: ImageSource,
        prompt: str,
        *,
        selection: SelectionPrompt | None = None,
        mask: MaskArray | None = None,
        mask_options: MaskOptions | None = None,
        generation_options: GenerationOptions | None = None,
    ) -> EditResult:
        if not prompt.strip():
            raise ValueError("A replacement prompt is required.")
        return self._run_masked_edit(
            mode="object_replacement",
            image=image,
            selection=selection,
            mask=mask,
            prompt=prompt,
            invert_selection=False,
            mask_options=mask_options,
            generation_options=generation_options,
        )

    def replace_background(
        self,
        image: ImageSource,
        prompt: str,
        *,
        foreground_selection: SelectionPrompt | None = None,
        foreground_mask: MaskArray | None = None,
        mask_options: MaskOptions | None = None,
        generation_options: GenerationOptions | None = None,
    ) -> EditResult:
        if not prompt.strip():
            raise ValueError("A background prompt is required.")
        return self._run_masked_edit(
            mode="background_replacement",
            image=image,
            selection=foreground_selection,
            mask=foreground_mask,
            prompt=prompt,
            invert_selection=True,
            mask_options=mask_options,
            generation_options=generation_options,
        )

    def _run_masked_edit(
        self,
        *,
        mode: str,
        image: ImageSource,
        selection: SelectionPrompt | None,
        mask: MaskArray | None,
        prompt: str,
        invert_selection: bool,
        mask_options: MaskOptions | None,
        generation_options: GenerationOptions | None,
    ) -> EditResult:
        image = preprocess_image(image)
        mask_options = mask_options or self._default_mask_options()
        generation_options = generation_options or self._default_generation_options()

        raw_mask = self._resolve_mask(image, selection, mask, mask_options)
        edit_mask = invert(raw_mask) if invert_selection else threshold(
            raw_mask, mask_options.threshold
        )
        edit_mask = dilate(edit_mask, mask_options.dilate)
        edit_mask = erode(edit_mask, mask_options.erode)
        validate_mask(edit_mask, image.shape)

        prepared = prepare_masked_crop(
            image,
            edit_mask,
            padding=mask_options.crop_padding,
            max_side=self.config.processing.generation_max_side,
            size_multiple=self.config.processing.size_multiple,
        )

        if self.config.memory.policy == "sequential":
            self.model_manager.release("sam2")
        with self.memory_tracker.measure("generation"):
            generated_crop = self.editor.edit(
                prepared.image,
                prepared.mask,
                prompt,
                generation_options,
            )

        generated_full = restore_crop(
            image,
            generated_crop.image,
            prepared.transform,
        )
        final = self.compositor.compose(
            image,
            generated_full,
            edit_mask,
            feather_radius=mask_options.feather,
        )

        return EditResult(
            image=final,
            mask=edit_mask,
            generated_image=generated_full,
            metadata={
                "mode": mode,
                "seed": generated_crop.seed,
                "stages": self.memory_tracker.results.copy(),
                **generated_crop.metadata,
            },
        )

    def _resolve_mask(
        self,
        image: ImageArray,
        selection: SelectionPrompt | None,
        mask: MaskArray | None,
        options: MaskOptions,
    ) -> MaskArray:
        if (selection is None) == (mask is None):
            raise ValueError("Provide exactly one of selection or mask.")
        if mask is not None:
            validate_mask(mask, image.shape)
            return threshold(mask, options.threshold)

        segmentation = self.segment(image, selection)
        index = (
            segmentation.best_index
            if options.candidate_index is None
            else options.candidate_index
        )
        if not 0 <= index < len(segmentation.masks):
            raise IndexError("candidate_index is outside the returned mask list.")
        return threshold(segmentation.masks[index], options.threshold)

    def _default_mask_options(self) -> MaskOptions:
        allowed = MaskOptions.__dataclass_fields__.keys()
        values = {
            key: value
            for key, value in self.config.mask_defaults.items()
            if key in allowed
        }
        return MaskOptions(**values)

    def _default_generation_options(self) -> GenerationOptions:
        allowed = GenerationOptions.__dataclass_fields__.keys()
        values = {
            key: value
            for key, value in self.config.generation_defaults.items()
            if key in allowed
        }
        return GenerationOptions(**values)
