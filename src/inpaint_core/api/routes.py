from __future__ import annotations

from fastapi import APIRouter, Depends

from ..processor import ImageProcessor
from ..types import EditResult, GenerationResult, ImageArray, SegmentationResult, SelectionPrompt
from ..validation import require_exactly_one
from .codec import DecodeError, decode_image, decode_mask, encode_image, encode_mask
from .deps import get_image_cache, get_processor
from .image_cache import ImageCache
from .schemas import (
    AddObjectByPromptRequest,
    AddObjectByReferenceRequest,
    EditResponse,
    GenerateImageRequest,
    GenerationResponse,
    OutpaintRequest,
    PromptEditRequest,
    RegisterImageRequest,
    RegisterImageResponse,
    RemoveObjectRequest,
    ReplaceBackgroundRequest,
    ReplaceObjectRequest,
    SegmentationResponse,
    SegmentRequest,
    SelectionIn,
    UpscaleRequest,
)

router = APIRouter(prefix="/api")


def _selection(selection_in: SelectionIn | None) -> SelectionPrompt | None:
    return selection_in.to_domain() if selection_in is not None else None


def _resolve_image(image: str | None, image_id: str | None, cache: ImageCache) -> ImageArray:
    """Resolve a request's image from an inline base64 payload or a
    previously-registered `image_id` (see POST /api/images) — exactly one
    must be provided, mirroring the "exactly one of X or Y" convention
    operations/common.py already uses for selection/mask.
    """
    require_exactly_one(image, image_id, "image", "image_id")
    if image_id is not None:
        cached = cache.get(image_id)
        if cached is None:
            raise DecodeError(f"Unknown or expired image_id: {image_id!r}.")
        return cached
    assert image is not None
    return decode_image(image)


def _segmentation_response(result: SegmentationResult) -> SegmentationResponse:
    return SegmentationResponse(
        masks=[encode_mask(mask) for mask in result.masks],
        scores=result.scores,
        best_index=result.best_index,
        metadata=result.metadata,
    )


def _edit_response(result: EditResult) -> EditResponse:
    return EditResponse(
        image=encode_image(result.image),
        mask=encode_mask(result.mask),
        generated_image=encode_image(result.generated_image) if result.generated_image is not None else None,
        metadata=result.metadata,
    )


def _generation_response(result: GenerationResult) -> GenerationResponse:
    return GenerationResponse(image=encode_image(result.image), seed=result.seed, metadata=result.metadata)


@router.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@router.post("/images", response_model=RegisterImageResponse)
def register_image(
    body: RegisterImageRequest, cache: ImageCache = Depends(get_image_cache)
) -> RegisterImageResponse:
    """Decode and cache an image once, returning an id that /api/segment and
    the masked-edit endpoints accept in place of `image` — lets a client
    upload a large image a single time and reuse it across many requests
    (e.g. repeated segmentation attempts against the same source image)
    instead of re-sending the full base64 payload every time.
    """
    image_id = cache.put(decode_image(body.image))
    return RegisterImageResponse(image_id=image_id)


@router.post("/segment", response_model=SegmentationResponse)
def segment(
    body: SegmentRequest,
    processor: ImageProcessor = Depends(get_processor),
    cache: ImageCache = Depends(get_image_cache),
) -> SegmentationResponse:
    image = _resolve_image(body.image, body.image_id, cache)
    result = processor.segment(image, body.selection.to_domain())
    return _segmentation_response(result)


@router.post("/remove-object", response_model=EditResponse)
def remove_object(
    body: RemoveObjectRequest, processor: ImageProcessor = Depends(get_processor)
) -> EditResponse:
    result = processor.remove_object(
        decode_image(body.image),
        selection=_selection(body.selection),
        mask=decode_mask(body.mask) if body.mask else None,
        mask_options=body.mask_options.to_domain() if body.mask_options else None,
        generation_options=body.generation_options.to_domain() if body.generation_options else None,
    )
    return _edit_response(result)


@router.post("/replace-object", response_model=EditResponse)
def replace_object(
    body: ReplaceObjectRequest, processor: ImageProcessor = Depends(get_processor)
) -> EditResponse:
    result = processor.replace_object(
        decode_image(body.image),
        body.prompt,
        selection=_selection(body.selection),
        mask=decode_mask(body.mask) if body.mask else None,
        mask_options=body.mask_options.to_domain() if body.mask_options else None,
        generation_options=body.generation_options.to_domain() if body.generation_options else None,
    )
    return _edit_response(result)


@router.post("/replace-background", response_model=GenerationResponse)
def replace_background(
    body: ReplaceBackgroundRequest, processor: ImageProcessor = Depends(get_processor)
) -> GenerationResponse:
    result = processor.replace_background(
        decode_image(body.image),
        body.prompt,
        generation_options=body.generation_options.to_domain() if body.generation_options else None,
    )
    return _generation_response(result)


@router.post("/add-object/prompt", response_model=GenerationResponse)
def add_object_by_prompt(
    body: AddObjectByPromptRequest, processor: ImageProcessor = Depends(get_processor)
) -> GenerationResponse:
    result = processor.add_object_by_prompt(
        decode_image(body.image),
        body.prompt,
        placement=body.placement.to_domain() if body.placement else None,
        generation_options=body.generation_options.to_domain() if body.generation_options else None,
    )
    return _generation_response(result)


@router.post("/add-object/reference", response_model=EditResponse)
def add_object_by_reference(
    body: AddObjectByReferenceRequest, processor: ImageProcessor = Depends(get_processor)
) -> EditResponse:
    result = processor.add_object_by_reference(
        decode_image(body.image),
        decode_image(body.reference),
        placement=body.placement.to_domain() if body.placement else None,
        mask=decode_mask(body.mask) if body.mask else None,
        reference_mask=decode_mask(body.reference_mask) if body.reference_mask else None,
        mask_options=body.mask_options.to_domain() if body.mask_options else None,
        generation_options=body.generation_options.to_domain() if body.generation_options else None,
    )
    return _edit_response(result)


@router.post("/prompt-edit", response_model=GenerationResponse)
def prompt_edit(
    body: PromptEditRequest, processor: ImageProcessor = Depends(get_processor)
) -> GenerationResponse:
    result = processor.prompt_edit(
        decode_image(body.image),
        body.prompt,
        generation_options=body.generation_options.to_domain() if body.generation_options else None,
    )
    return _generation_response(result)


@router.post("/generate", response_model=GenerationResponse)
def generate_image(
    body: GenerateImageRequest, processor: ImageProcessor = Depends(get_processor)
) -> GenerationResponse:
    result = processor.generate_image(
        body.prompt,
        width=body.width,
        height=body.height,
        generation_options=body.generation_options.to_domain() if body.generation_options else None,
    )
    return _generation_response(result)


@router.post("/outpaint", response_model=EditResponse)
def outpaint(body: OutpaintRequest, processor: ImageProcessor = Depends(get_processor)) -> EditResponse:
    result = processor.outpaint(
        decode_image(body.image),
        body.prompt,
        body.margins.to_domain(),
        generation_options=body.generation_options.to_domain() if body.generation_options else None,
    )
    return _edit_response(result)


@router.post("/upscale", response_model=GenerationResponse)
def upscale(body: UpscaleRequest, processor: ImageProcessor = Depends(get_processor)) -> GenerationResponse:
    result = processor.upscale(
        decode_image(body.image),
        options=body.options.to_domain() if body.options else None,
    )
    return _generation_response(result)
