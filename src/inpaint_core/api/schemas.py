from __future__ import annotations

from typing import Annotated, Literal, Union

from pydantic import BaseModel, ConfigDict, Field

from ..types import BoxPrompt, GenerationOptions, MaskOptions, OutpaintMargins, PointPrompt, UpscaleOptions

# ---------------------------------------------------------------------------
# Selection prompts (mirrors frontend/src/lib/types.ts's `kind` discriminator)
# ---------------------------------------------------------------------------


class PointPromptIn(BaseModel):
    kind: Literal["point"] = "point"
    points: list[tuple[float, float]]
    labels: list[Literal[0, 1]]

    def to_domain(self) -> PointPrompt:
        return PointPrompt(points=self.points, labels=list(self.labels))


class BoxPromptIn(BaseModel):
    kind: Literal["box"] = "box"
    x1: float
    y1: float
    x2: float
    y2: float

    def to_domain(self) -> BoxPrompt:
        return BoxPrompt(x1=self.x1, y1=self.y1, x2=self.x2, y2=self.y2)


SelectionIn = Annotated[Union[PointPromptIn, BoxPromptIn], Field(discriminator="kind")]


# ---------------------------------------------------------------------------
# Options
# ---------------------------------------------------------------------------


class MaskOptionsIn(BaseModel):
    threshold: int = 127
    dilate: int = 0
    erode: int = 0
    candidate_index: int | None = None

    def to_domain(self) -> MaskOptions:
        return MaskOptions(
            threshold=self.threshold,
            dilate=self.dilate,
            erode=self.erode,
            candidate_index=self.candidate_index,
        )


class GenerationOptionsIn(BaseModel):
    seed: int = 42
    # None (the default) lets each backend use its own tuned step count
    # (e.g. Flux2 Klein's 4 vs Flux Fill/OmniPaint's 28 — see
    # configs/default.yaml) instead of a single hardcoded value overriding
    # all of them regardless of which model actually handles the request.
    num_inference_steps: int | None = None
    guidance_scale: float | None = None

    def to_domain(self) -> GenerationOptions:
        return GenerationOptions(
            seed=self.seed,
            num_inference_steps=self.num_inference_steps,
            guidance_scale=self.guidance_scale,
        )


class OutpaintMarginsIn(BaseModel):
    left: int = 0
    top: int = 0
    right: int = 0
    bottom: int = 0

    def to_domain(self) -> OutpaintMargins:
        return OutpaintMargins(left=self.left, top=self.top, right=self.right, bottom=self.bottom)


class UpscaleOptionsIn(BaseModel):
    scale: int = 4
    face_enhance: bool = False
    tile: int = 0
    tile_pad: int = 10
    pre_pad: int = 0

    def to_domain(self) -> UpscaleOptions:
        return UpscaleOptions(
            scale=self.scale,
            face_enhance=self.face_enhance,
            tile=self.tile,
            tile_pad=self.tile_pad,
            pre_pad=self.pre_pad,
        )


# ---------------------------------------------------------------------------
# Request bodies — one per ImageProcessor operation
# ---------------------------------------------------------------------------


class SegmentRequest(BaseModel):
    image: str | None = None
    image_id: str | None = None
    selection: SelectionIn


class _ImageEditRequestBase(BaseModel):
    """Fields shared by every operation that edits an existing image via a
    mask (selection, or placement + mask) rather than a direct prompt-only
    instruction edit.

    Deliberately does NOT include a `selection`/`mask`-shaped field: each
    operation locates its region differently (selection, or placement +
    mask), so those stay on the individual subclasses below rather than
    being faked into a common shape.

    Also deliberately has no `image_id` field: unlike /api/segment (fired on
    every point/box a user tries against the same unchanged image, which is
    what /api/images exists to avoid re-uploading for), these "Run" edit
    operations fire once per explicit user action and no frontend caller
    sends image_id for them today — keep the request surface matching what's
    actually exercised instead of speculative unused capability.
    """

    image: str
    mask_options: MaskOptionsIn | None = None
    generation_options: GenerationOptionsIn | None = None


class _MaskedRequestBase(_ImageEditRequestBase):
    selection: SelectionIn | None = None
    mask: str | None = None


class RemoveObjectRequest(_MaskedRequestBase):
    pass


class ReplaceObjectRequest(_MaskedRequestBase):
    prompt: str


class ReplaceBackgroundRequest(BaseModel):
    """No mask/selection: operations/background_replacement.py now edits
    directly from a prompt instruction (Flux2 Klein) instead of compositing
    over an inverted foreground mask.

    extra="forbid" so a client still on the old contract (foreground_mask,
    foreground_selection, mask_options) gets a 422 instead of having that
    field silently dropped and getting a full-image edit it didn't ask for.
    """

    model_config = ConfigDict(extra="forbid")

    image: str
    prompt: str
    generation_options: GenerationOptionsIn | None = None


class AddObjectByPromptRequest(BaseModel):
    """No mask/mask_options: operations/object_insertion.py's by_prompt now
    edits directly from a prompt instruction (Flux2 Klein); `placement` only
    steers the instruction's wording (see _placement_description), it is not
    resolved into a mask.

    extra="forbid" so a client still sending `mask`/`mask_options` (the old
    contract) gets a 422 instead of the field being silently ignored.
    """

    model_config = ConfigDict(extra="forbid")

    image: str
    prompt: str
    placement: BoxPromptIn | None = None
    generation_options: GenerationOptionsIn | None = None


class AddObjectByReferenceRequest(_ImageEditRequestBase):
    reference: str
    placement: BoxPromptIn | None = None
    mask: str | None = None
    reference_mask: str | None = None


class PromptEditRequest(BaseModel):
    image: str
    prompt: str
    generation_options: GenerationOptionsIn | None = None


class GenerateImageRequest(BaseModel):
    prompt: str
    width: int = 1024
    height: int = 1024
    generation_options: GenerationOptionsIn | None = None


class OutpaintRequest(BaseModel):
    image: str
    prompt: str
    margins: OutpaintMarginsIn
    generation_options: GenerationOptionsIn | None = None


class UpscaleRequest(BaseModel):
    image: str
    options: UpscaleOptionsIn | None = None


class RegisterImageRequest(BaseModel):
    image: str


# ---------------------------------------------------------------------------
# Responses
# ---------------------------------------------------------------------------


class SegmentationResponse(BaseModel):
    masks: list[str]
    scores: list[float]
    best_index: int
    metadata: dict = Field(default_factory=dict)


class GenerationResponse(BaseModel):
    image: str
    seed: int
    metadata: dict = Field(default_factory=dict)


class EditResponse(BaseModel):
    image: str
    mask: str
    generated_image: str | None = None
    metadata: dict = Field(default_factory=dict)


class ErrorResponse(BaseModel):
    detail: str


class RegisterImageResponse(BaseModel):
    image_id: str
