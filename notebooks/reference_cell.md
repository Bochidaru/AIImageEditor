### Setup cell

```python
from pathlib import Path

from IPython.display import display
from PIL import Image

from inpaint_core import (
    BoxPrompt,
    GenerationOptions,
    ImageProcessor,
    OutpaintMargins,
    PointPrompt,
    UpscaleOptions,
    preprocess_image,
    preprocess_image_only,
)

root = Path.cwd()
if root.name == "notebooks":
    root = root.parent

processor = ImageProcessor.from_config(root / "configs" / "default.yaml")

raw_image = preprocess_image_only(root / "assets" / "cat.jpg")
selection = PointPrompt(
    points=[(raw_image.shape[1] / 2, raw_image.shape[0] / 2)],
    labels=[1],
)
omnipaint_max_side = int(processor.config.omnipaint.options["max_image_side"])
image, selection = preprocess_image(
    raw_image,
    selection,
    # The shared demo image is also used by OmniPaint modes.
    max_side=min(processor.config.processing.max_image_side, omnipaint_max_side),
)

segmentation = processor.segment(image, selection)
object_mask = segmentation.best_mask

output_dir = root / "outputs" / "notebook"
output_dir.mkdir(parents=True, exist_ok=True)


def show_and_save(result, filename):
    output_path = output_dir / filename
    Image.fromarray(result.image).save(output_path)
    display(Image.fromarray(result.image))
    print(output_path)


display(Image.fromarray(image))
display(Image.fromarray(object_mask))
```

### Cell 1 — Object Removal · OmniPaint

```python
removed = processor.remove_object(
    image,
    mask=object_mask,
)
show_and_save(removed, "01-object-removal.png")
```

### Cell 2 — Object Replacement · FLUX Fill

```python
replaced = processor.replace_object(
    image,
    "a golden retriever sitting in the same position",
    mask=object_mask,
)
show_and_save(replaced, "02-object-replacement.png")
```

### Cell 3 — Background Replacement · FLUX.2 Klein 4B

```python
background = processor.replace_background(
    image,
    (
        "a busy modern office with desks, computer monitors, chairs, and "
        "workers in the distance, natural indoor lighting"
    ),
    generation_options=GenerationOptions(
        seed=123,
        num_inference_steps=4,
        guidance_scale=1.0,
    ),
)

show_and_save(background, "03-background-replacement.png")
```

### Cell 4 — Add Object · Prompt or Reference Image

Add by prompt uses FLUX.2 Klein direct editing. `placement` is converted into
a natural-language location/size hint; it is not a hard pixel mask:

```python
placement = BoxPrompt(
    x1=image.shape[1] * 0.05,
    y1=image.shape[0] * 0.55,
    x2=image.shape[1] * 0.30,
    y2=image.shape[0] * 0.90,
)

inserted_prompt = processor.add_object_by_prompt(
    image,
    "a small red ball resting naturally on the ground",
    placement=placement,
    generation_options=GenerationOptions(seed=42, num_inference_steps=4),
)
show_and_save(inserted_prompt, "04a-add-object-prompt.png")
```

Add by reference uses OmniPaint. Replace `assets/cat2.jpg` with another
reference image when needed:

```python
raw_reference = preprocess_image_only(root / "assets" / "cat2.jpg")
reference_selection = PointPrompt(
    points=[(raw_reference.shape[1] / 2, raw_reference.shape[0] / 2)],
    labels=[1],
)
reference, reference_selection = preprocess_image(
    raw_reference,
    reference_selection,
    max_side=512,
)
reference_mask = processor.segment(reference, reference_selection).best_mask

inserted_reference = processor.add_object_by_reference(
    image,
    reference,
    placement=placement,
    reference_mask=reference_mask,
)
show_and_save(inserted_reference, "04b-add-object-reference.png")
```

### Cell 5 — Prompt Edit · FLUX.2 Klein 4B

```python
edited = processor.prompt_edit(
    image,
    "Turn the scene into winter while preserving the composition",
    generation_options=GenerationOptions(seed=42, num_inference_steps=4),
)
show_and_save(edited, "05-prompt-edit.png")
```

### Cell 6 — Text-to-Image · FLUX.2 Klein 4B

```python
generated = processor.generate_image(
    "A cinematic mountain lake at sunrise, realistic photography",
    width=1024,
    height=1024,
    generation_options=GenerationOptions(seed=42, num_inference_steps=4),
)
show_and_save(generated, "06-generate-image.png")
```

### Cell 7 — Outpainting · FLUX Fill

```python
extended = processor.outpaint(
    image,
    "Continue the original scene naturally on both sides",
    OutpaintMargins(left=128, right=128),
)
show_and_save(extended, "07-outpaint.png")
```

### Cell 8 — Upscaling · Real-ESRGAN

```python
upscaled = processor.upscale(
    image,
    options=UpscaleOptions(scale=4, tile=512),
)
show_and_save(upscaled, "08-upscale.png")
```

Optional cleanup and memory report:

```python
processor.save_memory_stats(root / "outputs" / "memory.json")
processor.release_models()
```

For reference insertion, `reference_mask` removes the reference background
before OmniPaint receives the subject. Placement should preferably be one
connected mask or box.

## Memory policy

- `sequential`: loading a new backend releases all currently registered models.
- `resident`: every loaded backend remains registered until `release_models()`.

Model placement and offload remain backend-specific. `ModelManager` owns model
lifetime, but does not know how Diffusers, SAM2, or Real-ESRGAN perform inference.

OmniPaint has a separate 32 GB-oriented profile: its target image is limited to
768 px on the longest side (the upstream maximum is 1024), text encoders are
not loaded, static prompt embeddings stay on CPU while idle, and regular BF16
weights use model CPU offload. Pipeline-level bitsandbytes INT8 is disabled for
OmniPaint because loading its LoRAs can leave quantized parameters on the meta
device, while its custom transformer forward also bypasses Diffusers' normal
offload hook. Available VRAM still depends on image size, CUDA allocator
behavior, and library versions. Verify the actual peak via
`save_memory_stats()` on the deployment GPU.

## Verification

```bash
python -m pytest -q
```

Unit tests use fake backends and do not download checkpoints. Real GPU inference
should be tested one mode at a time with `memory.policy: sequential`.

```bash
python scripts/smoke_test.py --mode segment
python scripts/smoke_test.py --mode remove
python scripts/smoke_test.py --mode all --reference assets/cat2.jpg
```
