# ImageEditor

An extensible image-editing core with a thin `ImageProcessor` facade. The
project deliberately separates business operations from model-specific code.

## Architecture

```text
ImageProcessor
  -> operations/                 what the user wants to do
       object_removal            -> OmniPaint
       object_replacement        -> FLUX Fill
       background_replacement    -> FLUX Fill
       object_insertion          -> FLUX Fill / OmniPaint
       prompt_edit               -> FLUX Kontext
       generation                -> base FLUX
       outpainting               -> FLUX Fill
       upscaling                 -> Real-ESRGAN
  -> backends/                   how a specific model is called
  -> ModelManager                lazy load, resident/sequential lifecycle
```

`backends/protocols.py` contains structural interfaces, not abstract parent
classes. Backends do not inherit from them. This keeps operations replaceable
and easy to unit-test with fake implementations.

Masked diffusion always receives the complete preprocessed image, padded to a
multiple of `processing.size_multiple`. Cropping and automatic preprocessing
inside `ImageProcessor` are disabled. Compositing is also disabled: masked
operations return the model's complete generated image after padding is removed.

## Setup

```bash
conda env create -f environment.yml
conda activate imageinpaint
pip install -e .
pip install git+https://github.com/facebookresearch/sam2.git
python scripts/install_omnipaint.py
```

FLUX and OmniPaint repositories may require Hugging Face access approval and
an authenticated `HF_TOKEN`. `scripts/install_omnipaint.py` installs the
official custom inference runtime; OmniPaint is not a drop-in Diffusers LoRA.

## Public API

```python
from pathlib import Path
from inpaint_core import (
    BoxPrompt,
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

processor = ImageProcessor.from_config(root / "configs/default.yaml")
selection = PointPrompt(points=[(480, 641)], labels=[1])
image, selection = preprocess_image(
    root / "assets/cat.jpg",
    selection,
    max_side=processor.config.processing.max_image_side,
)

removed = processor.remove_object(image, selection=selection)
replaced = processor.replace_object(image, "a golden retriever", selection=selection)
background = processor.replace_background(
    image, "a quiet beach at sunset", foreground_selection=selection
)

inserted_prompt = processor.add_object_by_prompt(
    image, "a red ball", placement=BoxPrompt(50, 50, 250, 250)
)

reference = preprocess_image_only(root / "assets/reference.png", max_side=512)
inserted_reference = processor.add_object_by_reference(
    image,
    reference,
    placement=BoxPrompt(50, 50, 250, 250),
)

edited = processor.prompt_edit(image, "Turn the scene into winter")
generated = processor.generate_image("A cinematic mountain lake", width=1024, height=1024)
extended = processor.outpaint(
    image,
    "Continue the scene naturally",
    OutpaintMargins(left=128, right=128),
)
upscaled = processor.upscale(image, options=UpscaleOptions(scale=4, tile=512))

processor.save_memory_stats(root / "outputs/memory.json")
processor.release_models()
```

For reference insertion, pass `reference_mask=` when the reference still has a
background. The core turns pixels outside that mask white before OmniPaint sees
the subject. Placement should preferably be one connected mask/box.

## Memory policy

- `sequential`: loading a new backend releases all currently registered models.
- `resident`: every loaded backend remains registered until `release_models()`.

Model placement and offload remain backend-specific. `ModelManager` owns model
lifetime, but does not know how Diffusers, SAM2, or Real-ESRGAN perform inference.

## Verification

```bash
python -m pytest -q
```

Unit tests use fake backends and therefore do not download checkpoints. Real GPU
inference should be tested one mode at a time with `memory.policy: sequential`.

```bash
python scripts/smoke_test.py --mode segment
python scripts/smoke_test.py --mode remove
python scripts/smoke_test.py --mode all --reference assets/reference.png
```
