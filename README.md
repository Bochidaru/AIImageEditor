# ImageEditor

Notebook-first image editing core for three initial modes:

- Object removal
- Object replacement
- Background replacement

The package keeps web frameworks out of the core. `ImageProcessor` is the public facade; segmentation, masked generation, mask transforms, compositing, and model lifecycle are separate components.

## Setup

```powershell
conda env create -f environment.yml || conda env update -f environment.yml --prune
conda activate imageinpaint
pip install git+https://github.com/facebookresearch/sam2.git
pip install -e . --no-deps
pytest
```

Select the `image-editor` environment as the kernel for `notebooks/object-editing-demo.ipynb`.

## Architecture

```text
ImageProcessor
├── Segmenter       image + selection -> masks
├── MaskProcessor   mask -> edit/composite masks
├── MaskedEditor    image + mask + prompt -> generated image
├── Compositor      original + generated + alpha -> final image
└── ModelManager    lazy loading and model lifecycle
```

Model identifiers and memory policy live in `configs/default.yaml`. The SAM3 adapter intentionally exposes an integration boundary because the exact checkpoint/API must be selected before implementation.

