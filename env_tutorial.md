```bash
conda env create -f environment.yml || conda env update -f environment.yml --prune
conda activate imageinpaint
pip install git+https://github.com/facebookresearch/sam2.git
pip install -e . --no-deps
```