```bash
conda env create -f environment.yml || conda env update -f environment.yml --prune
conda activate imageinpaint
pip install git+https://github.com/facebookresearch/sam2.git
pip install -e . --no-deps
python scripts/install_omnipaint.py
```

run this for cleaning cache (weights, checkpoints, ...)
```bash
export HF_HOME="$PWD/.runtime-cache/huggingface"
export TORCH_HOME="$PWD/.runtime-cache/torch"
export XDG_CACHE_HOME="$PWD/.runtime-cache/xdg"
```