# AI Image Editor — Installation and Usage Guide

This guide covers local installation, development, real-GPU inference, testing, and troubleshooting. Start with the [project showcase](README.md) for a visual overview, or use the [technical reference](TECHNICAL_REFERENCE.md) for architecture and API examples.

## Contents

1. [Architecture overview](#1-architecture-overview)
2. [System requirements](#2-system-requirements)
3. [Backend installation](#3-backend-installation)
4. [Frontend installation](#4-frontend-installation)
5. [Development workflow](#5-development-workflow)
6. [Real-GPU deployment](#6-real-gpu-deployment)
7. [Environment validation](#7-environment-validation)
8. [Unit tests](#8-unit-tests)
9. [GPU smoke tests](#9-gpu-smoke-tests)
10. [Jupyter notebook](#10-jupyter-notebook)
11. [Project structure](#11-project-structure)
12. [Environment variables](#12-environment-variables)
13. [Troubleshooting](#13-troubleshooting)

## 1. Architecture overview

The repository contains two main applications:

```text
AIImageEditor/
├── src/inpaint_core/   Python core, model adapters, and FastAPI service
└── frontend/           Next.js studio built with React and TypeScript
```

Request flow:

```text
Browser
  └─► Next.js frontend (port 3000)
        └─► FastAPI backend (port 8000)
              └─► ImageProcessor
                    ├── SAM2          object segmentation
                    ├── OmniPaint     object removal and insertion
                    ├── FLUX.1 Fill   replacement and outpainting
                    ├── FLUX.2 Klein  prompt editing and generation
                    └── Real-ESRGAN   upscaling
```

| # | Workflow | Model/backend |
|---:|---|---|
| 1 | Object removal | SAM2 + OmniPaint |
| 2 | Object replacement | SAM2 + FLUX.1 Fill |
| 3 | Background replacement | FLUX.2 Klein 4B |
| 4 | Add object by prompt | FLUX.2 Klein 4B |
| 5 | Add object by reference | SAM2 + OmniPaint Insertion |
| 6 | Prompt-based editing | FLUX.2 Klein 4B |
| 7 | Text-to-image | FLUX.2 Klein 4B |
| 8 | Outpainting | FLUX.1 Fill |
| 9 | Upscaling | Real-ESRGAN with optional GFPGAN |

## 2. System requirements

| Component | Requirement |
|---|---|
| Operating system | Windows 10/11, Linux, or WSL2 |
| Python | 3.12 |
| Environment manager | Miniconda or Anaconda |
| Node.js | 18 or newer |
| npm | 9 or newer |
| GPU | NVIDIA GPU with CUDA support; approximately 32 GB VRAM for every full workflow |
| Git | Required to clone OmniPaint and install SAM2 |

A GPU is not required for frontend development, API contract tests, or unit tests. Use the fake backend described in [Development workflow](#5-development-workflow).

## 3. Backend installation

### 3.1 Create the Conda environment

```bash
conda env create -f environment.yml
```

If the environment already exists, update it instead:

```bash
conda env update -f environment.yml --prune
```

Activate it before running any Python command:

```bash
conda activate imageinpaint
```

### 3.2 Install SAM2

```bash
pip install git+https://github.com/facebookresearch/sam2.git
```

### 3.3 Install the project package

```bash
pip install -e . --no-deps
```

`--no-deps` is intentional: the compatible dependency set is managed by `environment.yml`.

### 3.4 Install OmniPaint

The helper script clones the upstream runtime into `third_party/OmniPaint`:

```bash
python scripts/install_omnipaint.py
```

Run this command once. It exits with an error if the target directory already exists.

Do not run `third_party/OmniPaint/scripts/setup.sh` inside this environment. The upstream script pins versions of Diffusers and PEFT that conflict with the FLUX.2 integration. This project uses one tested compatibility set for all backends and applies a small runtime compatibility shim for an internal Diffusers symbol used by OmniPaint.

### 3.5 Install CUDA-enabled PyTorch on native Windows

On native Windows, install PyTorch from the matching CUDA package index:

```bash
pip uninstall torch torchvision
pip install torch==2.14.0 torchvision==0.29.0 --index-url https://download.pytorch.org/whl/cu130
```

On Linux and WSL2, use the packages specified by `environment.yml` unless your CUDA environment requires a platform-specific build.

### 3.6 Configure Hugging Face access

FLUX and OmniPaint checkpoints may require approved Hugging Face access. Create a token at [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens), accept the relevant model licenses, and expose the token in your shell.

Linux/macOS:

```bash
export HF_TOKEN=hf_xxxxxxxxxxxxxxxxxxxx
```

Windows PowerShell:

```powershell
$env:HF_TOKEN = "hf_xxxxxxxxxxxxxxxxxxxx"
```

When `artifacts.prefetch_on_init` is enabled, processor initialization downloads configured artifacts into the local caches. Pipelines are loaded into memory only when their first operation runs.

## 4. Frontend installation

Install the Next.js dependencies:

```bash
cd frontend
npm install
```

Create a local environment file from the template.

Windows Command Prompt:

```bat
copy frontend\.env.local.example frontend\.env.local
```

Linux/macOS:

```bash
cp frontend/.env.local.example frontend/.env.local
```

Default configuration:

```env
NEXT_PUBLIC_API_URL=http://localhost:8000
NEXT_PUBLIC_USE_MOCK_API=false
```

Set `NEXT_PUBLIC_USE_MOCK_API=true` only when you want the browser application to use its in-process mock without a running FastAPI service.

## 5. Development workflow

### UI development without a GPU

Use two terminals.

Terminal 1 — start FastAPI with in-memory fake backends:

```bash
conda activate imageinpaint
python scripts/run_api.py --fake
```

The API is available at `http://localhost:8000`.

Terminal 2 — start Next.js:

```bash
cd frontend
npm run dev
```

Open:

- Landing page: `http://localhost:3000`
- Editing studio: `http://localhost:3000/studio`
- API health check: `http://localhost:8000/api/health`

### Frontend-only mock mode

Set the following value in `frontend/.env.local`:

```env
NEXT_PUBLIC_USE_MOCK_API=true
```

Then run only the frontend:

```bash
cd frontend
npm run dev
```

## 6. Real-GPU deployment

### 6.1 Start the inference API

```bash
conda activate imageinpaint
python scripts/run_api.py --config configs/default.yaml
```

The first run can take a long time because configured checkpoints are downloaded and cached. The first request to each backend also includes its in-memory model load.

`run_api.py` accepts:

| Option | Default | Description |
|---|---|---|
| `--fake` | disabled | Use fake backends without CUDA or checkpoints |
| `--config PATH` | `configs/default.yaml` | Configuration file |
| `--host` | `127.0.0.1` | Interface to bind |
| `--port` | `8000` | Port to bind |

### 6.2 Build and run the frontend

```bash
cd frontend
npm run build
npm run start
```

For a remote deployment, update `NEXT_PUBLIC_API_URL` and `INPAINT_API_CORS_ORIGINS` before building and starting the applications.

## 7. Environment validation

Run the compatibility checker after installation:

```bash
conda activate imageinpaint

# Safe on a machine without an NVIDIA GPU
python scripts/check_environment.py

# Also require CUDA and BF16 support
python scripts/check_environment.py --require-cuda
```

The checker validates:

- Python and pinned library versions
- NumPy, SciPy, and OpenCV imports
- FLUX/Diffusers APIs
- OmniPaint's private Diffusers dependency
- SAM2 APIs
- Real-ESRGAN and BasicSR APIs
- CUDA availability and bfloat16 support when requested

## 8. Unit tests

Unit and API contract tests use fake backends. They do not need CUDA or checkpoint downloads.

```bash
conda activate imageinpaint
python -m pytest -q
```

Use verbose output when diagnosing a failure:

```bash
python -m pytest -v
```

`tests/test_api.py` sends requests to every FastAPI endpoint through `TestClient`, while the rest of the suite verifies preprocessing, operations, model lifecycle, and backend contracts.

## 9. GPU smoke tests

Test real inference one operation at a time after the model artifacts are available:

```bash
conda activate imageinpaint

python scripts/smoke_test.py --mode segment
python scripts/smoke_test.py --mode remove
python scripts/smoke_test.py --mode all --reference assets/cat2.jpg
```

Available modes:

| Mode | Operation |
|---|---|
| `segment` | Object segmentation |
| `remove` | Object removal |
| `replace` | Object replacement |
| `background` | Background replacement |
| `add-prompt` | Prompt-guided insertion |
| `add-reference` | Reference-guided insertion |
| `prompt-edit` | Prompt-based editing |
| `generate` | Text-to-image |
| `outpaint` | Outpainting |
| `upscale` | Upscaling |
| `all` | Run every mode |

Images are written to `outputs/smoke/`; memory statistics are written to `outputs/smoke/memory.json`.

## 10. Jupyter notebook

The CV showcase notebook runs all nine modes, displays processed inputs, masks, and outputs, and records image dimensions, parameters, runtime, and CUDA peak memory.

```bash
conda activate imageinpaint
jupyter lab notebooks/cv-showcase-all-modes.ipynb
```

Run the setup cells first, then execute one workflow cell at a time. Each workflow is profiled as a cold start and releases its models after completion. Generated artifacts are written to `outputs/showcase/`.

| Section | Workflow | Saved artifacts |
|---:|---|---|
| 1 | Object removal | input, SAM2 mask, output, run metadata |
| 2 | Object replacement | input, SAM2 mask, output, run metadata |
| 3 | Background replacement | input, output, run metadata |
| 4 | Add object by prompt | input, output, run metadata |
| 5 | Add object by reference | target, reference, two masks, output, run metadata |
| 6 | Prompt-based editing | input, output, run metadata |
| 7 | Text-to-image | output and run metadata |
| 8 | Outpainting | input, output, run metadata |
| 9 | Upscaling | input, output, run metadata |

The notebook uses a 768 px longest-side limit for OmniPaint target images and up to 2048 px for other image workflows. Reference images for OmniPaint are prepared separately. Selection points and placement boxes are drawn only on display copies, so overlays never enter model inference.

## 11. Project structure

```text
AIImageEditor/
├── assets/                       demo source images
├── configs/
│   └── default.yaml              model and memory configuration
├── docs/assets/showcase/         versioned README gallery images
├── frontend/                     Next.js application
│   └── src/
│       ├── app/                  App Router pages
│       ├── components/           React components
│       ├── hooks/                application hooks
│       └── lib/api/              real and mock API clients
├── notebooks/
│   └── cv-showcase-all-modes.ipynb
├── outputs/                      generated artifacts; gitignored
├── scripts/
│   ├── check_environment.py      dependency and CUDA validation
│   ├── install_omnipaint.py      clone the OmniPaint source
│   ├── run_api.py                launch FastAPI
│   └── smoke_test.py             real-inference smoke tests
├── src/inpaint_core/
│   ├── api/                      FastAPI application and schemas
│   ├── backends/                 SAM2, FLUX, OmniPaint, ESRGAN adapters
│   ├── models/                   model lifecycle management
│   ├── operations/               model-independent editing workflows
│   ├── testing/                  fake implementations
│   ├── config.py
│   └── processor.py              public ImageProcessor facade
├── tests/                        pytest suite
├── third_party/OmniPaint/        installed separately; gitignored
├── weights/                      downloaded Real-ESRGAN weights
├── environment.yml
├── README.md                     visual project overview
├── TECHNICAL_REFERENCE.md        architecture and Python examples
└── GUIDE.md                      this installation guide
```

## 12. Environment variables

### Frontend (`frontend/.env.local`)

| Variable | Default | Description |
|---|---|---|
| `NEXT_PUBLIC_API_URL` | `http://localhost:8000` | FastAPI base URL |
| `NEXT_PUBLIC_USE_MOCK_API` | `false` | Use the frontend mock instead of HTTP |

### Backend shell

| Variable | Description |
|---|---|
| `HF_TOKEN` | Hugging Face access token for gated checkpoints |
| `INPAINT_API_CORS_ORIGINS` | Comma-separated allowed origins; defaults to `http://localhost:3000` |

Never commit `.env.local`, access tokens, or local checkpoint paths.

## 13. Troubleshooting

### OmniPaint source not found

```text
FileNotFoundError: OmniPaint source not found: .../third_party/OmniPaint
```

Install the upstream source:

```bash
python scripts/install_omnipaint.py
```

### Diffusers, PEFT, or other version conflicts

Confirm that `imageinpaint` is active and restore the pinned environment:

```bash
conda activate imageinpaint
conda env update -f environment.yml --prune
pip install -e . --no-deps
```

Do not run OmniPaint's upstream setup script in the same environment.

### CUDA out of memory

In `configs/default.yaml`:

- Set `memory.policy: sequential` so the current model is released before another backend loads.
- Reduce OmniPaint's `max_image_side`, for example from 768 to 512.
- Enable the supported CPU-offload option for large backends.
- Test one mode per process while measuring the actual peak on the deployment GPU.

### Frontend cannot connect to the backend

Check the following:

1. `http://localhost:8000/api/health` returns `{"status":"ok"}`.
2. `frontend/.env.local` contains `NEXT_PUBLIC_API_URL=http://localhost:8000`.
3. `NEXT_PUBLIC_USE_MOCK_API` is `false` when using FastAPI.
4. The frontend origin is included in `INPAINT_API_CORS_ORIGINS`.
5. Restart Next.js after changing a `NEXT_PUBLIC_*` variable.

### Scientific-package conflicts

Avoid mixing unrelated pip wheels with Conda packages for NumPy, SciPy, and OpenCV. Restore the versions in `environment.yml`, then rerun `scripts/check_environment.py`.

### CUDA is not detected on Windows

Install the native Windows build from the configured CUDA index, then validate it:

```bash
pip install torch==2.14.0 torchvision==0.29.0 --index-url https://download.pytorch.org/whl/cu130
python scripts/check_environment.py --require-cuda
```

## Quick reference

```bash
# One-time setup
conda env create -f environment.yml
conda activate imageinpaint
pip install git+https://github.com/facebookresearch/sam2.git
pip install -e . --no-deps
python scripts/install_omnipaint.py
cd frontend && npm install && cd ..

# Development: terminal 1
conda activate imageinpaint
python scripts/run_api.py --fake

# Development: terminal 2
cd frontend
npm run dev

# Real inference API
conda activate imageinpaint
python scripts/run_api.py --config configs/default.yaml

# Verification
python scripts/check_environment.py --require-cuda
python -m pytest -q
python scripts/smoke_test.py --mode segment
```
