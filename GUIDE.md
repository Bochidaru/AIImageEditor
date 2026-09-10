# GUIDE.md — Hướng dẫn cài đặt & chạy dự án ImageEditor

> Đây là bản hướng dẫn tiếng Việt, chi tiết theo từng bước. Tài liệu tham
> chiếu chính thức (tiếng Anh) là [README.md](README.md) — nếu hai tài liệu
> có chỗ nào mâu thuẫn, hãy tin theo README.md.

## Mục lục

1. [Tổng quan kiến trúc](#1-tổng-quan-kiến-trúc)
2. [Yêu cầu hệ thống](#2-yêu-cầu-hệ-thống)
3. [Cài đặt Backend (Python)](#3-cài-đặt-backend-python)
4. [Cài đặt Frontend (Next.js)](#4-cài-đặt-frontend-nextjs)
5. [Chạy dự án (Development)](#5-chạy-dự-án-development)
6. [Chạy dự án (Production với GPU thật)](#6-chạy-dự-án-production-với-gpu-thật)
7. [Kiểm tra môi trường](#7-kiểm-tra-môi-trường)
8. [Chạy Unit Tests](#8-chạy-unit-tests)
9. [Smoke Test (GPU thật)](#9-smoke-test-gpu-thật)
10. [Sử dụng Jupyter Notebook](#10-sử-dụng-jupyter-notebook)
11. [Cấu trúc thư mục](#11-cấu-trúc-thư-mục)
12. [Các biến môi trường](#12-các-biến-môi-trường)
13. [Xử lý lỗi thường gặp](#13-xử-lý-lỗi-thường-gặp)

---

## 1. Tổng quan kiến trúc

Dự án gồm hai phần chính:

```
ImageEditor/
├── src/inpaint_core/   ← Backend Python (FastAPI + AI models)
└── frontend/           ← Frontend Next.js (React 19, TypeScript)
```

**Luồng hoạt động:**

```
Trình duyệt
  └─► Next.js (frontend, port 3000)
        └─► FastAPI (backend, port 8000)
              └─► ImageProcessor
                    ├── SAM2          (phân vùng đối tượng)
                    ├── OmniPaint     (xóa / chèn đối tượng)
                    ├── FLUX.1 Fill   (thay thế đối tượng, outpainting)
                    ├── FLUX.2 Klein  (chỉnh sửa theo prompt, tạo ảnh)
                    └── Real-ESRGAN   (upscaling)
```

| # | Chức năng | Model |
|---|-----------|-------|
| 1 | Xóa đối tượng | SAM2 + OmniPaint |
| 2 | Thay thế đối tượng | SAM2 + FLUX.1 Fill |
| 3 | Thay nền | FLUX.2 Klein 4B |
| 4 | Thêm đối tượng (prompt) | FLUX.2 Klein 4B |
| 5 | Thêm đối tượng (ảnh tham chiếu) | OmniPaint Insertion |
| 6 | Chỉnh sửa theo prompt | FLUX.2 Klein 4B |
| 7 | Outpainting | FLUX.1 Fill |
| 8 | Text-to-Image | FLUX.2 Klein 4B |
| 9 | Upscaling | Real-ESRGAN (+GFPGAN) |

---

## 2. Yêu cầu hệ thống

| Thành phần | Yêu cầu tối thiểu |
|---|---|
| **OS** | Windows 10/11, Linux, hoặc WSL2 |
| **Python** | 3.12 (bắt buộc đúng phiên bản) |
| **Conda** | Miniconda hoặc Anaconda |
| **Node.js** | >= 18 |
| **npm** | >= 9 |
| **GPU** | NVIDIA GPU với CUDA 13.0, >= 32 GB VRAM (để chạy full models) |
| **Git** | Để clone OmniPaint |

> **Lưu ý:** Nếu không có GPU, bạn vẫn có thể chạy frontend với backend **fake** (xem [Mục 5](#5-chạy-dự-án-development)).

---

## 3. Cài đặt Backend (Python)

### Bước 3.1 — Tạo Conda environment

```bash
# Tạo môi trường mới (hoặc cập nhật nếu đã tồn tại)
conda env create -f environment.yml
# Nếu đã tồn tại, dùng lệnh sau thay thế:
conda env update -f environment.yml --prune

# Kích hoạt môi trường
conda activate imageinpaint
```

> Environment được đặt tên là `imageinpaint`, dùng Python 3.12.

### Bước 3.2 — Cài đặt SAM2

```bash
pip install git+https://github.com/facebookresearch/sam2.git
```

### Bước 3.3 — Cài đặt package `inpaint-core`

```bash
pip install -e . --no-deps
```

### Bước 3.4 — Cài đặt OmniPaint

OmniPaint runtime được clone vào `third_party/OmniPaint`:

```bash
python scripts/install_omnipaint.py
```

> **Lưu ý:** Chỉ chạy lệnh này một lần. Nếu thư mục đã tồn tại, script sẽ báo lỗi.
> **Không** chạy script `setup.sh` trong thư mục OmniPaint gốc vì nó sẽ cài đè các phiên bản library không tương thích.

### Bước 3.5 — Cài đặt PyTorch với CUDA (Windows native)

Trên **Windows native** (không phải WSL2), PyTorch CUDA cần được cài từ nguồn riêng:

```bash
# Gỡ torch đã cài từ PyPI nếu có
pip uninstall torch torchvision

# Cài từ CUDA 13.0 index
pip install torch==2.14.0 torchvision==0.29.0 --index-url https://download.pytorch.org/whl/cu130
```

Trên **Linux / WSL2**, PyPI wheels đã bao gồm CUDA và không cần bước này.

### Bước 3.6 — Cấu hình Hugging Face Token (chỉ cần cho GPU thật)

Một số model (FLUX.1 Fill, FLUX.2 Klein, OmniPaint) yêu cầu quyền truy cập Hugging Face.
Tạo token tại [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens) rồi cài đặt:

```bash
# Linux/macOS
export HF_TOKEN=hf_xxxxxxxxxxxxxxxxxxxx

# Windows PowerShell
$env:HF_TOKEN = "hf_xxxxxxxxxxxxxxxxxxxx"
```

---

## 4. Cài đặt Frontend (Next.js)

```bash
cd frontend
npm install
```

### Cấu hình biến môi trường Frontend

Copy file `.env.local.example` thành `.env.local`:

```bash
# Windows
copy frontend\.env.local.example frontend\.env.local

# Linux/macOS
cp frontend/.env.local.example frontend/.env.local
```

Nội dung mặc định của `.env.local`:

```env
# URL của backend FastAPI (mặc định: http://localhost:8000)
NEXT_PUBLIC_API_URL=http://localhost:8000

# Đặt "true" để dùng mock API (không cần backend chạy)
NEXT_PUBLIC_USE_MOCK_API=false
```

---

## 5. Chạy dự án (Development)

### Chế độ phát triển UI (không cần GPU)

Sử dụng **fake backend** — backend sẽ trả về ảnh giả, không cần GPU hay model weights.

**Terminal 1 — Chạy fake backend:**

```bash
conda activate imageinpaint
python scripts/run_api.py --fake
```

Backend sẽ khởi động tại `http://localhost:8000`.

**Terminal 2 — Chạy frontend:**

```bash
cd frontend
npm run dev
```

Frontend sẽ khởi động tại `http://localhost:3000`.

Mở trình duyệt và truy cập:
- **Landing page:** `http://localhost:3000`
- **Studio (editor):** `http://localhost:3000/studio`

### Chế độ mock hoàn toàn (chỉ frontend, không cần backend)

Sửa `frontend/.env.local`:

```env
NEXT_PUBLIC_USE_MOCK_API=true
```

Sau đó chỉ cần chạy frontend, không cần chạy backend:

```bash
cd frontend
npm run dev
```

---

## 6. Chạy dự án (Production với GPU thật)

### Bước 6.1 — Chạy backend với config thật

```bash
conda activate imageinpaint
python scripts/run_api.py --config configs/default.yaml
```

Lần đầu chạy, `ImageProcessor.from_config()` sẽ **tải xuống** các model weights từ Hugging Face về local cache.
Quá trình này có thể mất nhiều thời gian tùy tốc độ mạng.

Các tùy chọn của `run_api.py`:

| Tham số | Mặc định | Mô tả |
|---|---|---|
| `--fake` | — | Dùng fake backends, không cần GPU |
| `--config PATH` | `configs/default.yaml` | Đường dẫn tới config YAML |
| `--host` | `127.0.0.1` | Host lắng nghe |
| `--port` | `8000` | Port lắng nghe |

### Bước 6.2 — Chạy frontend (production build)

```bash
cd frontend
npm run build
npm run start
```

---

## 7. Kiểm tra môi trường

Sau khi cài đặt xong, chạy script kiểm tra để đảm bảo tất cả library đúng phiên bản:

```bash
conda activate imageinpaint

# Kiểm tra không yêu cầu CUDA (an toàn trên máy không có GPU)
python scripts/check_environment.py

# Kiểm tra đầy đủ kể cả CUDA (yêu cầu GPU)
python scripts/check_environment.py --require-cuda
```

Script kiểm tra các mục:
- Python 3.12
- Tất cả library Python với phiên bản chính xác
- NumPy / SciPy / OpenCV imports
- FLUX / Diffusers API compatibility
- OmniPaint private Diffusers API
- SAM2 API
- Real-ESRGAN / BasicSR API
- CUDA availability và bfloat16 support

---

## 8. Chạy Unit Tests

Unit tests dùng fake backends, **không cần GPU, không cần tải model**:

```bash
conda activate imageinpaint
python -m pytest -q
```

Hoặc chạy với output chi tiết hơn:

```bash
python -m pytest -v
```

Tests nằm trong thư mục `tests/`, bao gồm `tests/test_api.py` kiểm tra toàn bộ các HTTP endpoint thông qua FastAPI `TestClient`.

---

## 9. Smoke Test (GPU thật)

Sau khi cài đặt model weights, kiểm tra từng chế độ bằng GPU thật:

```bash
conda activate imageinpaint

# Kiểm tra chỉ phân vùng (segment)
python scripts/smoke_test.py --mode segment

# Kiểm tra xóa đối tượng
python scripts/smoke_test.py --mode remove

# Kiểm tra tất cả chế độ (cần ảnh tham chiếu cho add-reference)
python scripts/smoke_test.py --mode all --reference assets/cat2.jpg
```

Các `--mode` hợp lệ:

| Mode | Chức năng |
|---|---|
| `segment` | Phân vùng đối tượng |
| `remove` | Xóa đối tượng |
| `replace` | Thay thế đối tượng |
| `background` | Thay nền |
| `add-prompt` | Thêm đối tượng bằng prompt |
| `add-reference` | Thêm đối tượng bằng ảnh tham chiếu |
| `prompt-edit` | Chỉnh sửa theo prompt |
| `generate` | Text-to-Image |
| `outpaint` | Outpainting |
| `upscale` | Upscaling |
| `all` | Chạy tất cả |

Kết quả ảnh được lưu tại `outputs/smoke/`. Thống kê VRAM lưu tại `outputs/smoke/memory.json`.

---

## 10. Sử dụng Jupyter Notebook

```bash
conda activate imageinpaint
cd notebooks
jupyter lab
```

Mở file notebook trong thư mục `notebooks/`. Chạy **setup cell** trước (khởi tạo `ImageProcessor` và load ảnh mẫu `assets/cat.jpg`), sau đó chạy từng cell thao tác:

| Cell | Chức năng | Output |
|---|---|---|
| Setup | Khởi tạo processor + segment ảnh | — |
| Cell 1 | Object Removal (OmniPaint) | `outputs/notebook/01-object-removal.png` |
| Cell 2 | Object Replacement (FLUX Fill) | `outputs/notebook/02-object-replacement.png` |
| Cell 3 | Background Replacement (FLUX.2 Klein) | `outputs/notebook/03-background-replacement.png` |
| Cell 4a | Add Object by Prompt | `outputs/notebook/04a-add-object-prompt.png` |
| Cell 4b | Add Object by Reference | `outputs/notebook/04b-add-object-reference.png` |
| Cell 5 | Prompt Edit (FLUX.2 Klein) | `outputs/notebook/05-prompt-edit.png` |
| Cell 6 | Text-to-Image (FLUX.2 Klein) | `outputs/notebook/06-generate-image.png` |
| Cell 7 | Outpainting (FLUX Fill) | `outputs/notebook/07-outpaint.png` |
| Cell 8 | Upscaling (Real-ESRGAN) | `outputs/notebook/08-upscale.png` |

---

## 11. Cấu trúc thư mục

```
ImageEditor/
├── assets/                  # Ảnh demo (cat.jpg, cat2.jpg, ...)
├── configs/
│   └── default.yaml         # Cấu hình model mặc định
├── frontend/                # Next.js frontend (React 19, TypeScript)
│   ├── src/
│   │   ├── app/             # Next.js App Router pages
│   │   │   ├── page.tsx     # Landing page (/)
│   │   │   └── studio/      # Studio page (/studio)
│   │   ├── components/      # React components
│   │   ├── hooks/           # Custom React hooks
│   │   └── lib/
│   │       └── api/
│   │           ├── client.ts  # HTTP client gọi backend thật
│   │           ├── mock.ts    # Mock API (offline)
│   │           └── index.ts   # Chọn client/mock qua env var
│   ├── .env.local           # Biến môi trường (gitignored)
│   └── .env.local.example   # Template biến môi trường
├── notebooks/               # Jupyter notebooks demo
├── outputs/                 # Ảnh output (gitignored)
├── scripts/
│   ├── run_api.py           # Khởi động FastAPI server
│   ├── check_environment.py # Kiểm tra môi trường
│   ├── smoke_test.py        # GPU smoke test
│   └── install_omnipaint.py # Clone OmniPaint repo
├── src/
│   └── inpaint_core/        # Python package chính
│       ├── api/             # FastAPI app + endpoints
│       ├── backends/        # Các model backends (SAM2, FLUX, OmniPaint, ...)
│       ├── operations/      # Business logic (remove, replace, ...)
│       ├── models/          # ModelManager
│       ├── processor.py     # ImageProcessor facade
│       ├── config.py        # AppConfig dataclass
│       └── testing/         # Fake backends cho testing
├── tests/                   # Unit tests (pytest)
├── weights/                 # Checkpoint Real-ESRGAN (tải về tự động; OmniPaint LoRA nằm trong HF cache, không phải đây)
├── third_party/
│   └── OmniPaint/           # OmniPaint source (clone bởi install_omnipaint.py)
├── environment.yml          # Conda environment spec
├── pyproject.toml           # Python project config
└── GUIDE.md                 # File này
```

---

## 12. Các biến môi trường

### Frontend (`frontend/.env.local`)

| Biến | Mặc định | Mô tả |
|---|---|---|
| `NEXT_PUBLIC_API_URL` | `http://localhost:8000` | URL của FastAPI backend |
| `NEXT_PUBLIC_USE_MOCK_API` | `false` | Dùng `true` để offline mock, không cần backend |

### Backend (shell environment)

| Biến | Mô tả |
|---|---|
| `HF_TOKEN` | Hugging Face access token (cần cho các gated model) |
| `INPAINT_API_CORS_ORIGINS` | Danh sách origins cho CORS, phân cách bởi dấu phẩy (mặc định: `http://localhost:3000`) |

---

## 13. Xử lý lỗi thường gặp

### OmniPaint source not found

```
FileNotFoundError: OmniPaint source not found: .../third_party/OmniPaint
```

**Giải pháp:** Chạy lại:

```bash
python scripts/install_omnipaint.py
```

### Lỗi phiên bản library (diffusers, peft, v.v.)

**Giải pháp:** Đảm bảo đang dùng đúng conda environment và cập nhật:

```bash
conda activate imageinpaint
conda env update -f environment.yml --prune
```

### CUDA out of memory

**Giải pháp:** Trong `configs/default.yaml`:
- Đặt `memory.policy: sequential` (giải phóng model trước khi tải model khác)
- Giảm `omnipaint.max_image_side` xuống (ví dụ: `512`)
- Bật `cpu_offload: true` cho tất cả models

### Frontend không kết nối được backend

Kiểm tra:
1. Backend đang chạy tại `http://localhost:8000`
2. `frontend/.env.local` có `NEXT_PUBLIC_API_URL=http://localhost:8000`
3. Truy cập `http://localhost:8000/api/health` — phải trả về `{"status":"ok"}`

### Lỗi pip check (dependency conflicts)

**Giải pháp:** Không mix pip và conda cho các package khoa học (numpy, scipy, opencv).
Dùng Conda cho chúng như đã cấu hình trong `environment.yml`.

### CUDA không nhận dạng trên Windows

**Giải pháp:** Cài PyTorch từ CUDA index riêng:

```bash
pip install torch==2.14.0 torchvision==0.29.0 --index-url https://download.pytorch.org/whl/cu130
```

---

## Tóm tắt nhanh

```bash
# ===== CÀI ĐẶT (một lần) =====
conda env create -f environment.yml
conda activate imageinpaint
pip install git+https://github.com/facebookresearch/sam2.git
pip install -e . --no-deps
python scripts/install_omnipaint.py

cd frontend && npm install && cd ..

# ===== CHẠY DEV (fake backend, không cần GPU) =====
# Terminal 1:
conda activate imageinpaint && python scripts/run_api.py --fake
# Terminal 2:
cd frontend && npm run dev
# Mo trinh duyet: http://localhost:3000

# ===== CHẠY PRODUCTION (GPU thật) =====
# Terminal 1:
conda activate imageinpaint && python scripts/run_api.py --config configs/default.yaml
# Terminal 2:
cd frontend && npm run build && npm run start

# ===== KIỂM TRA =====
python scripts/check_environment.py --require-cuda  # kiem tra moi truong
python -m pytest -q                                  # unit tests
python scripts/smoke_test.py --mode segment          # GPU smoke test
```
