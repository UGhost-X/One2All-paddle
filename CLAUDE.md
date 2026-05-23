# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

One2All Paddle is an industrial visual anomaly detection system. It provides a FastAPI backend for training anomaly detection models (Dinomaly, PatchCore), deploying inference services, managing industrial cameras (Basler/Hikrobot), and serving predictions via auto-generated microservices.

## Essential Commands

### Running the App

```bash
# Start the main API server (default port 8000)
python main.py --port 8000
```

### Dependency Management

**Use `uv` only — never pip.** Do NOT run `uv sync`.

```bash
uv add <package>        # Add a dependency
uv remove <package>     # Remove a dependency
uv run <script>         # Run a script within the venv
```

### Running Tests

Tests in `test/` are standalone scripts, not a formal pytest suite:

```bash
python test/ssim_test.py
python test/stat_anomaly_test.py
python test/feature_anomaly_test.py
```

## Architecture

### Entry Point: `main.py`

A monolithic FastAPI app (~1800 lines) serving:

- **Training** (`/train/*`) — Launch/stop/resume anomaly detection training tasks. Receives base64 images with COCO annotations, extracts ROIs, and trains per-part models. Supports incremental retraining from feedback (FN/FP data).
- **Deployment** (`/deploy/*`) — Generate and launch inference microservices from Jinja2 templates. Each service wraps all models for a `task_uuid` and listens on a port in the 9000–9999 range. Managed by `ModelDeployer`.
- **Data Management** (`/project/{id}/datasets`, `/project/{id}/models`) — Browse and delete datasets/models per project.
- **Camera** (`/camera/*`) — Managed via `routers/camera.py` (mounted as sub-router).

Model environments are configured at the top of `main.py`: `TIMM_HOME`, `HF_HOME`, and `TRANSFORMERS_CACHE` all point into `models/pretrained/`. The HF endpoint uses `hf-mirror.com`.

### Core Modules

**`utils/trainer.py`** — `ModelTrainer` class. Trains Dinomaly models using anomalib's `Engine`. Supports concurrent training (max 3), checkpoint resume, and per-`pos_id` model organization. Includes adaptive brightness/contrast augmentation and `letterbox_resize` preprocessing with SSIM-based FN sample deduplication.

**`utils/patchcore_trainer.py`** — `PatchCoreModelTrainer` class. Trains a separate PatchCore model per `pos_id`. Uses coreset subsampling for memory bank construction. Shares augmentation config (`configs/augmentations.yaml`) with the main trainer.

**`utils/deployer.py`** — `ModelDeployer` class. Manages the lifecycle of inference microservices: generates Python scripts from `templates/inference_service.py.j2`, spawns them as subprocesses, tracks PIDs and ports, and exposes service health/status. State is persisted in `output/_deploy_services.json`.

**`utils/config.py`** — Path resolution layer. Reads `.env` for `OUTPUT_DIR`, `PRODUCT_DIR`, `MODELS_DIR`, `INFERENCE_SCRIPTS_DIR`. All paths can be relative (to project root) or absolute. Provides `path_config` singleton.

**`utils/roi_pre_checker.py`** — ROI occlusion detection. Compares captured ROI stats against registered template stats to detect uniform occlusion (dark/bright/empty) before running inference.

**`utils/image_decoder.py`** — Fast image decoding. Prefers TurboJPEG for JPEGs, falls back to OpenCV `imdecode`.

### Camera Subsystem (`services/`)

Driver-based architecture for multi-vendor industrial cameras:

- **`ICameraDriver`** (`camera_driver_interface.py`) — Abstract interface defining connect, capture, configure methods and `DriverCapability` metadata.
- **`BaslerDriver`**, **`HikrobotDriver`** (`drivers/`) — Concrete implementations using pypylon and MvImport SDK respectively.
- **`CameraInstance`** (`camera_instance.py`) — Vendor-agnostic wrapper around a driver, manages connection lifecycle and capture.
- **`CameraManager`** (`camera_manager.py`) — Singleton managing multiple camera instances, device discovery, and status.
- **`routers/camera.py`** — FastAPI router exposing camera CRUD, connect/disconnect, capture, and parameter adjustment endpoints at `/camera/*`.

### Inference Service Template

`templates/inference_service.py.j2` is a Jinja2 template rendered by `ModelDeployer` to generate standalone inference scripts. Each generated script loads all models for a task, exposes a local HTTP API, and handles ROI extraction, SSIM-based FN detection, and anomaly scoring.

### Key Directories

| Directory | Purpose |
|---|---|
| `output/{project_id}/{task_uuid}/` | Trained model checkpoints and artifacts |
| `product/{project_id}/{task_uuid}/` | Training data (raw images + COCO annotations) |
| `models/pretrained/` | Cached pre-trained models (timm, HF, DinoV2) |
| `inference_services/` | Generated inference service scripts |
| `configs/` | YAML configs (augmentation pipeline) |
| `dependencies/MvImport/` | Hikrobot MVS SDK Python bindings (vendored) |
| `logs/` | Application logs |

### Training Data Flow

1. Frontend posts base64 images + COCO annotations → `/train/anomaly`
2. Backend saves raw images to `product/{pid}/train/{uuid}/raw_images/` and annotations
3. During training, ROIs are extracted via COCO segmentation masks
4. Models are saved per `pos_id` under `output/{pid}/{uuid}/{pos_id}/`
5. FN samples collected during inference are stored as `.npy` files alongside models for SSIM-based deduplication and potential retraining


### Others
使用中文回复，用英文思考和分析
