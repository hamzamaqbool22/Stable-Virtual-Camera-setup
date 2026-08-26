# Stable Virtual Camera (SEVA) — Vast.ai pipeline

Turn a car photo into a 3D camera-path video using [Stability AI Stable Virtual Camera](https://github.com/Stability-AI/stable-virtual-camera) **v1.1**.

Write/upload this folder to a Vast.ai GPU box (**RTX 6000 WS 96GB**). Do **not** download weights or run generation on Mac.

Non-commercial research license: accept it on [stabilityai/stable-virtual-camera](https://huggingface.co/stabilityai/stable-virtual-camera) before the first download.

This is **not real-time**. Official H100 + compile is ~165s for 80 frames. A 20–30s clip uses hundreds of frames. Start with `preview`.

## Layout

| Path | Purpose |
| --- | --- |
| `images/` | Drop car photos here |
| `models/` | `modelv1.1.safetensors` (Vast only) |
| `outputs/` | Copied MP4s |
| `stable-virtual-camera/` | Official repo (cloned by setup) |
| `generate.py` | Wrapper around official `img2trajvid_s-prob` |
| `main.py` | FastAPI (`POST /generate`, `/docs`) |
| `onstart.sh` | Paste into Vast On-start Script |
| `setup_vast.sh` | Conda env + torch cu128 + clone + weights + restart API |
| `environment.yml` | `seva` env, Python 3.11 (no CUDA torch) |

## Vast.ai template (Jupyter + FastAPI)

Use image `vastai/pytorch:cuda-12.8.1-auto`, launch **Jupyter-python notebook + SSH**, Jupyter Lab, direct HTTPS, **150 GB** disk, `cpu_arch=amd64`.

**Add port `8000` TCP.** Set `JUPYTER_DIR` to `/workspace`.

Append this to `PORTAL_CONFIG` (keep the existing Jupyter/TensorBoard entries):

```
|localhost:8000:18000:/docs:SEVA API
```

**On-start Script** — paste the contents of [`onstart.sh`](onstart.sh) (clone this repo, install FastAPI only, start uvicorn on 8000). Do not run `setup_vast.sh` in on-start.

After the instance is **Running**:

1. Instance Portal (1111) → **Jupyter** / **Jupyter Terminal**
2. Instance Portal → **SEVA API** (`/docs`)
3. In Jupyter terminal, once:

```bash
cd /workspace/Stable-Virtual-Camera-setup
chmod +x setup_vast.sh
./setup_vast.sh
```

That installs SEVA in conda env `seva` and **restarts** uvicorn on that Python so `/generate` can use the GPU.

## Vast.ai setup

```bash
cd /path/to/this/project
chmod +x setup_vast.sh
./setup_vast.sh
```

The script:

1. Creates/activates conda env `seva` with Python 3.11
2. Installs `torch==2.11.0` + `torchvision==0.26.0` from **cu128** (not default PyPI CUDA 13)
3. Asserts `torch.cuda.is_available()` is True
4. Clones `https://github.com/Stability-AI/stable-virtual-camera.git` and `pip install -e .`
5. Logs into Hugging Face and downloads `modelv1.1.safetensors` into `models/`

If you already have weights in `models/`, download is skipped.

Manual equivalent:

```bash
conda create -n seva python=3.11 -y
conda activate seva
which python && python --version

python -m pip install torch==2.11.0 torchvision==0.26.0 \
  --index-url https://download.pytorch.org/whl/cu128

python - <<'PY'
import torch
assert torch.cuda.is_available(), "CUDA not available"
print(torch.__version__, torch.cuda.get_device_name(0))
PY

git clone --recursive https://github.com/Stability-AI/stable-virtual-camera.git
cd stable-virtual-camera
python -m pip install -e .
cd ..

hf auth login
hf auth whoami
# also click Accept on https://huggingface.co/stabilityai/stable-virtual-camera
hf download stabilityai/stable-virtual-camera modelv1.1.safetensors --local-dir models
```

First generation also pulls the SD 2.1 VAE and OpenCLIP ViT-H-14 (setup prefetches the VAE).

## First runs (after setup)

Put a car photo at `images/car.jpg`, then:

```bash
conda activate seva
```

**1. Dolly zoom in (short preview, ~2.7s at 30 fps, 80 frames)**

```bash
python generate.py --image images/car.jpg --traj "dolly zoom-in" --profile preview
```

**2. Zoom out**

```bash
python generate.py --image images/car.jpg --traj "zoom-out" --profile preview
```

Also useful at this stage: `"dolly zoom-out"` and `"zoom-in"`.

**3. 360 / orbit, ~21s at 10 fps (210 frames)**

```bash
python generate.py --image images/car.jpg --traj orbit --profile standard
```

**4. Optional 30s orbit**

```bash
python generate.py --image images/car.jpg --traj orbit --profile long
```

MP4s land in `outputs/`, e.g. `outputs/car_dolly-zoom-in_preview.mp4`.

### Profiles

| Profile | Frames | FPS | Length | Steps |
| --- | ---: | ---: | ---: | ---: |
| `preview` | 80 | 30 | ~2.7s | 50 |
| `draft` | 80 | 30 | ~2.7s | 30 |
| `standard` | 210 | 10 | ~21s | 50 |
| `long` | 300 | 10 | ~30s | 50 |

Do not start with hundreds of frames at 30 fps.

## Camera paths

`--traj` accepts official names or aliases (`360` → `orbit`, `pan-left` → `move-left`).

| First | Then 360 | Later |
| --- | --- | --- |
| `dolly zoom-in` | `orbit` | `spiral`, `lemniscate` |
| `dolly zoom-out` | | `move-forward`, `move-backward` |
| `zoom-in` | | `move-up`, `move-down`, `move-left`, `move-right` |
| `zoom-out` | | `roll` |

`zoom-*` changes focal length in place. `dolly zoom-*` moves the camera and the lens (Vertigo). 360 is `orbit`.

Dolly/move paths default `--camera-scale 10`. Override with `--camera-scale 2` if motion is too large.

## Speed knobs (96GB)

`generate.py` already:

- Forces `torch.compile` (official demo only auto-enables it on nightly)
- Sets `--L_short 576`
- Raises VAE `encoding_t` / `decoding_t` to 8

If VRAM is tight: `--no-compile --encoding-t 1 --decoding-t 1`.

## FastAPI

`GET /health` works as soon as uvicorn starts. `POST /generate` needs CUDA + `models/modelv1.1.safetensors` (`ready: true` in `/health`).

```bash
# after setup_vast.sh
curl -s http://127.0.0.1:8000/health

curl -X POST http://127.0.0.1:8000/generate \
  -F "image=@images/front.png" \
  -F "traj=dolly zoom-in" \
  -F "profile=preview" \
  --output out.mp4
```

React later: `POST /generate` with `multipart/form-data` (`image`, `traj`, `profile`). OpenAPI: `/docs`.

Local `python -m uvicorn main:app` on Mac will serve `/health` but `/generate` will 503 without CUDA.
