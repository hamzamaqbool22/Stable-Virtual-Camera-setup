"""FastAPI wrapper around generate_video() for Vast.ai + React later.

Start (on Vast, after setup_vast.sh so SEVA/CUDA are on this Python):

    python -m uvicorn main:app --host 0.0.0.0 --port 8000
"""

from __future__ import annotations

import asyncio
import tempfile
import threading
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

from generate import (
    DEFAULT_MODELS_DIR,
    IMAGE_SUFFIXES,
    OFFICIAL_TRAJS,
    PROFILES,
    PROJECT_ROOT,
    WEIGHT_NAME,
    generate_video,
    resolve_traj,
)

app = FastAPI(title="Stable Virtual Camera", version="1.1")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

_gpu_lock = threading.Lock()
UPLOAD_DIR = PROJECT_ROOT / "uploads"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)


def _cuda_status() -> dict:
    cuda = False
    gpu = None
    try:
        import torch

        cuda = bool(torch.cuda.is_available())
        if cuda:
            gpu = torch.cuda.get_device_name(0)
    except Exception:
        pass
    return {"cuda": cuda, "gpu": gpu}


@app.get("/health")
def health() -> dict:
    weights = (Path(DEFAULT_MODELS_DIR) / WEIGHT_NAME).is_file()
    seva_cloned = (PROJECT_ROOT / "stable-virtual-camera").is_dir()
    return {
        "ok": True,
        "ready": weights and seva_cloned and _cuda_status()["cuda"],
        "weights": weights,
        "seva_cloned": seva_cloned,
        **_cuda_status(),
    }


@app.get("/trajs")
def trajs() -> dict:
    return {"trajs": list(OFFICIAL_TRAJS)}


@app.get("/profiles")
def profiles() -> dict:
    return {"profiles": PROFILES}


@app.post("/generate")
async def generate(
    image: UploadFile = File(...),
    traj: str = Form("dolly zoom-in"),
    profile: str = Form("preview"),
    camera_scale: float | None = Form(None),
    compile_model: bool = Form(True),
):
    try:
        resolve_traj(traj)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    if profile not in PROFILES:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown profile {profile!r}. Use: {', '.join(PROFILES)}",
        )

    suffix = Path(image.filename or "upload.png").suffix.lower()
    if suffix not in IMAGE_SUFFIXES:
        suffix = ".png"

    status = health()
    if not status["ready"]:
        raise HTTPException(
            status_code=503,
            detail=(
                "GPU/weights not ready. On Vast run: conda activate seva && "
                "./setup_vast.sh then restart uvicorn in that env. "
                f"status={status}"
            ),
        )

    with tempfile.NamedTemporaryFile(
        dir=UPLOAD_DIR, suffix=suffix, delete=False
    ) as tmp:
        tmp.write(await image.read())
        tmp_path = Path(tmp.name)

    loop = asyncio.get_running_loop()

    def _run() -> Path:
        if not _gpu_lock.acquire(blocking=False):
            raise RuntimeError("A generation job is already running. Wait and retry.")
        try:
            return generate_video(
                image=tmp_path,
                traj=traj,
                profile=profile,
                compile_model=compile_model,
                camera_scale=camera_scale,
            )
        finally:
            _gpu_lock.release()
            tmp_path.unlink(missing_ok=True)

    try:
        out_path = await loop.run_in_executor(None, _run)
    except RuntimeError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except FileNotFoundError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return FileResponse(
        path=out_path,
        media_type="video/mp4",
        filename=out_path.name,
    )
