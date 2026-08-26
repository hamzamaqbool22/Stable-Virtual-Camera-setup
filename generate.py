#!/usr/bin/env python3
"""Generate a camera-path video from a single image using official SEVA v1.1.

Intended to run on Vast.ai (Linux + NVIDIA). Not for macOS.

Example:
    python generate.py --image images/car.jpg --traj "dolly zoom-in" --profile preview
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
SEVA_DIR = PROJECT_ROOT / "stable-virtual-camera"
DEFAULT_MODELS_DIR = PROJECT_ROOT / "models"
DEFAULT_IMAGES_DIR = PROJECT_ROOT / "images"
DEFAULT_OUTPUTS_DIR = PROJECT_ROOT / "outputs"
WEIGHT_NAME = "modelv1.1.safetensors"

# Official strings from seva.geometry.get_preset_pose_fov
OFFICIAL_TRAJS = (
    "orbit",
    "spiral",
    "lemniscate",
    "zoom-in",
    "zoom-out",
    "dolly zoom-in",
    "dolly zoom-out",
    "move-forward",
    "move-backward",
    "move-up",
    "move-down",
    "move-left",
    "move-right",
    "roll",
)

# User-facing aliases -> official traj_prior
TRAJ_ALIASES: dict[str, str] = {
    "360": "orbit",
    "360°": "orbit",
    "orbit": "orbit",
    "spiral": "spiral",
    "lemniscate": "lemniscate",
    "infinity": "lemniscate",
    "zoom-in": "zoom-in",
    "zoom in": "zoom-in",
    "zoomin": "zoom-in",
    "zoom-out": "zoom-out",
    "zoom out": "zoom-out",
    "zoomout": "zoom-out",
    "dolly zoom-in": "dolly zoom-in",
    "dolly zoom in": "dolly zoom-in",
    "dolly-zoom-in": "dolly zoom-in",
    "dolly-zoom in": "dolly zoom-in",
    "dollyzoomin": "dolly zoom-in",
    "dolly zoom-out": "dolly zoom-out",
    "dolly zoom out": "dolly zoom-out",
    "dolly-zoom-out": "dolly zoom-out",
    "dollyzoomout": "dolly zoom-out",
    "move-forward": "move-forward",
    "move forward": "move-forward",
    "move-backward": "move-backward",
    "move backward": "move-backward",
    "move-up": "move-up",
    "move up": "move-up",
    "pan-up": "move-up",
    "pan up": "move-up",
    "move-down": "move-down",
    "move down": "move-down",
    "pan-down": "move-down",
    "pan down": "move-down",
    "move-left": "move-left",
    "move left": "move-left",
    "pan-left": "move-left",
    "pan left": "move-left",
    "move-right": "move-right",
    "move right": "move-right",
    "pan-right": "move-right",
    "pan right": "move-right",
    "roll": "roll",
}

# Frame count, fps, diffusion steps. Duration ≈ num_targets / fps.
PROFILES: dict[str, dict[str, float | int]] = {
    "preview": {"num_targets": 80, "fps": 30.0, "num_steps": 50},
    "standard": {"num_targets": 210, "fps": 10.0, "num_steps": 50},
    "long": {"num_targets": 300, "fps": 10.0, "num_steps": 50},
    "draft": {"num_targets": 80, "fps": 30.0, "num_steps": 30},
}

IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}


def resolve_traj(name: str) -> str:
    key = " ".join(name.strip().lower().replace("_", "-").split())
    if key in TRAJ_ALIASES:
        return TRAJ_ALIASES[key]
    if name in OFFICIAL_TRAJS:
        return name
    allowed = ", ".join(OFFICIAL_TRAJS)
    raise ValueError(f"Unknown trajectory {name!r}. Use one of: {allowed}")


def traj_slug(traj: str) -> str:
    return traj.replace(" ", "-")


def default_camera_scale(traj: str) -> float:
    # Official tip: pans/dollies often need 10.0; zooms/orbit stay at 2.0.
    if traj.startswith("move-") or traj.startswith("dolly"):
        return 10.0
    return 2.0


def resolve_image(image: str | None) -> Path:
    if image:
        path = Path(image)
        if not path.is_absolute():
            path = (PROJECT_ROOT / path).resolve()
        if not path.is_file():
            raise FileNotFoundError(f"Image not found: {path}")
        return path

    images_dir = DEFAULT_IMAGES_DIR
    candidates = sorted(
        p
        for p in images_dir.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES
    )
    if not candidates:
        raise FileNotFoundError(
            f"No images in {images_dir}. Pass --image or drop a photo in images/."
        )
    return candidates[0]


def _enable_torch_compile() -> None:
    """Official demo only compiles on nightly; force it on stable torch too."""
    import torch

    try:
        torch._dynamo.config.cache_size_limit = 128
        torch._dynamo.config.accumulated_cache_size_limit = 1024
        torch._dynamo.config.force_parameter_static_shapes = False
    except Exception:
        pass
    os.environ.setdefault("TORCHINDUCTOR_AUTOGRAD_CACHE", "1")
    os.environ.setdefault("TORCHINDUCTOR_FX_GRAPH_CACHE", "1")

    import seva.eval as seva_eval

    seva_eval.IS_TORCH_NIGHTLY = True


def _import_demo(compile_model: bool):
    if not SEVA_DIR.is_dir():
        raise FileNotFoundError(
            f"Official repo missing at {SEVA_DIR}. Run ./setup_vast.sh first."
        )
    if str(SEVA_DIR) not in sys.path:
        sys.path.insert(0, str(SEVA_DIR))
    if compile_model:
        _enable_torch_compile()
    import demo  # noqa: PLC0415  # imported after path + compile patch

    return demo


def generate_video(
    image: str | Path | None = None,
    traj: str = "dolly zoom-in",
    profile: str = "preview",
    *,
    models_dir: str | Path = DEFAULT_MODELS_DIR,
    output_dir: str | Path = DEFAULT_OUTPUTS_DIR,
    compile_model: bool = True,
    camera_scale: float | None = None,
    num_targets: int | None = None,
    fps: float | None = None,
    num_steps: int | None = None,
    encoding_t: int = 8,
    decoding_t: int = 8,
    seed: int = 23,
    cfg: tuple[float, float] = (4.0, 2.0),
) -> Path:
    """Run official img2trajvid_s-prob and copy the MP4 into outputs/.

    Returns the copied MP4 path. Safe to call from FastAPI later.
    """
    os.chdir(PROJECT_ROOT)

    traj_official = resolve_traj(str(traj))
    if profile not in PROFILES:
        raise ValueError(f"Unknown profile {profile!r}. Use: {', '.join(PROFILES)}")
    spec = PROFILES[profile]
    n_targets = int(num_targets if num_targets is not None else spec["num_targets"])
    video_fps = float(fps if fps is not None else spec["fps"])
    steps = int(num_steps if num_steps is not None else spec["num_steps"])
    scale = float(camera_scale if camera_scale is not None else default_camera_scale(traj_official))

    image_path = resolve_image(str(image) if image is not None else None)
    models_path = Path(models_dir).resolve()
    outputs_path = Path(output_dir).resolve()
    outputs_path.mkdir(parents=True, exist_ok=True)
    weight_path = models_path / WEIGHT_NAME
    if not weight_path.is_file():
        raise FileNotFoundError(
            f"Missing {weight_path}. Run ./setup_vast.sh or place {WEIGHT_NAME} in models/."
        )

    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available. Run this on the Vast.ai GPU instance.")

    print(f"image:         {image_path}")
    print(f"traj:          {traj_official}")
    print(f"profile:       {profile} ({n_targets} frames @ {video_fps:g} fps, ~{n_targets / video_fps:.1f}s)")
    print(f"steps:         {steps}")
    print(f"camera_scale:  {scale}")
    print(f"compile:       {compile_model}")
    print(f"weights:       {weight_path}")
    print(f"gpu:           {torch.cuda.get_device_name(0)}")

    demo = _import_demo(compile_model=compile_model)
    slug = traj_slug(traj_official)
    save_subdir = f"{slug}/{profile}"

    demo.main(
        data_path=str(image_path.parent),
        data_items=image_path.name,
        version=1.1,
        task="img2trajvid_s-prob",
        save_subdir=save_subdir,
        use_traj_prior=True,
        pretrained_model_name_or_path=str(models_path),
        weight_name=WEIGHT_NAME,
        seed=seed,
        replace_or_include_input=True,
        traj_prior=traj_official,
        cfg=list(cfg),
        guider_types=[1, 2],
        num_targets=n_targets,
        L_short=576,
        chunk_strategy="interp",
        camera_scale=scale,
        video_save_fps=video_fps,
        num_steps=steps,
        encoding_t=encoding_t,
        decoding_t=decoding_t,
    )

    scene_dir = (
        PROJECT_ROOT
        / "work_dirs"
        / "demo"
        / "img2trajvid_s-prob"
        / save_subdir
        / image_path.stem
    )
    src_mp4 = scene_dir / "samples-rgb.mp4"
    if not src_mp4.is_file():
        raise FileNotFoundError(f"SEVA finished but MP4 was not found at {src_mp4}")

    dest = outputs_path / f"{image_path.stem}_{slug}_{profile}.mp4"
    shutil.copy2(src_mp4, dest)
    print(f"saved:         {dest}")
    return dest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Stable Virtual Camera: image + camera path -> MP4 (Vast.ai GPU)."
    )
    parser.add_argument(
        "--image",
        default=None,
        help="Input image. Default: first file in images/",
    )
    parser.add_argument(
        "--traj",
        default="dolly zoom-in",
        help="Camera path. First tests: 'dolly zoom-in', 'dolly zoom-out', "
        "'zoom-in', 'zoom-out'. Then 360: orbit. "
        f"Official names: {', '.join(OFFICIAL_TRAJS)}",
    )
    parser.add_argument(
        "--profile",
        default="preview",
        choices=tuple(PROFILES),
        help="preview=80f/30fps (~2.7s), standard=210f/10fps (~21s), "
        "long=300f/10fps (~30s), draft=80f/30fps/30 steps",
    )
    parser.add_argument("--models-dir", default=str(DEFAULT_MODELS_DIR))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUTS_DIR))
    parser.add_argument(
        "--no-compile",
        action="store_true",
        help="Disable torch.compile (slower, slightly lower VRAM on first graph).",
    )
    parser.add_argument(
        "--camera-scale",
        type=float,
        default=None,
        help="Override camera motion scale. Default 10 for dolly/move, 2 otherwise.",
    )
    parser.add_argument("--num-targets", type=int, default=None)
    parser.add_argument("--fps", type=float, default=None)
    parser.add_argument("--num-steps", type=int, default=None)
    parser.add_argument("--encoding-t", type=int, default=8)
    parser.add_argument("--decoding-t", type=int, default=8)
    parser.add_argument("--seed", type=int, default=23)
    return parser


def main(argv: list[str] | None = None) -> Path:
    args = build_parser().parse_args(argv)
    return generate_video(
        image=args.image,
        traj=args.traj,
        profile=args.profile,
        models_dir=args.models_dir,
        output_dir=args.output_dir,
        compile_model=not args.no_compile,
        camera_scale=args.camera_scale,
        num_targets=args.num_targets,
        fps=args.fps,
        num_steps=args.num_steps,
        encoding_t=args.encoding_t,
        decoding_t=args.decoding_t,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
