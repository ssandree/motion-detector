"""Ensure the locally built CUDA OpenCV is preferred over pip wheels."""

from __future__ import annotations

import os
import sys
from pathlib import Path

_DEFAULT_PREFIX = Path.home() / ".local" / "opencv-cuda"
_BOOTSTRAPPED = False


def bootstrap_opencv_cuda(prefix: Path | None = None) -> Path | None:
    """Prepend CUDA OpenCV site-packages / lib paths. Safe to call repeatedly."""
    global _BOOTSTRAPPED
    root = Path(os.environ.get("OPENCV_CUDA_PREFIX", prefix or _DEFAULT_PREFIX))
    site = root / "lib" / "python" / "site-packages"
    lib = root / "lib"
    if not site.is_dir():
        return None

    # Skip when the CUDA build does not ship a config for this interpreter
    # (e.g. OpenCV built for 3.14 but the active venv is 3.12).
    cv2_dir = site / "cv2"
    major, minor = sys.version_info[:2]
    has_versioned = (cv2_dir / f"config-{major}.{minor}.py").is_file()
    has_generic = (cv2_dir / "config-3.py").is_file()
    if not (has_versioned or has_generic):
        return None

    site_s = str(site.resolve())
    # Move to front even if already present.
    sys.path = [p for p in sys.path if p != site_s]
    sys.path.insert(0, site_s)

    lib_dirs: list[str] = []
    if lib.is_dir():
        lib_dirs.append(str(lib.resolve()))
    # Conda ffmpeg libs used by the CUDA OpenCV videoio build.
    conda_lib = Path(sys.prefix) / "lib"
    if conda_lib.is_dir():
        lib_dirs.append(str(conda_lib.resolve()))

    current = os.environ.get("LD_LIBRARY_PATH", "")
    parts = [p for p in current.split(":") if p and p not in lib_dirs]
    os.environ["LD_LIBRARY_PATH"] = ":".join([*lib_dirs, *parts])

    os.environ.setdefault("OPENCV_CUDA_PREFIX", str(root.resolve()))
    _BOOTSTRAPPED = True
    return site


def reload_cv2_if_needed() -> None:
    """If a non-CUDA cv2 was already imported, drop it so the next import is CUDA."""
    mod = sys.modules.get("cv2")
    if mod is None:
        return
    path = getattr(mod, "__file__", "") or ""
    if "opencv-cuda" in path.replace("\\", "/"):
        return
    # Remove cv2 and submodules so bootstrap path wins on re-import.
    for name in list(sys.modules):
        if name == "cv2" or name.startswith("cv2."):
            del sys.modules[name]
