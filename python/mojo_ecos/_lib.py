"""ctypes bridge to the Mojo solver."""

from __future__ import annotations

import ctypes
import os
import shutil
import subprocess

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
LIB = os.environ.get("MOJO_ECOS_LIB") or os.path.join(
    ROOT, "dist", "libmojo-ecos.so"
)

I = ctypes.c_int64
F = ctypes.c_double

_lib: ctypes.CDLL | None = None


class BuildError(RuntimeError):
    pass


def build(force: bool = False) -> str:
    source = os.path.join(ROOT, "src", "ecos.mojo")
    if os.environ.get("MOJO_ECOS_LIB"):
        if os.path.exists(LIB):
            return LIB
        raise BuildError(f"MOJO_ECOS_LIB does not exist: {LIB}")
    if (
        not force
        and os.path.exists(LIB)
        and os.path.getmtime(LIB) >= os.path.getmtime(source)
    ):
        return LIB
    pixi = shutil.which("pixi")
    if pixi:
        command = [pixi, "run", "--manifest-path", os.path.join(ROOT, "pixi.toml"), "build"]
    else:
        command = ["bash", os.path.join(ROOT, "build", "build.sh")]
    proc = subprocess.run(command, capture_output=True, text=True, timeout=1800)
    if proc.returncode or not os.path.exists(LIB):
        raise BuildError((proc.stderr or proc.stdout).strip()[:4000])
    return LIB


def lib() -> ctypes.CDLL:
    global _lib
    if _lib is None:
        _lib = ctypes.CDLL(build())
        fn = _lib.mecos_solve
        fn.argtypes = [I] * 27 + [F, F, F]
        fn.restype = I
    return _lib


def addr(array: np.ndarray) -> int:
    return int(array.ctypes.data)
