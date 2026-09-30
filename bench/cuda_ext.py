"""JIT build of the .cu files.

cpp_extension instead of CMake. This is what makes the repo build on both
Windows and the pod without me maintaining two build systems: it reuses the
exact toolchain torch was compiled against and needs no build files.

Windows still needs MSVC Build Tools, and python has to start from a shell
where cl.exe is on PATH (x64 Native Tools Command Prompt, or run
vcvarsall.bat x64 first). The failure looks like a torch bug when it isn't.
"""

import os
import pathlib

from torch.utils.cpp_extension import load

ROOT = pathlib.Path(__file__).resolve().parent.parent
KERNELS = ROOT / "kernels"
BUILD = ROOT / ".build"

_LOADED = {}


def fast_math_enabled():
    return os.environ.get("KB_FAST_MATH", "0") == "1"


def cuda_flags():
    # -lineinfo is not optional. Without it ncu can't map stalls back to
    # source lines and most of the point of profiling goes away.
    flags = ["-O3", "-lineinfo"]
    if fast_math_enabled():
        # off by default: it changes rsqrtf precision, which would quietly
        # wreck the fp64 error column
        flags.append("--use_fast_math")
    return flags


def load_kernel(name, sources):
    if name in _LOADED:
        return _LOADED[name]

    BUILD.mkdir(exist_ok=True)
    paths = []
    for src in sources:
        p = KERNELS / src
        if not p.exists():
            raise FileNotFoundError("missing kernel source: {}".format(p))
        paths.append(str(p))

    _LOADED[name] = load(
        name=name,
        sources=paths,
        extra_cflags=["-O3"],
        extra_cuda_cflags=cuda_flags(),
        build_directory=str(BUILD),
        verbose=os.environ.get("KB_VERBOSE_BUILD", "0") == "1",
    )
    return _LOADED[name]
