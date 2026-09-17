"""Machine + library versions, recorded into every result file.

Without this I have no way to tell six weeks later whether two JSON files are
comparable.
"""

import json
import os
import platform
import shutil
import subprocess
import sys

import torch


def _version_of(module_name):
    try:
        mod = __import__(module_name)
        return getattr(mod, "__version__", "unknown")
    except Exception:
        return None


def _smi(query):
    exe = shutil.which("nvidia-smi")
    if exe is None:
        return None
    try:
        out = subprocess.run(
            [exe, "--query-gpu=" + query, "--format=csv,noheader"],
            capture_output=True, text=True, timeout=20, check=True,
        )
        return out.stdout.strip().splitlines()[0].strip()
    except Exception:
        return None


def l2_cache_bytes(device=0):
    # torch only exposes L2_cache_size on newer versions. Fallback is
    # deliberately too big: an oversized flush buffer wastes a bit of time,
    # an undersized one silently benchmarks L2 instead of DRAM.
    props = torch.cuda.get_device_properties(device)
    size = getattr(props, "L2_cache_size", None)
    if isinstance(size, int) and size > 0:
        return size
    return 128 << 20


def probe_ncu():
    """Can Nsight read hardware counters here? Returns (tier, detail).

    Tier A = counters work. Tier B = timing only. Worth checking before
    burning a whole session on a pod that can't profile.
    """
    exe = shutil.which("ncu")
    if exe is None:
        return "B", "ncu not on PATH"

    src = ("import torch; x = torch.randn(1024, device='cuda'); "
           "y = x.sum(); torch.cuda.synchronize()")
    cmd = [exe, "--metrics", "sm__cycles_elapsed.avg", "--csv",
           sys.executable, "-c", src]
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    except Exception as exc:
        return "B", "ncu probe raised: {}".format(exc)

    blob = (out.stdout or "") + (out.stderr or "")
    if "ERR_NVGPUCTRPERM" in blob:
        return "B", ("counters blocked (ERR_NVGPUCTRPERM). Bare linux: set "
                     "NVreg_RestrictProfilingToAdminUsers=0 and reboot. In a "
                     "container the host has to grant SYS_ADMIN.")
    if out.returncode != 0:
        return "B", "ncu exited {}: {}".format(out.returncode, blob.strip()[-400:])
    return "A", "ok"


def ncu_version():
    exe = shutil.which("ncu")
    if exe is None:
        return None
    try:
        out = subprocess.run([exe, "--version"], capture_output=True,
                             text=True, timeout=30)
        return out.stdout.strip().splitlines()[0].strip()
    except Exception:
        return None


def capture(device=0, probe_profiler=True):
    if not torch.cuda.is_available():
        raise RuntimeError("no CUDA device visible to torch")

    props = torch.cuda.get_device_properties(device)
    tier, detail = probe_ncu() if probe_profiler else ("unknown", "not probed")

    return {
        "gpu_name": props.name,
        "compute_capability": "{}.{}".format(props.major, props.minor),
        "sm_count": props.multi_processor_count,
        "total_memory_bytes": props.total_memory,
        "l2_cache_bytes": l2_cache_bytes(device),
        "driver_version": _smi("driver_version"),
        "cuda_runtime": torch.version.cuda,
        "torch_version": torch.__version__,
        "triton_version": _version_of("triton"),
        "numpy_version": _version_of("numpy"),
        "python_version": sys.version.split()[0],
        "platform": platform.system(),
        "platform_release": platform.release(),
        "hostname": platform.node(),
        # these two change what "fp32 matmul" actually means, so record them
        # rather than inheriting whatever the environment happened to set
        "allow_tf32_matmul": torch.backends.cuda.matmul.allow_tf32,
        "allow_tf32_cudnn": torch.backends.cudnn.allow_tf32,
        "profile_tier": tier,
        "profile_tier_detail": detail,
        "ncu_version": ncu_version(),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
    }


def set_numeric_flags(allow_tf32=False):
    # Off by default so fp32 means fp32 everywhere. Flip it on purpose if you
    # want to benchmark the TF32 path; the result file records which.
    torch.backends.cuda.matmul.allow_tf32 = allow_tf32
    torch.backends.cudnn.allow_tf32 = allow_tf32


if __name__ == "__main__":
    print(json.dumps(capture(), indent=2))
