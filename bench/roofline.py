"""Measure both roofs on whatever GPU we happen to be on.

Spec sheets are useless here. I don't always get the same card, nobody hits
theoretical peak anyway, and on consumer Ada/Ampere the tensor-core number
with fp32 accumulate is lower than the marketing figure. So: big copy for the
bandwidth roof, big cuBLAS GEMM for the compute roof, both at runtime.

"% of measured roof" still isn't perfectly portable across architectures, so
cross-GPU comparisons go in a table with a caveat, never in one chart.
"""

import functools

import torch

from . import env, timing


def _copy(dst, src):
    dst.copy_(src)


def measure_bandwidth(device, dtype=torch.float16, reps=50):
    # 4x L2 so the working set can't sit in cache. A copy touches every byte
    # twice, once read once write.
    l2 = env.l2_cache_bytes(torch.device(device).index or 0)
    nbytes = max(4 * l2, 512 << 20)
    n = nbytes // torch.tensor([], dtype=dtype).element_size()

    src = torch.empty(n, dtype=dtype, device=device).uniform_(-1.0, 1.0)
    dst = torch.empty_like(src)

    # no flush: the buffer already dwarfs L2, and flushing would add another
    # full-bandwidth pass to every rep
    res = timing.benchmark(functools.partial(_copy, dst, src), device,
                           warmup=10, rep=reps, flush_l2=False)

    moved = 2 * src.numel() * src.element_size()
    gbs = moved / (res["median_ms"] * 1e-3) / 1e9

    del src, dst
    torch.cuda.empty_cache()
    return {
        "gbs": gbs,
        "method": "d2d copy (read+write)",
        "buffer_bytes": nbytes,
        "median_ms": res["median_ms"],
        "noisy": res["noisy"],
    }


def _matmul(a, b, out):
    torch.matmul(a, b, out=out)


def measure_compute(device, dtype=torch.float16, n=8192, reps=30):
    if dtype == torch.float64:
        n = min(n, 2048)  # fp64 runs at 1/64 rate on consumer cards

    a = torch.empty((n, n), dtype=dtype, device=device).normal_(0, 0.02)
    b = torch.empty((n, n), dtype=dtype, device=device).normal_(0, 0.02)
    out = torch.empty((n, n), dtype=dtype, device=device)

    res = timing.benchmark(functools.partial(_matmul, a, b, out), device,
                           warmup=10, rep=reps, flush_l2=False)
    tflops = (2.0 * n * n * n) / (res["median_ms"] * 1e-3) / 1e12

    del a, b, out
    torch.cuda.empty_cache()
    return {
        "tflops": tflops,
        "method": "cuBLAS GEMM N={}".format(n),
        "n": n,
        "median_ms": res["median_ms"],
        "noisy": res["noisy"],
    }


def measure(device, dtype):
    bw = measure_bandwidth(device, dtype=dtype)
    comp = measure_compute(device, dtype=dtype)
    return {
        "dtype": str(dtype).replace("torch.", ""),
        "bandwidth": bw,
        "compute": comp,
        # FLOP/byte where the two roofs cross. Below it, GB/s is the metric
        # that means something; above it, TFLOP/s.
        "ridge_flop_per_byte": (comp["tflops"] * 1e12) / (bw["gbs"] * 1e9),
    }


def classify(arithmetic_intensity, ridge):
    if arithmetic_intensity < 0.5 * ridge:
        return "memory-bound"
    if arithmetic_intensity > 2.0 * ridge:
        return "compute-bound"
    return "balanced"
