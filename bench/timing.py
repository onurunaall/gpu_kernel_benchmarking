"""CUDA-event timing with an L2 flush between reps.

Not using triton.testing.do_bench for two reasons. Its signature has moved
around between triton releases, which would make my own results from
different weekends incomparable. And the flush buffer needs to be sized from
the actual device: the 4090 has 72 MB of L2, which is bigger than some
default flush buffers, and an undersized flush turns a DRAM measurement into
an L2 measurement.

Everything here takes a zero-arg callable. Build them with functools.partial.
"""

import statistics

import torch

from . import env

_FLUSH = {}


def flush_buffer(device):
    key = torch.device(device).index or 0
    if key not in _FLUSH:
        nbytes = max(2 * env.l2_cache_bytes(key), 256 << 20)
        _FLUSH[key] = torch.empty(nbytes, dtype=torch.int8, device=device)
    return _FLUSH[key]


def benchmark(fn, device, warmup=25, rep=100, flush_l2=True):
    """Median latency in ms, plus q20/q80 and a noise flag.

    Warmup matters more than people think: triton JIT-compiles and autotunes
    on first call, cuBLAS runs shape heuristics, and clocks ramp. The first
    calls measure none of what you want.

    Median not mean. Scheduler noise is one-sided, so the mean drifts upward
    for reasons that say nothing about the kernel.
    """
    torch.cuda.set_device(device)
    flush = flush_buffer(device) if flush_l2 else None

    for _ in range(warmup):
        fn()
    torch.cuda.synchronize(device)

    starts = [torch.cuda.Event(enable_timing=True) for _ in range(rep)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(rep)]

    for i in range(rep):
        if flush is not None:
            flush.zero_()
        starts[i].record()
        fn()
        ends[i].record()

    torch.cuda.synchronize(device)

    times = sorted(s.elapsed_time(e) for s, e in zip(starts, ends))

    def at(q):
        idx = int(round(q * (len(times) - 1)))
        return times[max(0, min(len(times) - 1, idx))]

    q20, q80 = at(0.20), at(0.80)

    # Can't lock clocks on most consumer cards or inside a container, so
    # instead of pretending they're pinned I just measure the spread.
    noisy = (q80 / q20) > 1.10 if q20 > 0 else True

    return {
        "median_ms": statistics.median(times),
        "q20_ms": q20,
        "q80_ms": q80,
        "min_ms": times[0],
        "reps": rep,
        "warmup": warmup,
        "l2_flushed": flush_l2,
        "noisy": noisy,
    }
