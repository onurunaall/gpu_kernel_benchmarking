# gpu-kernel-bench

PyTorch vs Triton vs hand-written CUDA, one kernel at a time, with the
methodology written down.

Every kernel exists four times, all checked against the same fp64 oracle
and timed by the same harness on the same GPU:

| impl | what it is |
|---|---|
| `eager` | plain PyTorch, what you'd write without thinking about kernels |
| `compile` | `torch.compile` of the eager code (generates Triton kernels, hands matmul to cuBLAS) |
| `triton` | a hand-written Triton kernel |
| `cuda` | a hand-written CUDA C++ kernel, JIT-built with `cpp_extension` |

| op | regime | what it exercises |
|---|---|---|
| `vector_add` | memory-bound | calibration: should sit at the copy roof |
| `reduce_sum` | memory-bound | full reduction, combining partial sums across blocks |
| `transpose` | memory-bound | access patterns only, zero FLOPs, shared-memory tiling |
| `softmax` | memory-bound | two row reductions (max, sum), numerical stability |
| `rmsnorm` | memory-bound | the fusion win over a chain of eager ops |
| `matmul` | compute-bound | tiling, register blocking, tensor cores |

The three-way comparison isn't the interesting part. Plenty of repos plot
three bars. What I care about is the layer under it: a roofline measured on
the actual device, an fp64 correctness oracle, and Nsight counters that
explain *why* each version lands where it does.

---

## Quick start

```bash
python run_all.py                     # every op, fp16: correctness, timing, figures
python run_all.py --trace             # + which kernels each impl launches
python run_all.py --ncu               # + Nsight Compute counters (tier A only)
python run_all.py --ops softmax matmul --dtypes fp16 fp32
python run_all.py --help
```

`run_all.py` is the control script: it runs the machine check, the
benchmark sweep, the optional profilers and the figures, in that order. The
`kb` CLI below does the same steps one at a time.

## Layout

```
run_all.py        control script, runs everything
bench/            the harness: timing, roofline, oracle checks, profilers, plots
ops/              one file per kernel: eager, compile and triton impls + sizes
kernels/          the CUDA source for each op
results/          JSON output (git-ignored unless force-added)
figures/          PNG + markdown tables, regenerated from results/
```

---

## Picking the metric

The metric follows from the regime, not from taste. For every kernel the
harness computes

```
AI = FLOPs / analytic DRAM bytes      [FLOP/byte]
```

and compares it against the measured ridge point `peak FLOP/s / peak GB/s`.

* `AI << ridge` → memory-bound. Headline number is **GB/s as a fraction of
  the measured bandwidth roof**. TFLOP/s here is arithmetically correct and
  tells you nothing.
* `AI >> ridge` → compute-bound. Headline number is **TFLOP/s as a fraction
  of the measured cuBLAS roof**.
* Very small sizes are neither, they're launch-bound, and raw latency is the
  only honest number.

Latency and speedup over baseline are always reported since that's what
anyone actually cares about. The roof percentage is what tells me whether
there's headroom left.

Both roofs are **measured, not read off a spec sheet**: a large d2d copy for
bandwidth, a large cuBLAS GEMM for compute. That keeps the repo hardware
agnostic and avoids a real trap, since on consumer Ada and Ampere the
tensor-core figure with fp32 accumulate is lower than the marketing number.

---

## Methodology

| Concern | What the harness does |
|---|---|
| JIT / autotune / clock ramp | 25 untimed warmup calls, thrown away |
| Timing | CUDA events, never `time.time()` |
| L2 residency | flush buffer sized at 2x the **actual** L2 between reps |
| Noise | median + q20/q80, flagged `NOISY` if q80/q20 > 1.10 |
| Correctness | fp64 oracle, not PyTorch's own low-precision output |
| TF32 | pinned explicitly, off by default, recorded in the JSON |
| Fast math | **off** by default, it changes `rsqrtf` and would wreck the error column |
| Reproducibility | GPU, driver, CUDA, torch, triton, OS recorded per run |

Two of these are worth more than a table row.

**The flush buffer is sized at runtime.** The 4090 has 72 MB of L2, bigger
than some default flush buffers. Flush too small and you benchmark L2 instead
of DRAM, and the symptom is a kernel that appears to beat 100% of DRAM
bandwidth.

**The oracle is fp64, not PyTorch.** For a reduction in fp16, PyTorch's
output is one summation order among many, not ground truth. Comparing against
it measures agreement, not correctness. The harness generates inputs in
float64, rounds them to the working dtype, then promotes the *rounded* values
back to float64 for the reference, so the oracle sees exactly the bits the
kernel sees and input rounding doesn't get mixed into kernel error. Both
PyTorch and my kernel get scored against it. Sometimes mine wins, which is a
more interesting line than any speedup.

The error is element-wise relative error, with the denominator floored at
the dtype's smallest normal number. Without the floor, fp16 outputs that land
in the subnormal range (|y| < 6e-5, a handful of elements in each RMSNorm
config) score 10%+ "relative error" for every implementation, eager included,
just from storing the result. Matmul uses max |error| / max |output| instead:
its outputs are sums of signed products, some cancel to near zero, and
element-wise relative error on those says nothing about the kernel.

---

## Profiling

Three profilers, from least to most demanding:

| tool | answers | needs | command |
|---|---|---|---|
| `torch.profiler` | which kernels does one call launch, how long does each run | nothing | `kb trace` / `run_all.py --trace` |
| Nsight Systems | timeline: launches, gaps between kernels, CPU overhead | `nsys` installed | see below |
| Nsight Compute | *why*: DRAM traffic, coalescing, occupancy, stalls | counter permission (tier A) | `kb profile` / `run_all.py --ncu` |

`kb trace` writes a summary JSON and a Chrome trace to `results/`. Open the
`.chrome.json` at https://ui.perfetto.dev to see the timeline. It's the one
to reach for on RunPod, where counters are usually blocked: it still shows
that eager RMSNorm is a chain of separate kernels and the fused versions
are one.

For an Nsight Systems timeline of one call, wrap the same child process
`kb profile` uses. It calls `cudaProfilerStart/Stop` around exactly one call
after warmup, and labels it with an NVTX range:

```bash
nsys profile --capture-range=cudaProfilerApi --trace=cuda,nvtx -o results/rmsnorm_eager \
    python -m bench.profiler --child --op rmsnorm --dtype fp16 --impl eager --config-index 3
```

### Counter tiers

Hardware counters need elevated permission (driver 418+). The harness detects
which tier it's in and records it in every result file.

* **Tier A**: `ncu` reads counters. Full analysis.
* **Tier B**: counters blocked (`ERR_NVGPUCTRPERM`). Timing works;
  occupancy, stall reasons and measured DRAM traffic don't.

Check before spending a session on it:

```bash
uv run kb env
```

**RunPod Serverless can't reach tier A.** No `SYS_ADMIN`, no shell, and
worker recycling means two sweeps might not have run on the same physical
card. Use a GPU Pod. See `RUNPOD.md`.

Counters and timings always come from separate runs. `ncu` serialises
launches and replays each kernel many times, so its durations are not
benchmark numbers. `ncu` runs with `--profile-from-start off`, so input
generation and warmup launches stay out of the report and the kernel count
is the launches of exactly one call.

### What gets collected

`--set full` is avoided, it replays every kernel dozens of times. The explicit
list in `bench/profiler.py` is the minimum that explains a result, and three
derived numbers do the actual explaining:

* **sector bytes used %**: how much of each fetched 32-byte sector the kernel
  actually used. 100% is perfectly coalesced. Raw sectors per request is
  reported too, but its ideal value depends on access width (2 for coalesced
  16-bit loads, 4 for 32-bit, 16 for 128-bit), so on its own it can't tell
  a vectorized kernel from an uncoalesced one.
* **traffic ratio**: measured `dram__bytes_*` over the analytic minimum.
  Above 1 means re-reading something. Below 1 means a cache is absorbing
  traffic, which on a 72 MB L2 part is normal and is itself a finding.
* **dominant stall**: `long_scoreboard` is waiting on global memory,
  `barrier` is `__syncthreads()` imbalance, `mio_throttle` is shared memory or
  SFU pressure, `not_selected` usually means you're near peak.

---

## Setup

Uses [uv](https://docs.astral.sh/uv/).

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh     # linux / macos
# windows: powershell -c "irm https://astral.sh/uv/install.ps1 | iex"

git clone <this repo>
cd gpu-kernel-bench
uv sync
```

On a RunPod PyTorch image, torch is already installed with a matching CUDA
toolkit and you don't want uv pulling a second 2.5 GB copy:

```bash
uv venv --system-site-packages
uv pip install matplotlib numpy
```

`triton` ships with the linux torch wheel. On Windows there's no triton, so
the triton implementations drop out automatically and `eager`, `compile` and
`cuda` still run.

Windows also needs MSVC Build Tools for the CUDA extension, and python has to
start from a shell where `cl.exe` is on PATH (x64 Native Tools Command
Prompt, or run `vcvarsall.bat x64` first). The failure message looks like a
torch bug and isn't one.

---

## Usage

```bash
uv run kb env                                    # machine + ncu probe
uv run kb list                                   # ops and config indices
uv run kb roofline --dtype fp16                  # just the roofs
uv run kb bench --op vector_add --dtype fp16     # sweep one op
uv run kb bench --op all --dtype fp16 --dtype fp32
uv run kb profile --op rmsnorm --impl cuda --config-index 2
uv run kb trace --op rmsnorm --config-index 3    # torch.profiler, every impl
uv run kb plot                                   # regenerate figures
```

Results go to `results/*.json`, figures to `figures/`. Plots always
regenerate from the JSON, so re-running on a different GPU in six months
needs no code changes. `kb plot` and `kb list` don't need a GPU, so results
copied back to a laptop can be plotted there.

### Figures

| file | shows |
|---|---|
| `*_latency.png` | median latency per size, log scale |
| `*_speedup.png` | speedup over PyTorch eager per size |
| `*_roof.png` | % of the measured bandwidth or compute roof, whichever applies |
| `*_roofline.png` | every measurement on the measured roofline (skipped for transpose, which has no FLOPs) |
| `*_table.md` | the numbers behind all of the above, including error vs the oracle |
| `overview_<gpu>_<dtype>.png` | one heatmap: geomean speedup over eager, every op x impl |
| `kernels_<op>_<dtype>_<size>.png` | from `kb trace`: kernel launches and GPU time per call, per impl |

Windows results are never merged into a linux chart. WDDM batches kernel
submissions and adds launch overhead linux doesn't have, and at small sizes
the ranking between implementations can genuinely flip. The OS is recorded
per run.

---

## Adding a kernel

1. Write `kernels/<name>.cu`.
2. Write `ops/<name>.py`: subclass `registry.Op`, decorate with
   `@registry.register`, fill in `configs`, `make_inputs`, `reference`,
   `flops`, `bytes`, `impls`. Override `error` if element-wise relative
   error is the wrong measure (see `ops/matmul.py`). Keep `configs` at 8
   sizes or fewer: `torch.compile` stops recompiling after 8 shapes of one
   function and silently runs eager after that.
3. One import line in `ops/__init__.py`.
4. `uv run kb bench --op <name> --dtype fp16`
5. `uv run kb profile --op <name> --impl cuda --config-index 2`
6. Write the analysis paragraph.

---

## Roadmap

**Phase 1, memory-bound.** Where a fused kernel genuinely beats eager torch,
because the win comes from removing DRAM round trips and launches rather than
out-computing NVIDIA.

- [x] vector add (calibrates the harness against the bandwidth roof)
- [x] sum reduction (warp shuffles, one atomic per block)
- [x] transpose (coalescing, shared memory bank conflicts)
- [x] softmax (online max/sum, numerical stability)
- [x] RMSNorm forward (the fusion win)

**Phase 2, compute-bound.** Expect to lose to cuBLAS. Tracking "% of cuBLAS"
across versions, not framing it as a contest.

- [ ] naive GEMM (the baseline to beat by 20x)
- [x] tiled GEMM with shared memory + register blocking
- [ ] register blocking + 128-bit loads
- [ ] tensor core GEMM via `wmma` / `mma` PTX

Phase 2 is where one-kernel-per-weekend stops being one kernel per weekend.

**Phase 3, later.** Fused backward passes (LayerNorm backward has a
non-obvious reduction pattern), then flash-attention-style tiled attention,
which is several weekends on its own.

---

## Analysis template

Each kernel gets a section in this shape. This is the deliverable, the bar
chart isn't.

> **RMSNorm, fp16, 8192x4096, RTX 4090.**
> AI 2.0 FLOP/byte against a ridge of ~X, so memory-bound. My kernel hit A%
> of the measured bandwidth roof against eager's B%. Eager launched N
> kernels, the fused version launched 1. Measured DRAM traffic came out at R x
> the analytic minimum; the kernel reads x twice but the second pass hit L2,
> so R landed below 1. Achieved occupancy O%, capped by registers per thread.
> Dominant stall was `long_scoreboard`, so next thing to try is more loads in
> flight per thread.

---

## Known limitations

* The roofline is first-order. A kernel can be capped by shared memory
  bandwidth, register pressure, issue rate, L2 bandwidth or atomics
  contention, none of which show up on a roofline plot. If something sits at
  40% of both roofs the roofline is telling me nothing and the stall breakdown
  is the tool to reach for.
* Measured copy bandwidth isn't a valid ceiling for every access pattern. A
  kernel that benefits from L2 residency can legitimately beat it; one with
  strided or atomic-heavy access may be structurally unable to reach it. If a
  result isn't explainable from the counters, the roof is the wrong reference
  for that kernel.
* `bytes()` is the algorithmic minimum, chosen because it's reproducible and
  dtype-aware. Measured traffic from ncu sits next to it precisely because
  they differ.
* Outputs are allocated inside each timed call, which includes caching
  allocator overhead. Matches real usage, but very small configs measure
  allocator and launch cost as much as kernel cost.
* No clock locking. Unavailable on most consumer cards and inside containers.
  The harness reports quantile spread instead of pretending clocks are pinned.
