"""Run the whole benchmark suite from one script.

    python run_all.py                           # every op, fp16, timing + figures
    python run_all.py --ops rmsnorm softmax     # only some ops
    python run_all.py --dtypes fp16 fp32        # more than one dtype
    python run_all.py --trace                   # + torch.profiler kernel traces
    python run_all.py --ncu                     # + Nsight Compute counters (tier A only)
    python run_all.py --quick                   # fewer timing reps, for a smoke test

What it does, in order:

    1. machine check: GPU, library versions, can Nsight read counters here?
    2. benchmark: correctness against the fp64 oracle, then timing, for every
       op x dtype x impl x size. One JSON per op/dtype in results/.
    3. (--trace) torch.profiler on one size per op: which kernels each impl
       launches and how long each runs. Works without special permissions.
    4. (--ncu) Nsight Compute counters on one size per op: DRAM traffic,
       coalescing, occupancy, stall reasons. Needs profile tier A.
    5. figures and markdown tables in figures/.

Run it from the repo root, or with `uv run python run_all.py`.
"""

import argparse
import sys

import torch

import ops  # noqa: F401  importing this registers every op
from bench import env, kernel_trace, plots, profiler, registry, runner


def parse_args():
    all_ops = sorted(registry.all_ops())
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ops", nargs="+", default=all_ops, choices=all_ops,
                    help="default: all of them")
    ap.add_argument("--dtypes", nargs="+", default=["fp16"],
                    choices=["fp16", "bf16", "fp32"])
    ap.add_argument("--impls", nargs="+", default=None,
                    help="e.g. eager triton cuda. Default: every impl")
    ap.add_argument("--trace", action="store_true",
                    help="also record torch.profiler kernel traces")
    ap.add_argument("--ncu", action="store_true",
                    help="also collect Nsight Compute counters")
    ap.add_argument("--profile-config", type=int, default=None,
                    help="size index to trace/profile (see `kb list`). "
                         "Default: the middle size of each op")
    ap.add_argument("--quick", action="store_true",
                    help="5 warmup + 20 timed reps instead of 25 + 100")
    ap.add_argument("--tf32", action="store_true",
                    help="allow TF32 for fp32 matmul (off by default)")
    return ap.parse_args()


def profile_config_index(op_name, requested):
    if requested is not None:
        return requested
    return len(registry.get(op_name).configs()) // 2


def impls_to_profile(op_name, dtype_key, requested):
    available = sorted(registry.get(op_name).impls(runner.DTYPES[dtype_key]))
    if requested is None:
        return available
    return [impl for impl in available if impl in requested]


def step(title):
    print("\n" + "=" * 72)
    print(title)
    print("=" * 72)


def main():
    args = parse_args()
    if not torch.cuda.is_available():
        sys.exit("no CUDA device visible to torch")

    env.set_numeric_flags(allow_tf32=args.tf32)
    warmup, rep = (5, 20) if args.quick else (25, 100)

    step("1. machine check")
    machine = env.capture(probe_profiler=args.ncu)
    print("GPU:        {}".format(machine["gpu_name"]))
    print("torch:      {}   triton: {}   CUDA: {}".format(
        machine["torch_version"], machine["triton_version"], machine["cuda_runtime"]))
    if args.ncu:
        print("ncu tier:   {} ({})".format(machine["profile_tier"],
                                           machine["profile_tier_detail"]))

    step("2. benchmark: correctness + timing")
    result_paths = runner.run_and_save(
        args.ops, args.dtypes, warmup=warmup, rep=rep,
        check_max_elements=64 * 1024 * 1024, impl_filter=args.impls,
        probe_profiler=False)

    trace_paths = []
    if args.trace:
        step("3. kernel traces (torch.profiler)")
        for op_name in args.ops:
            for dtype_key in args.dtypes:
                cfg = profile_config_index(op_name, args.profile_config)
                for impl in impls_to_profile(op_name, dtype_key, args.impls):
                    summary, path = kernel_trace.trace(op_name, dtype_key, impl, cfg)
                    kernel_trace.print_summary(summary)
                    trace_paths.append(path)

    if args.ncu:
        step("4. hardware counters (Nsight Compute)")
        if machine["profile_tier"] != "A":
            print("skipped: counters not readable here. {}".format(
                machine["profile_tier_detail"]))
        else:
            for op_name in args.ops:
                for dtype_key in args.dtypes:
                    cfg = profile_config_index(op_name, args.profile_config)
                    for impl in impls_to_profile(op_name, dtype_key, args.impls):
                        print("{} / {} / {} ...".format(op_name, dtype_key, impl))
                        try:
                            _, path = profiler.profile(op_name, dtype_key, impl, cfg)
                            print("  -> {}".format(path))
                        except RuntimeError as exc:
                            print("  failed: {}".format(str(exc)[:300]))

    step("5. figures")
    figures = plots.render(result_paths)
    figures += plots.render_overview(plots.all_results())
    figures += plots.render_traces(trace_paths)
    for path in figures:
        print("-> {}".format(path))


if __name__ == "__main__":
    main()
