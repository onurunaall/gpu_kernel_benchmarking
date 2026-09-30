"""Nsight Compute wrapper.

Counters and timings come from separate runs. ncu serialises launches and
replays each kernel many times to collect counters, so its durations are not
benchmark numbers and I never use them as such.

--set full is avoided, it replays every kernel dozens of times and turns a
30 second profile into ten minutes. The list below is the minimum that
actually explains a result.
"""

import argparse
import collections
import csv
import io
import json
import operator
import pathlib
import shutil
import subprocess
import sys

import torch

METRICS = [
    "gpu__time_duration.sum",
    # roofline placement
    "sm__throughput.avg.pct_of_peak_sustained_elapsed",
    "gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed",
    # real DRAM traffic, to compare against the analytic byte count
    "dram__bytes_read.sum",
    "dram__bytes_write.sum",
    # coalescing
    "l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum",
    "l1tex__t_requests_pipe_lsu_mem_global_op_ld.sum",
    "l1tex__t_sectors_pipe_lsu_mem_global_op_st.sum",
    "l1tex__t_requests_pipe_lsu_mem_global_op_st.sum",
    # occupancy and what caps it
    "sm__warps_active.avg.pct_of_peak_sustained_active",
    "launch__registers_per_thread",
    "launch__shared_mem_per_block_static",
    "launch__grid_size",
    "launch__block_size",
    # shared memory bank conflicts
    "l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum",
    "l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_st.sum",
    # why warps aren't issuing
    "smsp__average_warps_issue_stalled_long_scoreboard_per_issue_active.ratio",
    "smsp__average_warps_issue_stalled_barrier_per_issue_active.ratio",
    "smsp__average_warps_issue_stalled_mio_throttle_per_issue_active.ratio",
    "smsp__average_warps_issue_stalled_lg_throttle_per_issue_active.ratio",
    "smsp__average_warps_issue_stalled_not_selected_per_issue_active.ratio",
    "smsp__average_warps_issue_stalled_wait_per_issue_active.ratio",
]

STALL_PRE = "smsp__average_warps_issue_stalled_"
STALL_POST = "_per_issue_active.ratio"

STALL_MEANING = {
    "long_scoreboard": "waiting on global memory, needs more occupancy or "
                       "more loads in flight",
    "barrier": "__syncthreads() imbalance between warps",
    "mio_throttle": "shared memory or SFU pressure",
    "lg_throttle": "local/global instruction queue full",
    "not_selected": "plenty of eligible warps, SM is issue bound, usually "
                    "means you're near peak",
    "wait": "fixed instruction latency, try more ILP per thread",
}


def available_metrics():
    # metric spellings drift between ncu releases, so drop unknown ones with
    # a warning instead of aborting the whole profile
    exe = shutil.which("ncu")
    if exe is None:
        return []
    try:
        out = subprocess.run([exe, "--query-metrics"], capture_output=True,
                             text=True, timeout=180)
    except Exception:
        return METRICS
    if out.returncode != 0:
        return METRICS

    keep, dropped = [], []
    for m in METRICS:
        if m.split(".")[0] in out.stdout:
            keep.append(m)
        else:
            dropped.append(m)
    if dropped:
        print("warning: ncu doesn't know {}".format(", ".join(dropped)),
              file=sys.stderr)
    return keep


UNIT_SCALE = {
    "Kbyte": 1e3, "Mbyte": 1e6, "Gbyte": 1e9,
    "Kbyte/s": 1e3, "Mbyte/s": 1e6, "Gbyte/s": 1e9,
    "usecond": 1e3, "msecond": 1e6,  # normalise durations to ns
}


def parse_csv(text):
    """ncu --csv -> {kernel name: {metric: value}}"""
    lines = text.splitlines()
    start = None
    for i, line in enumerate(lines):
        if "Metric Name" in line:
            start = i
            break
    if start is None:
        return {}

    reader = csv.DictReader(io.StringIO("\n".join(lines[start:])))
    out = collections.defaultdict(dict)
    for row in reader:
        kernel = (row.get("Kernel Name") or row.get("Kernel") or "").strip()
        metric = (row.get("Metric Name") or "").strip()
        if not kernel or not metric:
            continue
        try:
            value = float((row.get("Metric Value") or "").strip().replace(",", ""))
        except ValueError:
            continue
        out[kernel][metric] = value * UNIT_SCALE.get(
            (row.get("Metric Unit") or "").strip(), 1.0)
    return dict(out)


def derive(counters, analytic_bytes=None):
    """The three numbers that actually explain a result."""
    d = {}

    ld_s = counters.get("l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum")
    ld_r = counters.get("l1tex__t_requests_pipe_lsu_mem_global_op_ld.sum")
    if ld_s and ld_r:
        # 4 sectors/request is ideal for a coalesced 32-bit access, so
        # normalise by 4: 1.0 is perfect, 2.0 means twice the traffic needed
        d["load_sectors_per_request"] = ld_s / ld_r
        d["load_coalescing_factor"] = (ld_s / ld_r) / 4.0

    st_s = counters.get("l1tex__t_sectors_pipe_lsu_mem_global_op_st.sum")
    st_r = counters.get("l1tex__t_requests_pipe_lsu_mem_global_op_st.sum")
    if st_s and st_r:
        d["store_sectors_per_request"] = st_s / st_r
        d["store_coalescing_factor"] = (st_s / st_r) / 4.0

    rd = counters.get("dram__bytes_read.sum")
    wr = counters.get("dram__bytes_write.sum")
    if rd is not None and wr is not None:
        d["measured_dram_bytes"] = rd + wr
        if analytic_bytes:
            # >1 means re-reading something, <1 means a cache is absorbing
            # traffic (normal on a 72 MB L2 part)
            d["traffic_ratio"] = (rd + wr) / analytic_bytes

    stalls = {}
    for name, value in counters.items():
        if name.startswith(STALL_PRE) and name.endswith(STALL_POST):
            stalls[name[len(STALL_PRE):-len(STALL_POST)]] = value
    if stalls:
        reason, value = max(stalls.items(), key=operator.itemgetter(1))
        d["stall_reasons"] = stalls
        d["dominant_stall"] = reason
        d["dominant_stall_value"] = value
        d["dominant_stall_meaning"] = STALL_MEANING.get(reason, "see ncu docs")

    return d


def profile(op_name, dtype_key, impl_name, config_index, out_path=None):
    exe = shutil.which("ncu")
    if exe is None:
        raise RuntimeError("ncu not on PATH, this is tier B")

    cmd = [
        exe, "--csv", "--target-processes", "all",
        "--metrics", ",".join(available_metrics()),
        sys.executable, "-m", "bench.profiler", "--child",
        "--op", op_name, "--dtype", dtype_key, "--impl", impl_name,
        "--config-index", str(config_index),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    blob = (proc.stdout or "") + "\n" + (proc.stderr or "")
    if "ERR_NVGPUCTRPERM" in blob:
        raise RuntimeError(
            "ncu can't read counters here (ERR_NVGPUCTRPERM). Bare linux: "
            "NVreg_RestrictProfilingToAdminUsers=0 plus reboot. Container: "
            "host has to pass --cap-add=SYS_ADMIN.")

    per_kernel = parse_csv(proc.stdout)
    if not per_kernel:
        raise RuntimeError("parsed nothing from ncu:\n{}".format(blob[-2000:]))

    import ops  # noqa: F401  registers everything

    from . import registry, runner
    op = registry.get(op_name)
    cfg = op.configs()[config_index]
    analytic = op.bytes(cfg, runner.DTYPES[dtype_key])

    result = {
        "op": op_name,
        "dtype": dtype_key,
        "impl": impl_name,
        "config": cfg,
        "config_label": op.config_label(cfg),
        "analytic_bytes": analytic,
        "kernel_count": len(per_kernel),
        "kernels": {},
    }
    for kernel, counters in per_kernel.items():
        result["kernels"][kernel] = {
            "counters": counters,
            "derived": derive(counters, analytic_bytes=analytic),
        }

    if out_path is None:
        out_dir = pathlib.Path(__file__).resolve().parent.parent / "results"
        out_dir.mkdir(exist_ok=True)
        out_path = out_dir / "ncu_{}_{}_{}_{}.json".format(
            op_name, dtype_key, impl_name,
            op.config_label(cfg).replace("^", "").replace("=", ""))
    pathlib.Path(out_path).write_text(json.dumps(result, indent=2))
    return result, out_path


def _child(args):
    import ops  # noqa: F401

    from . import registry, runner

    torch.cuda.init()
    op = registry.get(args.op)
    dtype = runner.DTYPES[args.dtype]
    cfg = op.configs()[args.config_index]

    gen = torch.Generator(device="cuda")
    gen.manual_seed(1234)
    inputs, _ = op.make_inputs(cfg, dtype, "cuda", gen)
    impl = op.impls(dtype)[args.impl]

    # a few warm calls so JIT/autotune happens first. ncu profiles every
    # launch including these, and averages per kernel name. Fine for
    # counters, and another reason these never become timings.
    with torch.no_grad():
        for _ in range(3):
            impl(*inputs)
        torch.cuda.synchronize()
        impl(*inputs)
        torch.cuda.synchronize()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--child", action="store_true")
    ap.add_argument("--op", required=True)
    ap.add_argument("--dtype", default="fp16")
    ap.add_argument("--impl", required=True)
    ap.add_argument("--config-index", type=int, default=0)
    args = ap.parse_args()

    if args.child:
        _child(args)
        return

    result, path = profile(args.op, args.dtype, args.impl, args.config_index)
    print(json.dumps(result, indent=2))
    print("-> {}".format(path))


if __name__ == "__main__":
    main()
