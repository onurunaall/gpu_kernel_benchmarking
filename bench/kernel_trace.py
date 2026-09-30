"""Which GPU kernels does one call launch, and how long does each take?

Uses torch.profiler, which records kernel launches through CUPTI's activity
API. Unlike Nsight Compute counters this needs no elevated permission, so
it works on tier B machines too (RunPod pods, most containers).

Two outputs per trace:

* a summary JSON: kernels per call, GPU time per kernel per call. This is
  what explains "eager launched 7 kernels, the fused version launched 1".
* a Chrome trace JSON: open it at https://ui.perfetto.dev (or
  chrome://tracing) to see the CPU launch calls and GPU kernels on a
  timeline, including the idle gaps between kernels.

The durations are real kernel execution times, but the profiler adds its own
overhead to every launch, so timings still come from runner.py, not here.
"""

import collections
import json
import pathlib

import torch
from torch.profiler import ProfilerActivity, profile

from . import registry, runner

RESULTS = pathlib.Path(__file__).resolve().parent.parent / "results"


def _is_gpu_event(event):
    return event.device_type == torch.autograd.DeviceType.CUDA


def trace(op_name, dtype_key, impl_name, config_index, calls=10, device="cuda"):
    """Profile `calls` calls of one impl on one config. Returns (summary, path)."""
    op = registry.get(op_name)
    dtype = runner.DTYPES[dtype_key]
    cfg = op.configs()[config_index]
    label = op.config_label(cfg)

    gen = torch.Generator(device=device)
    gen.manual_seed(1234)
    inputs, _ = op.make_inputs(cfg, dtype, device, gen)
    impl = op.impls(dtype)[impl_name]

    with torch.no_grad():
        # warm up first so compile / autotune kernels don't show up
        for _ in range(3):
            impl(*inputs)
        torch.cuda.synchronize()

        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
            for _ in range(calls):
                impl(*inputs)
            torch.cuda.synchronize()

    # add up launches and GPU time per kernel name
    launches = collections.Counter()
    gpu_us = collections.Counter()
    for event in prof.events():
        if _is_gpu_event(event):
            launches[event.name] += 1
            gpu_us[event.name] += event.time_range.elapsed_us()

    kernels = [
        {
            "name": name,
            "launches_per_call": launches[name] / calls,
            "gpu_us_per_call": gpu_us[name] / calls,
        }
        for name in sorted(gpu_us, key=gpu_us.get, reverse=True)
    ]

    summary = {
        "op": op_name,
        "dtype": dtype_key,
        "impl": impl_name,
        "config": cfg,
        "config_label": label,
        "calls": calls,
        "gpu_name": torch.cuda.get_device_name(device),
        "launches_per_call": sum(launches.values()) / calls,
        "gpu_us_per_call": sum(gpu_us.values()) / calls,
        "kernels": kernels,
    }

    RESULTS.mkdir(exist_ok=True)
    stem = "trace_{}_{}_{}_cfg{}".format(op_name, dtype_key, impl_name, config_index)
    summary_path = RESULTS / (stem + ".json")
    chrome_path = RESULTS / (stem + ".chrome.json")
    summary["chrome_trace"] = chrome_path.name
    summary_path.write_text(json.dumps(summary, indent=2))
    prof.export_chrome_trace(str(chrome_path))
    return summary, summary_path


def print_summary(summary):
    print("{} / {} / {} / {}: {:.0f} kernel launches, {:.1f} us GPU time per call".format(
        summary["op"], summary["dtype"], summary["config_label"], summary["impl"],
        summary["launches_per_call"], summary["gpu_us_per_call"]))
    for k in summary["kernels"]:
        print("    {:8.1f} us  x{:<4g} {}".format(
            k["gpu_us_per_call"], k["launches_per_call"], k["name"][:90]))


def all_traces():
    return sorted(p for p in RESULTS.glob("trace_*.json")
                  if not p.name.endswith(".chrome.json"))
