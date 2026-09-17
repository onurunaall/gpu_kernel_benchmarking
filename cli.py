"""CLI. See README for the usual invocations."""

import argparse
import json

import torch

import ops  # noqa: F401  importing this registers every op

from . import env, plots, profiler, registry, roofline, runner


def cmd_env(args):
    payload = env.capture(probe_profiler=not args.no_probe)
    print(json.dumps(payload, indent=2))
    if payload["profile_tier"] == "A":
        print("\ntier A: counters available, full analysis works here.")
    elif payload["profile_tier"] == "B":
        print("\ntier B: timing works, counters don't.\n{}".format(
            payload["profile_tier_detail"]))


def cmd_roofline(args):
    env.set_numeric_flags(allow_tf32=args.tf32)
    for dtype_key in args.dtype:
        print(json.dumps(roofline.measure(args.device, runner.DTYPES[dtype_key]),
                         indent=2))


def cmd_bench(args):
    env.set_numeric_flags(allow_tf32=args.tf32)
    op_names = sorted(registry.all_ops()) if args.op == ["all"] else args.op
    paths = runner.run_and_save(
        op_names, args.dtype, device=args.device, warmup=args.warmup,
        rep=args.rep, check_max_elements=args.check_max_elements,
        impl_filter=args.impl or None, probe_profiler=not args.no_probe,
    )
    if not args.no_plot:
        for path in plots.render(paths):
            print("-> {}".format(path))


def cmd_profile(args):
    result, path = profiler.profile(args.op, args.dtype, args.impl,
                                    args.config_index)
    interesting = ("load_sectors_per_request", "store_sectors_per_request",
                   "measured_dram_bytes", "traffic_ratio",
                   "dominant_stall", "dominant_stall_meaning")
    for kernel, data in result["kernels"].items():
        print("\nkernel: {}".format(kernel))
        for key in interesting:
            if key in data["derived"]:
                print("  {:<28} {}".format(key, data["derived"][key]))
        occ = data["counters"].get("sm__warps_active.avg.pct_of_peak_sustained_active")
        if occ is not None:
            print("  {:<28} {:.1f}%".format("achieved occupancy", occ))
    print("\nkernels launched: {}".format(result["kernel_count"]))
    print("-> {}".format(path))


def cmd_plot(args):
    paths = args.result or plots.all_results()
    if not paths:
        print("nothing in results/")
        return
    for path in plots.render(paths):
        print("-> {}".format(path))


def cmd_list(args):
    for name, op in sorted(registry.all_ops().items()):
        dtypes = ", ".join(str(d).replace("torch.", "") for d in op.supported_dtypes)
        print("{:<14} {}".format(name, dtypes))
        for i, cfg in enumerate(op.configs()):
            print("    [{}] {}".format(i, op.config_label(cfg)))


def build_parser():
    ap = argparse.ArgumentParser(prog="kb")
    sub = ap.add_subparsers(dest="command", required=True)

    p = sub.add_parser("env", help="machine info + ncu permission probe")
    p.add_argument("--no-probe", action="store_true")
    p.set_defaults(func=cmd_env)

    p = sub.add_parser("roofline", help="measure the two roofs")
    p.add_argument("--dtype", action="append", choices=sorted(runner.DTYPES))
    p.add_argument("--device", default="cuda")
    p.add_argument("--tf32", action="store_true")
    p.set_defaults(func=cmd_roofline)

    p = sub.add_parser("bench", help="correctness + timing sweep")
    p.add_argument("--op", action="append", help="op name, or 'all'")
    p.add_argument("--dtype", action="append", choices=sorted(runner.DTYPES))
    p.add_argument("--impl", action="append", help="restrict to these impls")
    p.add_argument("--device", default="cuda")
    p.add_argument("--warmup", type=int, default=25)
    p.add_argument("--rep", type=int, default=100)
    p.add_argument("--check-max-elements", type=int, default=64 * 1024 * 1024,
                   help="skip the fp64 oracle above this many elements")
    p.add_argument("--tf32", action="store_true")
    p.add_argument("--no-plot", action="store_true")
    p.add_argument("--no-probe", action="store_true")
    p.set_defaults(func=cmd_bench)

    p = sub.add_parser("profile", help="collect ncu counters")
    p.add_argument("--op", required=True)
    p.add_argument("--dtype", default="fp16", choices=sorted(runner.DTYPES))
    p.add_argument("--impl", required=True)
    p.add_argument("--config-index", type=int, default=0)
    p.set_defaults(func=cmd_profile)

    p = sub.add_parser("plot", help="regenerate figures from results/")
    p.add_argument("--result", action="append")
    p.set_defaults(func=cmd_plot)

    p = sub.add_parser("list", help="registered ops and their configs")
    p.set_defaults(func=cmd_list)

    return ap


def main():
    args = build_parser().parse_args()
    if hasattr(args, "dtype") and not args.dtype:
        args.dtype = ["fp16"]
    if args.command == "bench" and not args.op:
        args.op = ["all"]
    if args.command != "env" and not torch.cuda.is_available():
        raise SystemExit("no CUDA device visible")
    args.func(args)


if __name__ == "__main__":
    main()
