"""Correctness first, then timing, then dump JSON.

A fast wrong kernel isn't a data point, so anything that fails tolerance gets
recorded with its error and marked failed instead of being quietly charted.
"""

import datetime
import functools
import json
import pathlib
import traceback

import torch

from . import cuda_ext, env, registry, roofline, timing

DTYPES = {
    "fp16": torch.float16,
    "bf16": torch.bfloat16,
    "fp32": torch.float32,
    "fp64": torch.float64,
}

RESULTS = pathlib.Path(__file__).resolve().parent.parent / "results"


def _check(op, impl_fn, inputs, oracle_inputs):
    try:
        with torch.no_grad():
            got = impl_fn(*inputs)
            want = op.reference(oracle_inputs)
        err = registry.max_rel_error(got, want)
        del got, want
        return err, None
    except Exception:
        return float("nan"), traceback.format_exc(limit=6)


def run_op(op_name, dtype_key, device="cuda", warmup=25, rep=100,
           check_max_elements=None, roofs=None, impl_filter=None):
    op = registry.get(op_name)
    dtype = DTYPES[dtype_key]
    if dtype not in op.supported_dtypes:
        raise ValueError("{} doesn't support {}".format(op_name, dtype_key))

    if roofs is None:
        roofs = roofline.measure(device, dtype)
    ridge = roofs["ridge_flop_per_byte"]

    gen = torch.Generator(device=device)
    gen.manual_seed(1234)

    impls = op.impls(dtype)
    if impl_filter:
        impls = {k: v for k, v in impls.items() if k in impl_filter}

    records = []

    for cfg in op.configs():
        skip = op.skip_reason(cfg, dtype, device)
        if skip:
            print("  skip {}: {}".format(op.config_label(cfg), skip))
            continue

        inputs, oracle_inputs = op.make_inputs(cfg, dtype, device, gen)

        # the fp64 oracle is 4x the memory of fp16 and runs at 1/64 rate, so
        # only check below a size cap
        n_elem = max(t.numel() for t in inputs)
        do_check = check_max_elements is None or n_elem <= check_max_elements

        nbytes = op.bytes(cfg, dtype)
        flops = op.flops(cfg)
        ai = flops / nbytes
        regime = roofline.classify(ai, ridge)
        label = op.config_label(cfg)

        print("  {} [AI={:.3f} FLOP/B, {}]".format(label, ai, regime))

        for impl_name, impl_fn in impls.items():
            rec = {
                "config": dict(cfg),
                "config_label": label,
                "impl": impl_name,
                "arithmetic_intensity": ai,
                "regime": regime,
                "analytic_bytes": nbytes,
                "flops": flops,
                "correctness_checked": do_check,
                "max_rel_error": None,
                "status": "ok",
                "error": None,
            }

            if do_check:
                err, exc = _check(op, impl_fn, inputs, oracle_inputs)
                rec["max_rel_error"] = err
                if exc is not None:
                    rec["status"] = "exception"
                    rec["error"] = exc
                    print("    {:<8} EXCEPTION".format(impl_name))
                    records.append(rec)
                    continue
                tol = op.tolerance(dtype)
                if not err <= tol:
                    rec["status"] = "incorrect"
                    rec["error"] = "rel_err {:.3e} > tol {:.3e}".format(err, tol)
                    print("    {:<8} WRONG  rel_err={:.3e}".format(impl_name, err))
                    records.append(rec)
                    continue

            try:
                t = timing.benchmark(functools.partial(impl_fn, *inputs),
                                     device, warmup=warmup, rep=rep)
            except Exception:
                rec["status"] = "exception"
                rec["error"] = traceback.format_exc(limit=6)
                print("    {:<8} EXCEPTION while timing".format(impl_name))
                records.append(rec)
                continue

            rec.update(t)
            secs = t["median_ms"] * 1e-3
            rec["effective_gbs"] = nbytes / secs / 1e9
            rec["effective_tflops"] = flops / secs / 1e12
            rec["pct_of_bandwidth_roof"] = 100.0 * rec["effective_gbs"] / roofs["bandwidth"]["gbs"]
            rec["pct_of_compute_roof"] = 100.0 * rec["effective_tflops"] / roofs["compute"]["tflops"]

            if regime == "compute-bound":
                head = "{:6.2f} TFLOP/s ({:4.1f}% of roof)".format(
                    rec["effective_tflops"], rec["pct_of_compute_roof"])
            else:
                head = "{:6.1f} GB/s ({:4.1f}% of roof)".format(
                    rec["effective_gbs"], rec["pct_of_bandwidth_roof"])
            err_s = ("" if rec["max_rel_error"] is None
                     else "  rel_err={:.2e}".format(rec["max_rel_error"]))
            print("    {:<8} {:8.4f} ms  {}{}{}".format(
                impl_name, t["median_ms"], head, err_s,
                "  NOISY" if t["noisy"] else ""))

            records.append(rec)

        del inputs, oracle_inputs
        torch.cuda.empty_cache()

    return {
        "op": op_name,
        "dtype": dtype_key,
        "timestamp_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "roofline": roofs,
        "records": records,
    }


def result_path(op_name, dtype_key, environment):
    RESULTS.mkdir(exist_ok=True)
    gpu = environment["gpu_name"].replace(" ", "_").replace("/", "_")
    stamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return RESULTS / "{}_{}_{}_{}.json".format(op_name, dtype_key, gpu, stamp)


def run_and_save(op_names, dtype_keys, device="cuda", warmup=25, rep=100,
                 check_max_elements=None, impl_filter=None, probe_profiler=True):
    environment = env.capture(probe_profiler=probe_profiler)
    environment["fast_math"] = cuda_ext.fast_math_enabled()

    print("GPU: {} (SM {}, {} SMs, L2 {:.0f} MB)".format(
        environment["gpu_name"], environment["compute_capability"],
        environment["sm_count"], environment["l2_cache_bytes"] / 1e6))
    print("profile tier: {} ({})".format(
        environment["profile_tier"], environment["profile_tier_detail"]))
    print("TF32 matmul: {}   fast_math: {}".format(
        environment["allow_tf32_matmul"], environment["fast_math"]))

    written = []
    for dtype_key in dtype_keys:
        dtype = DTYPES[dtype_key]
        print("\nmeasuring roofline for {} ...".format(dtype_key))
        roofs = roofline.measure(device, dtype)
        print("  bandwidth: {:.1f} GB/s".format(roofs["bandwidth"]["gbs"]))
        print("  compute:   {:.2f} TFLOP/s".format(roofs["compute"]["tflops"]))
        print("  ridge:     {:.1f} FLOP/byte".format(roofs["ridge_flop_per_byte"]))

        for op_name in op_names:
            op = registry.get(op_name)
            if dtype not in op.supported_dtypes:
                print("\n{} / {}: unsupported, skipping".format(op_name, dtype_key))
                continue
            print("\n{} / {}".format(op_name, dtype_key))
            payload = run_op(op_name, dtype_key, device=device, warmup=warmup,
                             rep=rep, check_max_elements=check_max_elements,
                             roofs=roofs, impl_filter=impl_filter)
            payload["env"] = environment
            path = result_path(op_name, dtype_key, environment)
            path.write_text(json.dumps(payload, indent=2))
            print("  -> {}".format(path))
            written.append(path)

    return written
