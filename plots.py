"""Figures and README tables, always regenerated from the JSON.

Nothing here is hand-made, so re-running on a different GPU regenerates
everything without touching code.

Never merge results from different GPUs or different operating systems into
one chart. Windows uses WDDM, which batches kernel submissions and adds
launch overhead linux doesn't have, and at small sizes the ranking between
implementations can genuinely flip.
"""

import collections
import json
import pathlib

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402

FIGURES = pathlib.Path(__file__).resolve().parent.parent / "figures"
RESULTS = pathlib.Path(__file__).resolve().parent.parent / "results"

IMPL_ORDER = ["eager", "compile", "triton", "cuda"]


def _ordered(names):
    known = [n for n in IMPL_ORDER if n in names]
    return known + sorted(n for n in names if n not in IMPL_ORDER)


def load(path):
    return json.loads(pathlib.Path(path).read_text())


def _by_impl(payload):
    out = collections.defaultdict(dict)
    for rec in payload["records"]:
        if rec.get("status") == "ok" and "median_ms" in rec:
            out[rec["impl"]][rec["config_label"]] = rec
    return out


def _labels(payload):
    seen = []
    for rec in payload["records"]:
        if rec["config_label"] not in seen:
            seen.append(rec["config_label"])
    return seen


def plot_latency(payload, out_path):
    grouped = _by_impl(payload)
    labels = _labels(payload)

    fig, ax = plt.subplots(figsize=(9, 5))
    for impl in _ordered(grouped.keys()):
        xs, ys = [], []
        for i, label in enumerate(labels):
            rec = grouped[impl].get(label)
            if rec is not None:
                xs.append(i)
                ys.append(rec["median_ms"])
        ax.plot(xs, ys, marker="o", label=impl)

    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_yscale("log")
    ax.set_ylabel("median latency [ms], log scale")
    ax.set_title("{} / {} on {}".format(
        payload["op"], payload["dtype"], payload["env"]["gpu_name"]))
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_roof(payload, out_path):
    # which roof depends on the regime. Plotting TFLOP/s for a memory-bound
    # kernel gives a technically correct and totally useless number.
    grouped = _by_impl(payload)
    labels = _labels(payload)
    impls = _ordered(grouped.keys())

    regimes = {r["config_label"]: r["regime"] for r in payload["records"]}
    mem = sum(1 for r in regimes.values() if r != "compute-bound")
    use_bw = mem >= len(regimes) / 2.0
    key = "pct_of_bandwidth_roof" if use_bw else "pct_of_compute_roof"
    ylabel = ("% of measured bandwidth roof" if use_bw
              else "% of measured cuBLAS roof")

    width = 0.8 / max(1, len(impls))
    fig, ax = plt.subplots(figsize=(9, 5))
    for k, impl in enumerate(impls):
        xs, ys = [], []
        for i, label in enumerate(labels):
            rec = grouped[impl].get(label)
            if rec is not None:
                xs.append(i + k * width - 0.4 + width / 2)
                ys.append(rec[key])
        ax.bar(xs, ys, width=width, label=impl)

    ax.axhline(100.0, linestyle="--", linewidth=1, color="0.4")
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_ylabel(ylabel)
    ax.set_title("{} / {} on {}  (roof {:.0f} GB/s, {:.1f} TFLOP/s)".format(
        payload["op"], payload["dtype"], payload["env"]["gpu_name"],
        payload["roofline"]["bandwidth"]["gbs"],
        payload["roofline"]["compute"]["tflops"]))
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def markdown_table(payload):
    grouped = _by_impl(payload)
    labels = _labels(payload)
    impls = _ordered(grouped.keys())
    base = "eager" if "eager" in grouped else (impls[0] if impls else None)

    rows = ["| size | impl | ms | GB/s | % roof | vs {} | max rel err |".format(base),
            "|---|---|---|---|---|---|---|"]
    for label in labels:
        base_rec = grouped.get(base, {}).get(label)
        for impl in impls:
            rec = grouped[impl].get(label)
            if rec is None:
                rows.append("| {} | {} | - | - | - | - | - |".format(label, impl))
                continue
            speedup = ("{:.2f}x".format(base_rec["median_ms"] / rec["median_ms"])
                       if base_rec else "-")
            err = ("-" if rec.get("max_rel_error") is None
                   else "{:.2e}".format(rec["max_rel_error"]))
            rows.append("| {} | {} | {:.4f} | {:.1f} | {:.1f}% | {} | {} |".format(
                label, impl, rec["median_ms"], rec["effective_gbs"],
                rec["pct_of_bandwidth_roof"], speedup, err))
    return "\n".join(rows)


def render(paths, fig_dir=None):
    fig_dir = pathlib.Path(fig_dir or FIGURES)
    fig_dir.mkdir(exist_ok=True)

    written = []
    for path in paths:
        payload = load(path)
        stem = pathlib.Path(path).stem
        latency = fig_dir / "{}_latency.png".format(stem)
        roof = fig_dir / "{}_roof.png".format(stem)
        table = fig_dir / "{}_table.md".format(stem)
        plot_latency(payload, latency)
        plot_roof(payload, roof)
        table.write_text(markdown_table(payload))
        written += [latency, roof, table]
    return written


def all_results():
    return sorted(p for p in RESULTS.glob("*.json")
                  if not p.name.startswith("ncu_"))
