"""Figures and README tables, always regenerated from the JSON.

Nothing here is hand-made, so re-running on a different GPU regenerates
everything without touching code.

Never merge results from different GPUs or different operating systems into
one chart. Windows uses WDDM, which batches kernel submissions and adds
launch overhead linux doesn't have, and at small sizes the ranking between
implementations can genuinely flip.

Per benchmark result file (one op, one dtype):
    *_latency.png    median latency per size
    *_roof.png       % of the measured roof per size
    *_speedup.png    speedup over PyTorch eager per size
    *_roofline.png   every measurement placed on the measured roofline
    *_table.md       the numbers behind all of the above

Across result files:
    overview_<gpu>_<dtype>.png   speedup over eager, every op x every impl

From kernel_trace.py summaries:
    kernels_<op>_<dtype>_<config>.png   kernel launches and GPU time per call
"""

import collections
import json
import math
import pathlib

import matplotlib

matplotlib.use("Agg")

import matplotlib.colors as mcolors  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.ticker as mticker  # noqa: E402

FIGURES = pathlib.Path(__file__).resolve().parent.parent / "figures"
RESULTS = pathlib.Path(__file__).resolve().parent.parent / "results"

IMPL_ORDER = ["eager", "compile", "triton", "cuda"]

# one fixed color and marker per impl, so "cuda" looks the same in every chart
IMPL_STYLE = {
    "eager":   {"label": "PyTorch eager", "color": "#2a78d6", "marker": "o"},
    "compile": {"label": "torch.compile", "color": "#eb6834", "marker": "s"},
    "triton":  {"label": "Triton",        "color": "#1baf7a", "marker": "^"},
    "cuda":    {"label": "CUDA",          "color": "#eda100", "marker": "D"},
}
OTHER_STYLE = {"color": "#898781", "marker": "x"}

# chart chrome
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"

plt.rcParams.update({
    "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE,
    "axes.edgecolor": AXIS,
    "axes.labelcolor": INK_SECONDARY,
    "axes.titlecolor": INK,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "xtick.color": INK_SECONDARY,
    "ytick.color": INK_SECONDARY,
    "grid.color": GRID,
    "grid.linewidth": 0.8,
    "legend.frameon": False,
    "font.family": "sans-serif",
    "font.size": 10,
})


def _style(impl):
    return IMPL_STYLE.get(impl, dict(OTHER_STYLE, label=impl))


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


def _title(payload, what):
    return "{}: {} ({}, {})".format(
        payload["op"], what, payload["dtype"], payload["env"]["gpu_name"])


def _uses_bandwidth_roof(payload):
    # which roof depends on the regime. Plotting TFLOP/s for a memory-bound
    # kernel gives a technically correct and totally useless number.
    regimes = {r["config_label"]: r["regime"] for r in payload["records"]}
    mem = sum(1 for r in regimes.values() if r != "compute-bound")
    return mem >= len(regimes) / 2.0


def _plain_log_ticks(axis):
    """Log-scale ticks at 1-2-5 steps, labelled 0.5, 20, 300 instead of
    5x10^-1 etc. Enough labels even when the data spans less than a decade."""
    plain = mticker.FuncFormatter(lambda value, _: "{:g}".format(value))
    axis.set_major_locator(mticker.LogLocator(subs=(1.0, 2.0, 5.0)))
    axis.set_major_formatter(plain)
    axis.set_minor_formatter(mticker.NullFormatter())


def _save(fig, out_path):
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _grouped_bars(ax, grouped, labels, value_of, skip=()):
    """One group of bars per size, one bar per impl."""
    impls = [i for i in _ordered(grouped.keys()) if i not in skip]
    width = 0.8 / max(1, len(impls))
    for k, impl in enumerate(impls):
        xs, ys = [], []
        for i, label in enumerate(labels):
            rec = grouped[impl].get(label)
            value = value_of(rec) if rec is not None else None
            if value is not None:
                xs.append(i + k * width - 0.4 + width / 2)
                ys.append(value)
        ax.bar(xs, ys, width=width * 0.92, color=_style(impl)["color"],
               label=_style(impl)["label"])
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.grid(True, axis="y")
    ax.set_axisbelow(True)


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
        style = _style(impl)
        ax.plot(xs, ys, marker=style["marker"], color=style["color"],
                linewidth=2, markersize=6, label=style["label"])

    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_yscale("log")
    _plain_log_ticks(ax.yaxis)
    ax.set_ylabel("median latency [ms], log scale (lower is better)")
    ax.set_title(_title(payload, "latency"))
    ax.grid(True, which="both")
    ax.legend()
    _save(fig, out_path)


def plot_roof(payload, out_path):
    grouped = _by_impl(payload)
    labels = _labels(payload)

    use_bw = _uses_bandwidth_roof(payload)
    key = "pct_of_bandwidth_roof" if use_bw else "pct_of_compute_roof"
    ylabel = ("% of measured bandwidth roof" if use_bw
              else "% of measured cuBLAS roof")

    fig, ax = plt.subplots(figsize=(9, 5))
    _grouped_bars(ax, grouped, labels, lambda rec: rec[key])
    ax.axhline(100.0, linewidth=1, color=INK_SECONDARY)
    ax.set_ylabel(ylabel + " (higher is better)")
    ax.set_title("{}  (roof {:.0f} GB/s, {:.1f} TFLOP/s)".format(
        _title(payload, "fraction of roof"),
        payload["roofline"]["bandwidth"]["gbs"],
        payload["roofline"]["compute"]["tflops"]))
    ax.legend()
    _save(fig, out_path)


def plot_speedup(payload, out_path, baseline="eager"):
    """Speedup over PyTorch eager at every size. Above 1 = faster than eager."""
    grouped = _by_impl(payload)
    if baseline not in grouped:
        return False
    labels = _labels(payload)

    def speedup(rec):
        base = grouped[baseline].get(rec["config_label"])
        return base["median_ms"] / rec["median_ms"] if base else None

    # the baseline itself would be a row of 1.0 bars; the line at 1 is enough
    fig, ax = plt.subplots(figsize=(9, 5))
    _grouped_bars(ax, grouped, labels, speedup, skip=(baseline,))
    ax.axhline(1.0, linewidth=1, color=INK_SECONDARY)
    ax.set_ylabel("speedup over {} (higher is better)".format(
        _style(baseline)["label"]))
    ax.set_title(_title(payload, "speedup"))
    ax.legend()
    _save(fig, out_path)
    return True


def plot_roofline(payload, out_path):
    """Every (impl, size) as a point on the measured roofline.

    x = arithmetic intensity, y = achieved TFLOP/s, both log scale. The
    slanted line is the bandwidth roof (AI x GB/s), the flat line the compute
    roof. The distance from a point up to the line above it is the headroom.
    """
    grouped = _by_impl(payload)
    records = [rec for per_cfg in grouped.values() for rec in per_cfg.values()]
    if not records or all(rec["flops"] == 0 for rec in records):
        return False  # e.g. transpose: no FLOPs, nothing to place

    gbs = payload["roofline"]["bandwidth"]["gbs"]
    tflops = payload["roofline"]["compute"]["tflops"]
    ridge = payload["roofline"]["ridge_flop_per_byte"]

    ais = [rec["arithmetic_intensity"] for rec in records]
    lo = min(min(ais), ridge) / 4.0
    hi = max(max(ais), ridge) * 4.0
    xs = [lo * (hi / lo) ** (i / 200.0) for i in range(201)]
    roof = [min(tflops, ai * gbs / 1e3) for ai in xs]  # GB/s x FLOP/B -> TFLOP/s

    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.plot(xs, roof, color=INK, linewidth=2)
    ax.axvline(ridge, color=AXIS, linewidth=1)
    ax.text(ridge, 0.02, " ridge {:.0f} FLOP/B".format(ridge),
            transform=ax.get_xaxis_transform(), color=INK_SECONDARY, fontsize=8)

    for impl in _ordered(grouped.keys()):
        recs = list(grouped[impl].values())
        style = _style(impl)
        ax.scatter([r["arithmetic_intensity"] for r in recs],
                   [r["effective_tflops"] for r in recs],
                   color=style["color"], marker=style["marker"], s=40,
                   edgecolors=SURFACE, linewidths=1.5, label=style["label"],
                   zorder=3)

    ax.set_xscale("log")
    ax.set_yscale("log")
    _plain_log_ticks(ax.xaxis)
    _plain_log_ticks(ax.yaxis)
    ax.set_xlabel("arithmetic intensity [FLOP/byte]")
    ax.set_ylabel("achieved TFLOP/s")
    ax.set_title("{}  (roof {:.0f} GB/s, {:.1f} TFLOP/s)".format(
        _title(payload, "roofline"), gbs, tflops))
    ax.grid(True, which="major")
    ax.legend(loc="upper left")  # the roof never reaches that corner
    _save(fig, out_path)
    return True


def markdown_table(payload):
    grouped = _by_impl(payload)
    labels = _labels(payload)
    impls = _ordered(grouped.keys())
    base = "eager" if "eager" in grouped else (impls[0] if impls else None)
    roof_key = ("pct_of_bandwidth_roof" if _uses_bandwidth_roof(payload)
                else "pct_of_compute_roof")

    rows = ["| size | impl | ms | GB/s | TFLOP/s | % roof | vs {} | max rel err |".format(base),
            "|---|---|---|---|---|---|---|---|"]
    for label in labels:
        base_rec = grouped.get(base, {}).get(label)
        for impl in impls:
            rec = grouped[impl].get(label)
            if rec is None:
                rows.append("| {} | {} | - | - | - | - | - | - |".format(label, impl))
                continue
            speedup = ("{:.2f}x".format(base_rec["median_ms"] / rec["median_ms"])
                       if base_rec else "-")
            err = ("-" if rec.get("max_rel_error") is None
                   else "{:.2e}".format(rec["max_rel_error"]))
            rows.append("| {} | {} | {:.4f} | {:.1f} | {:.2f} | {:.1f}% | {} | {} |".format(
                label, impl, rec["median_ms"], rec["effective_gbs"],
                rec["effective_tflops"], rec[roof_key], speedup, err))
    return "\n".join(rows)


def render(paths, fig_dir=None):
    fig_dir = pathlib.Path(fig_dir or FIGURES)
    fig_dir.mkdir(exist_ok=True)

    written = []
    for path in paths:
        payload = load(path)
        stem = pathlib.Path(path).stem

        latency = fig_dir / "{}_latency.png".format(stem)
        plot_latency(payload, latency)
        roof = fig_dir / "{}_roof.png".format(stem)
        plot_roof(payload, roof)
        written += [latency, roof]

        speedup = fig_dir / "{}_speedup.png".format(stem)
        if plot_speedup(payload, speedup):
            written.append(speedup)
        roofline = fig_dir / "{}_roofline.png".format(stem)
        if plot_roofline(payload, roofline):
            written.append(roofline)

        table = fig_dir / "{}_table.md".format(stem)
        table.write_text(markdown_table(payload))
        written.append(table)
    return written


# ---------------------------------------------------------------------------
# across ops


def _geomean(values):
    return math.exp(sum(math.log(v) for v in values) / len(values))


def _latest_per_op(paths):
    """{(gpu, platform, dtype): {op: payload}}, newest file wins."""
    groups = collections.defaultdict(dict)
    for path in sorted(paths):  # file names end in a UTC timestamp
        payload = load(path)
        env = payload["env"]
        key = (env["gpu_name"], env["platform"], payload["dtype"])
        groups[key][payload["op"]] = payload
    return groups


def plot_overview(payloads, out_path, title):
    """Heatmap: rows are ops, columns impls, cell = geomean speedup over
    eager across sizes. Blue = faster than eager, red = slower, gray = even.
    """
    ops = sorted(payloads)
    # eager is the baseline, its column would be all 1.00x
    impls = _ordered({rec["impl"] for p in payloads.values() for rec in p["records"]}
                     - {"eager"})

    speedups = []
    for op in ops:
        grouped = _by_impl(payloads[op])
        row = []
        for impl in impls:
            ratios = [
                grouped["eager"][label]["median_ms"] / rec["median_ms"]
                for label, rec in grouped.get(impl, {}).items()
                if label in grouped.get("eager", {})
            ]
            row.append(_geomean(ratios) if ratios else float("nan"))
        speedups.append(row)

    # diverging on a log scale so 2x faster and 2x slower are equally far
    # from the gray midpoint at 1x
    cmap = mcolors.LinearSegmentedColormap.from_list(
        "slower_even_faster", ["#e34948", "#f0efec", "#2a78d6"])
    norm = mcolors.LogNorm(vmin=1 / 4.0, vmax=4.0)

    fig, ax = plt.subplots(figsize=(1.9 * len(impls) + 2.5, 0.6 * len(ops) + 1.8))
    # values past 4x get the end color; missing (nan) cells stay blank
    image = ax.imshow(speedups, cmap=cmap, norm=norm, aspect="auto")
    for i, row in enumerate(speedups):
        for j, value in enumerate(row):
            text = "n/a" if math.isnan(value) else "{:.2f}x".format(value)
            ax.text(j, i, text, ha="center", va="center", color=INK, fontsize=9)

    ax.set_xticks(range(len(impls)))
    ax.set_xticklabels([_style(i)["label"] for i in impls])
    ax.set_yticks(range(len(ops)))
    ax.set_yticklabels(ops)
    ax.tick_params(length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    bar = fig.colorbar(image, ax=ax)
    bar.set_ticks([0.25, 0.5, 1, 2, 4], labels=["0.25x", "0.5x", "1x", "2x", "4x"])
    bar.ax.yaxis.set_minor_formatter(mticker.NullFormatter())
    bar.set_label("geomean speedup over PyTorch eager")
    ax.set_title(title)
    _save(fig, out_path)


def render_overview(paths, fig_dir=None):
    fig_dir = pathlib.Path(fig_dir or FIGURES)
    fig_dir.mkdir(exist_ok=True)
    written = []
    for (gpu, platform, dtype), payloads in _latest_per_op(paths).items():
        name = "overview_{}_{}_{}.png".format(
            gpu.replace(" ", "_").replace("/", "_"), platform, dtype)
        plot_overview(payloads, fig_dir / name,
                      "speedup over eager, {}, {} ({})".format(dtype, gpu, platform))
        written.append(fig_dir / name)
    return written


# ---------------------------------------------------------------------------
# kernel traces


def plot_kernel_breakdown(summaries, out_path):
    """Two panels for one op/dtype/size: kernel launches per call, and GPU
    time per call. Explains a latency gap: several launches with DRAM round
    trips between them vs one fused kernel.
    """
    order = _ordered([s["impl"] for s in summaries])
    summaries = sorted(summaries, key=lambda s: order.index(s["impl"]))
    names = [_style(s["impl"])["label"] for s in summaries]
    colors = [_style(s["impl"])["color"] for s in summaries]
    ys = list(range(len(summaries)))

    fig, (left, right) = plt.subplots(1, 2, figsize=(10, 0.55 * len(summaries) + 1.8),
                                      sharey=True)
    left.barh(ys, [s["launches_per_call"] for s in summaries], color=colors, height=0.6)
    left.set_xlabel("GPU kernel launches per call")
    right.barh(ys, [s["gpu_us_per_call"] for s in summaries], color=colors, height=0.6)
    right.set_xlabel("GPU busy time per call [us]")

    for ax, key, fmt in ((left, "launches_per_call", "{:g}"),
                         (right, "gpu_us_per_call", "{:.1f}")):
        for y, s in zip(ys, summaries):
            ax.annotate(fmt.format(s[key]), (s[key], y), xytext=(4, 0),
                        textcoords="offset points", va="center",
                        color=INK_SECONDARY, fontsize=9)
        ax.grid(True, axis="x")
        ax.set_axisbelow(True)
        ax.margins(x=0.15)

    left.set_yticks(ys)
    left.set_yticklabels(names)
    left.invert_yaxis()
    first = summaries[0]
    fig.suptitle("{}: kernels per call ({}, {}, {})".format(
        first["op"], first["dtype"], first["config_label"], first["gpu_name"]),
        color=INK)
    _save(fig, out_path)


def render_traces(trace_paths, fig_dir=None):
    fig_dir = pathlib.Path(fig_dir or FIGURES)
    fig_dir.mkdir(exist_ok=True)
    groups = collections.defaultdict(list)
    for path in trace_paths:
        s = load(path)
        groups[(s["gpu_name"], s["op"], s["dtype"], s["config_label"])].append(s)

    written = []
    for (gpu, op, dtype, label), summaries in groups.items():
        name = "kernels_{}_{}_{}_{}.png".format(
            op, dtype, label.replace("^", "").replace("=", ""),
            gpu.replace(" ", "_").replace("/", "_"))
        plot_kernel_breakdown(summaries, fig_dir / name)
        written.append(fig_dir / name)
    return written


def all_results():
    """Benchmark result files only, not ncu or trace outputs."""
    return sorted(p for p in RESULTS.glob("*.json")
                  if not p.name.startswith(("ncu_", "trace_")))
