"""Draw the README benchmark charts from results/results.csv.

Usage: python scripts/plot_benchmarks.py [results/results.csv] [--out docs/assets]

Writes light and dark SVG variants of each chart (the README switches between them with
<picture>). Colors follow the entity: our JAX backend blue, our NumPy backend orange, other
libraries gray; values are labeled directly on every bar, so identity never relies on color.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

THEMES = {
    # Categorical slots 1-2 of the validated reference palette, checked against GitHub's page
    # backgrounds (#ffffff, #0d1117); baselines in a neutral gray; text in text tokens.
    "light": {
        "jax": "#2a78d6",
        "numpy": "#eb6834",
        "other": "#8a8985",
        "text": "#1f2328",
        "muted": "#59636e",
        "grid": "#d8dee4",
    },
    "dark": {
        "jax": "#3987e5",
        "numpy": "#d95926",
        "other": "#8b8a85",
        "text": "#f0f6fc",
        "muted": "#9198a1",
        "grid": "#30363d",
    },
}
EXT = "svg"  # chart file format
SCENARIOS = {"S1": "S1 · 1-D CV (2/1)", "S2": "S2 · 2-D CV (4/2)", "S5": "S5 · INS (15/6)"}


def load(path: Path) -> dict[tuple[str, str, str, str], float]:
    values = {}
    with path.open(newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            values[(r["scenario"], r["filter"], r["metric"], r["library"])] = float(r["value"])
    return values


def fmt_time(ns: float) -> str:
    return f"{ns / 1000:.2f} µs" if ns < 10_000 else f"{ns / 1000:.1f} µs"


def style_axes(ax: plt.Axes, t: dict[str, str]) -> None:
    ax.set_facecolor("none")
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(t["grid"])
    ax.tick_params(colors=t["muted"], length=0, labelsize=9)
    ax.grid(axis="x", color=t["grid"], linewidth=0.8)
    ax.set_axisbelow(True)


def bar_panel(
    ax: plt.Axes,
    t: dict[str, str],
    title: str,
    series: list[tuple[str, str, dict[str, float]]],
    xmin: float,
    xmax: float,
) -> None:
    """Grouped horizontal bars on a log scale: one group per scenario, one bar per series."""
    height = 0.8 / len(series)
    groups = list(SCENARIOS)
    for i, (label, color, by_scenario) in enumerate(series):
        ys = [g - 0.4 + height * (i + 0.5) for g in range(len(groups))]
        xs = [by_scenario[s] for s in groups]
        # Bars start at the axis minimum, so the width is the distance to the value. The
        # height minus a sliver leaves a surface-colored gap between adjacent bars.
        widths = [x - xmin for x in xs]
        ax.barh(ys, widths, height=height * 0.82, color=color, label=label, left=xmin)
        for y, x in zip(ys, xs, strict=True):
            ax.text(x * 1.12, y, fmt_time(x), va="center", ha="left", fontsize=8.5, color=t["text"])
    ax.set_xscale("log")
    ax.set_xlim(xmin, xmax)
    ax.set_yticks(range(len(groups)), [SCENARIOS[g] for g in groups], color=t["text"])
    ax.invert_yaxis()
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: fmt_time(v)))
    ax.set_title(title, loc="left", fontsize=10.5, color=t["text"], fontweight="bold", pad=8)
    style_axes(ax, t)


def speed_chart(values: dict, theme: str, out: Path) -> None:
    t = THEMES[theme]

    def series(filt: str, library: str) -> dict[str, float]:
        return {s: values[(s, filt, "time_per_step", library)] for s in SCENARIOS}

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.5, 3.8), sharex=True)
    fig.patch.set_alpha(0)
    bar_panel(
        ax1,
        t,
        "Batch: whole sequence in one call",
        [
            ("kalman-py (JAX)", t["jax"], series("KF batch", "kalman-py")),
            ("kalman-py (NumPy)", t["numpy"], series("KF batch", "kalman-py-numpy")),
            ("pykalman", t["other"], series("KF batch", "pykalman")),
        ],
        200,
        400_000,
    )
    bar_panel(
        ax2,
        t,
        "Per step: one predict + update call",
        [
            ("kalman-py (NumPy)", t["numpy"], series("KF per-step", "kalman-py")),
            ("FilterPy", t["other"], series("KF per-step", "filterpy")),
        ],
        200,
        400_000,
    )
    ax2.tick_params(axis="y", labelleft=False)
    handles = [
        plt.Rectangle((0, 0), 1, 1, color=t["jax"]),
        plt.Rectangle((0, 0), 1, 1, color=t["numpy"]),
        plt.Rectangle((0, 0), 1, 1, color=t["other"]),
    ]
    fig.legend(
        handles,
        [
            "kalman-py · JAX backend",
            "kalman-py · NumPy backend",
            "Other library (pykalman / FilterPy)",
        ],
        loc="upper center",
        ncol=3,
        frameon=False,
        fontsize=9,
        labelcolor=t["text"],
        handlelength=1.2,
    )
    fig.text(
        0.01,
        0.01,
        "Time per filter step, log scale (lower is better). Identical estimates in every case.",
        fontsize=8.5,
        color=t["muted"],
    )
    fig.tight_layout(rect=(0, 0.05, 1, 0.92))
    fig.savefig(out / f"benchmark-speed-{theme}.{EXT}", transparent=True, dpi=110)
    plt.close(fig)


def stability_chart(values: dict, theme: str, out: Path) -> None:
    t = THEMES[theme]
    rows = [
        ("kalman-py, square_root=True", "kalman-py-sqrt", t["numpy"]),
        ("kalman-py, JAX backend", "kalman-py-jax", t["jax"]),
        ("Textbook filter (P = (I − KH)P)", "naive", t["other"]),
        ("kalman-py, NumPy default (Joseph)", "kalman-py", t["numpy"]),
    ]
    steps = [values[("S4", "KF", "steps_to_failure", lib)] for _, lib, _ in rows]
    total = 1_000_000
    fig, ax = plt.subplots(figsize=(10.5, 2.6))
    fig.patch.set_alpha(0)
    ys = range(len(rows))
    ax.barh(list(ys), steps, height=0.62, color=[c for *_, c in rows])
    for y, s in zip(ys, steps, strict=True):
        label = "1,000,000 · never failed" if s >= total else f"{s:,.0f}"
        ax.text(s + total * 0.01, y, label, va="center", ha="left", fontsize=8.5, color=t["text"])
    ax.set_yticks(list(ys), [name for name, *_ in rows], color=t["text"])
    ax.invert_yaxis()
    ax.set_xlim(0, total * 1.3)  # room for labels past the longest bars
    ax.set_xticks([0, 250_000, 500_000, 750_000, 1_000_000])
    ax.xaxis.set_major_formatter(
        FuncFormatter(lambda v, _: "1M" if v >= 1_000_000 else f"{v / 1000:,.0f}k")
    )
    ax.set_title(
        "S4 · float32 stability: steps before the covariance stops being usable",
        loc="left",
        fontsize=10.5,
        color=t["text"],
        fontweight="bold",
        pad=8,
    )
    style_axes(ax, t)
    fig.text(
        0.01,
        0.02,
        "1,000,000 ill-conditioned steps in float32 (higher is better). FilterPy: n/a, "
        "it converts float32 to float64.",
        fontsize=8.5,
        color=t["muted"],
    )
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    fig.savefig(out / f"benchmark-stability-{theme}.{EXT}", transparent=True, dpi=110)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("csv", type=Path, nargs="?", default=Path("results/results.csv"))
    parser.add_argument("--out", type=Path, default=Path("docs/assets"))
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    # Text as paths: browsers lack DejaVu Sans and would fall back to a serif font.
    plt.rcParams.update({"font.family": "DejaVu Sans", "svg.fonttype": "path"})
    values = load(args.csv)
    for theme in THEMES:
        speed_chart(values, theme, args.out)
        stability_chart(values, theme, args.out)
    print(f"wrote charts to {args.out}")


if __name__ == "__main__":
    main()
