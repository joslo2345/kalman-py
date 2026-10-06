"""Turn results/results.csv into the README comparison table.

Usage: python scripts/make_table.py results/results.csv <our-library-name> [--readme README.md]

For each row, "Ours vs best other" compares our library with the best library that isn't one of
our own variants (names starting with "<ours>-", e.g. another backend); above 1.00x means ours is
better. Footnotes come from notes.md next to the CSV. With --readme, the table replaces the text
between <!-- BENCH:START --> and <!-- BENCH:END --> in that file instead of being printed.
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
from collections import defaultdict
from pathlib import Path

LOWER_IS_BETTER = {
    "time_per_step",
    "cycles_per_step",
    "rmse",
    "heap_allocations",
    "peak_memory",
    "flash_bytes",
    "ram_bytes",
}
HIGHER_IS_BETTER = {"steps_to_failure", "steps_to_indefinite"}
START, END = "<!-- BENCH:START -->", "<!-- BENCH:END -->"


def _times(r: float) -> str:
    return f"{r:.2f}x" if r >= 0.01 else f"{r:.2g}x"


def ratio(metric: str, vals: dict[str, float], ours: str) -> str:
    # Libraries named "<ours>-something" are our own variants (e.g. another backend).
    others = [v for lib, v in vals.items() if not lib.startswith(ours)]
    if ours not in vals or not others:
        return "n/a"
    if metric in LOWER_IS_BETTER:
        best = min(others)
        return "–" if vals[ours] == 0 else _times(best / vals[ours])
    if metric in HIGHER_IS_BETTER:
        best = max(others)
        return "–" if best == 0 else _times(vals[ours] / best)
    return "–"  # nees, max_abs_diff: read the values directly


def fmt(value: float) -> str:
    return f"{value:,.0f}" if abs(value) >= 1000 else f"{value:.4g}"


def make_table(rows: list[dict[str, str]], ours: str, notes: list[str]) -> str:
    libs = [ours] + sorted({r["library"] for r in rows} - {ours})
    cells: dict[tuple[str, ...], dict[str, float]] = defaultdict(dict)
    for r in rows:
        key = (r["scenario"], r["filter"], r["precision"], r["metric"], r["unit"])
        cells[key][r["library"]] = float(r["value"])

    environments = sorted(
        {(r["cpu"], r["os"], r["toolchain"], r["commit"], r["date"]) for r in rows}
    )
    lines = [
        "Measured on "
        + "; ".join(
            f"{cpu}, {os_}, {tool}, commit {commit}, {date}"
            for cpu, os_, tool, commit, date in environments
        )
        + ".",
        "",
        "| Scenario | Filter | Precision | Metric | "
        + " | ".join(libs)
        + " | Ours vs best other |",
        "|" + "---|" * (len(libs) + 5),
    ]
    for (scen, filt, prec, metric, unit), vals in sorted(cells.items()):
        values = [fmt(vals[lib]) if lib in vals else "n/a" for lib in libs]
        lines.append(
            f"| {scen} | {filt} | {prec} | {metric} ({unit}) | "
            + " | ".join(values)
            + f" | {ratio(metric, vals, ours)} |"
        )
    if notes:
        lines += ["", *notes]
    return "\n".join(lines)


def update_readme(readme: Path, table: str) -> None:
    text = readme.read_text(encoding="utf-8")
    pattern = re.compile(re.escape(START) + r".*?" + re.escape(END), re.DOTALL)
    if not pattern.search(text):
        raise SystemExit(f"{readme} has no {START} ... {END} block")
    readme.write_text(
        pattern.sub(lambda _: f"{START}\n{table}\n{END}", text, count=1), encoding="utf-8"
    )


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")  # the output has non-ASCII (e.g. ⚠️, –) on Windows
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("csv", type=Path)
    parser.add_argument("ours")
    parser.add_argument("--readme", type=Path)
    args = parser.parse_args()

    with args.csv.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    notes_file = args.csv.with_name("notes.md")
    notes = notes_file.read_text(encoding="utf-8").splitlines() if notes_file.exists() else []
    table = make_table(rows, args.ours, notes)
    if args.readme:
        update_readme(args.readme, table)
    else:
        print(table)


if __name__ == "__main__":
    main()
