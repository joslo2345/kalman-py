"""Compare benchmark timings between two results CSVs and flag regressions.

Usage: python scripts/compare_results.py BASELINE.csv CURRENT.csv <our-library-name>
           [--tolerance 0.10] [--fail]

Matches time_per_step rows of our own libraries (names starting with <our-library-name>) by
(library, scenario, filter, precision) and prints a Markdown table of current/baseline ratios.
A ratio above 1 + tolerance is a regression; with --fail the exit status is 1 if any exist.
Shared CI runners vary by 10-20% between runs (see the repo guide), so without --fail this only
reports, as GitHub warning annotations.
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from pathlib import Path

Key = tuple[str, str, str, str]


def timings(path: Path, ours: str) -> dict[Key, float]:
    with path.open(newline="", encoding="utf-8") as f:
        return {
            (r["library"], r["scenario"], r["filter"], r["precision"]): float(r["value"])
            for r in csv.DictReader(f)
            if r["metric"] == "time_per_step" and r["library"].startswith(ours)
        }


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")  # the output has non-ASCII (e.g. ⚠️, –) on Windows
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("baseline", type=Path)
    parser.add_argument("current", type=Path)
    parser.add_argument("ours")
    parser.add_argument("--tolerance", type=float, default=0.10)
    parser.add_argument("--fail", action="store_true")
    args = parser.parse_args()

    base, cur = timings(args.baseline, args.ours), timings(args.current, args.ours)
    shared = sorted(base.keys() & cur.keys())
    if not shared:
        print("No common time_per_step rows to compare.")
        return
    lines = [
        "| Library | Scenario | Filter | Baseline (ns) | Current (ns) | Change |",
        "|---|---|---|---|---|---|",
    ]
    regressions = []
    for key in shared:
        ratio = cur[key] / base[key]
        flag = " ⚠️" if ratio > 1 + args.tolerance else ""
        lines.append(
            f"| {key[0]} | {key[1]} | {key[2]} | {base[key]:,.1f} | {cur[key]:,.1f} | "
            f"{ratio - 1:+.1%}{flag} |"
        )
        if flag:
            regressions.append((key, ratio))
    print("\n".join(lines))
    for (library, scenario, filt, _), ratio in regressions:
        message = f"{library} {scenario} {filt}: {ratio - 1:+.1%} time per step vs baseline"
        if os.environ.get("GITHUB_ACTIONS"):
            print(f"::{'error' if args.fail else 'warning'} title=Benchmark regression::{message}")
        else:
            print(f"regression: {message}", file=sys.stderr)
    if regressions and args.fail:
        sys.exit(1)


if __name__ == "__main__":
    main()
