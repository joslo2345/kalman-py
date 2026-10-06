"""Reading and writing results/results.csv and its footnotes."""

from __future__ import annotations

import csv
from collections.abc import Iterable, Sequence
from pathlib import Path

HEADER = [
    "library",
    "library_version",
    "scenario",
    "filter",
    "precision",
    "metric",
    "value",
    "unit",
    "commit",
    "cpu",
    "os",
    "toolchain",
    "date",
]


def append_rows(path: Path, rows: Iterable[Sequence[str]]) -> None:
    """Append rows, writing the header first if the file is new."""
    path.parent.mkdir(parents=True, exist_ok=True)
    new = not path.exists() or path.stat().st_size == 0
    with path.open("a", newline="") as f:
        out = csv.writer(f)
        if new:
            out.writerow(HEADER)
        for row in rows:
            if len(row) != len(HEADER):
                raise ValueError(f"row has {len(row)} fields, expected {len(HEADER)}: {row}")
            out.writerow(row)


def add_note(results_csv: Path, note: str) -> None:
    """Record a footnote for the table (e.g. why a cell is n/a) next to the CSV, once."""
    notes = results_csv.with_name("notes.md")
    existing = notes.read_text().splitlines() if notes.exists() else []
    line = f"- {note}"
    if line not in existing:
        notes.parent.mkdir(parents=True, exist_ok=True)
        with notes.open("a") as f:
            f.write(line + "\n")
