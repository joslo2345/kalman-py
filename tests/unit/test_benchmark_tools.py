from pathlib import Path

import numpy as np
import pytest

from benchmarks.baselines.naive import naive_filter
from kalman_py import KalmanFilter
from scripts.make_table import make_table, ratio, update_readme
from tests.scenarios import LinearScenario


def row(library: str, metric: str, value: float, scenario: str = "S2") -> dict[str, str]:
    return {
        "library": library,
        "library_version": "1",
        "scenario": scenario,
        "filter": "KF",
        "precision": "float64",
        "metric": metric,
        "value": str(value),
        "unit": "u",
        "cpu": "arm64",
        "os": "Darwin",
        "toolchain": "python-3.12",
        "commit": "abc123",
        "date": "2026-10-05",
    }


def test_ratio_compares_with_best_non_variant() -> None:
    vals = {"ours": 10.0, "ours-numpy": 1.0, "a": 50.0, "b": 20.0}
    assert ratio("time_per_step", vals, "ours") == "2.00x"  # best other is b; variant ignored
    assert ratio("steps_to_failure", {"ours": 10.0, "a": 5.0}, "ours") == "2.00x"
    assert ratio("steps_to_failure", {"ours": 2753.0, "a": 1e6}, "ours") == "0.0028x"
    assert ratio("nees", vals, "ours") == "–"
    assert ratio("rmse", {"a": 1.0}, "ours") == "n/a"
    assert ratio("rmse", {"ours": 1.0, "ours-jax": 1.0}, "ours") == "n/a"


def test_table_has_environment_columns_and_notes() -> None:
    rows = [row("ours", "rmse", 0.5), row("other", "rmse", 1.0), row("other", "peak_memory", 2e6)]
    table = make_table(rows, "ours", ["- a footnote"])
    lines = table.splitlines()
    assert lines[0] == "Measured on arm64, Darwin, python-3.12, commit abc123, 2026-10-05."
    assert "| Scenario | Filter | Precision | Metric | ours | other | Ours vs best other |" in lines
    assert "| S2 | KF | float64 | rmse (u) | 0.5 | 1 | 2.00x |" in lines
    assert "| S2 | KF | float64 | peak_memory (u) | n/a | 2,000,000 | n/a |" in lines
    assert lines[-1] == "- a footnote"


def test_update_readme_replaces_only_the_marked_block(tmp_path: Path) -> None:
    readme = tmp_path / "README.md"
    readme.write_text("intro\n<!-- BENCH:START -->\nold\n<!-- BENCH:END -->\noutro\n")
    update_readme(readme, "new table")
    assert (
        readme.read_text() == "intro\n<!-- BENCH:START -->\nnew table\n<!-- BENCH:END -->\noutro\n"
    )
    readme.write_text("no markers")
    with pytest.raises(SystemExit):
        update_readme(readme, "x")


def test_failure_criteria() -> None:
    pytest.importorskip("filterpy")
    pytest.importorskip("jax")
    from benchmarks.accuracy import steps_to_failure, steps_to_indefinite

    covs = np.tile(np.eye(2, dtype=np.float32), (10, 1, 1))
    assert steps_to_failure(covs) == steps_to_indefinite(covs) == 10
    # PD exactly, but too ill-conditioned to factor in float32.
    covs[4] = np.array([[1.0, 1.0], [1.0, 1.0 + 2e-8]], dtype=np.float32) + np.float32(0)
    covs[4, 1, 1] = np.nextafter(np.float32(1.0), np.float32(2.0))
    assert steps_to_indefinite(covs) == 10
    covs[7] = np.diag([1.0, -1e-3]).astype(np.float32)
    assert steps_to_indefinite(covs) == 7
    covs[2, 0, 0] = np.nan
    assert steps_to_failure(covs) == steps_to_indefinite(covs) == 2


def test_naive_baseline_matches_kalman_filter_when_well_conditioned(
    cv_scenario: LinearScenario,
) -> None:
    sc = cv_scenario
    means, covs = naive_filter(sc.F, sc.H, sc.Q, sc.R, sc.x0, sc.P0, sc.zs)
    expected = KalmanFilter(**sc.params).filter(sc.zs)
    np.testing.assert_allclose(means, expected.means, rtol=1e-8, atol=1e-10)
    np.testing.assert_allclose(covs, expected.covs, rtol=1e-6, atol=1e-10)


@pytest.mark.slow
def test_benchmark_harness_smoke(tmp_path: Path) -> None:
    import subprocess
    import sys

    pytest.importorskip("filterpy")
    pytest.importorskip("pykalman")
    out = tmp_path / "results.csv"
    subprocess.run(
        [
            sys.executable,
            "benchmarks/run.py",
            "c,cpu,os,tool,date",
            "--repeats",
            "1",
            "--scenarios",
            "S1",
            "--out",
            str(out),
        ],
        check=True,
        cwd=Path(__file__).resolve().parents[2],
    )
    lines = out.read_text().splitlines()
    assert lines[0].startswith("library,library_version,scenario")
    assert len(lines) == 1 + 16
    assert (tmp_path / "notes.md").exists()
