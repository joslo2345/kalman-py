import os
from pathlib import Path

import numpy as np
import pytest

from kalman_py import KalmanFilter

# Benchmark tooling lives in the repository, not in the sdist.
naive_filter = pytest.importorskip("benchmarks.baselines.naive").naive_filter
make_table_module = pytest.importorskip("scripts.make_table")
make_table, ratio, update_readme = (
    make_table_module.make_table,
    make_table_module.ratio,
    make_table_module.update_readme,
)
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
    readme.write_text(
        "intro\n<!-- BENCH:START -->\nold\n<!-- BENCH:END -->\noutro\n", encoding="utf-8"
    )
    update_readme(readme, "new table")
    assert (
        readme.read_text(encoding="utf-8")
        == "intro\n<!-- BENCH:START -->\nnew table\n<!-- BENCH:END -->\noutro\n"
    )
    readme.write_text("no markers", encoding="utf-8")
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
    lines = out.read_text(encoding="utf-8").splitlines()
    assert lines[0].startswith("library,library_version,scenario")
    assert len(lines) == 1 + 16
    assert (tmp_path / "notes.md").exists()


def test_compare_results_flags_regressions_beyond_tolerance(tmp_path: Path) -> None:
    import subprocess
    import sys

    header = "library,library_version,scenario,filter,precision,metric,value,unit,commit,cpu,os,toolchain,date\n"
    rest = ",ns,c,cpu,os,tool,date\n"

    def write(name: str, ours: float, variant: float, other: float) -> Path:
        path = tmp_path / name
        path.write_text(
            header
            + f"ours,1,S1,KF batch,float64,time_per_step,{ours}{rest}"
            + f"ours-numpy,1,S1,KF batch,float64,time_per_step,{variant}{rest}"
            + f"other,1,S1,KF batch,float64,time_per_step,{other}{rest}",  # not ours: ignored
            encoding="utf-8",
        )
        return path

    base = write("base.csv", 100.0, 100.0, 100.0)
    script = Path(__file__).resolve().parents[2] / "scripts" / "compare_results.py"

    def run(current: Path, *flags: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [sys.executable, str(script), str(base), str(current), "ours", *flags],
            capture_output=True,
            encoding="utf-8",
            # The real environment (Windows needs SystemRoot to start Python), minus GitHub's
            # annotation mode, so the report prints the same everywhere.
            env={
                **{k: v for k, v in os.environ.items() if k != "GITHUB_ACTIONS"},
                "PYTHONIOENCODING": "utf-8",
            },
            check=False,  # the exit status is what's being tested
        )

    within = write("within.csv", 109.0, 95.0, 500.0)
    assert run(within, "--fail").returncode == 0

    slower = write("slower.csv", 100.0, 120.0, 100.0)
    report = run(slower)
    assert report.returncode == 0  # report-only by default
    assert "| ours-numpy | S1 | KF batch | 100.0 | 120.0 | +20.0% ⚠️ |" in report.stdout
    assert "other" not in report.stdout
    assert run(slower, "--fail").returncode == 1
