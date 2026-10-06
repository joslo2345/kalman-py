"""Speed, accuracy and memory of linear Kalman filtering (S1, S2, S5) against FilterPy and pykalman.

Usage: python benchmarks/run.py "commit,cpu,os,toolchain,date" [--repeats 30] [--scenarios S1,S2,S5]

Appends rows to results/results.csv (schema in the repo guide, Step 7):
library,library_version,scenario,filter,precision,metric,value,unit,commit,cpu,os,toolchain,date

Two modes, since the baselines work differently:
- per-step: one predict + update call per measurement (real-time loop): ours vs FilterPy;
- batch: the whole sequence in one call: our JAX and NumPy backends vs pykalman.
Only predict + update is timed: setup, loading and conversion are outside the timed function.
"""

from __future__ import annotations

import argparse
import statistics
import sys
import time
import tracemalloc
import warnings
from collections.abc import Callable
from functools import partial
from importlib.metadata import version
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import jax

from benchmarks.results import add_note, append_rows
from benchmarks.scenarios import Scenario, load
from kalman_py import KalmanFilter

warnings.filterwarnings("ignore", category=SyntaxWarning)  # FilterPy's docstrings
from filterpy.kalman import KalmanFilter as FPKalmanFilter
from pykalman import KalmanFilter as PKKalmanFilter

jax.config.update("jax_enable_x64", True)

OURS = "kalman-py"
Run = Callable[[Scenario], np.ndarray]


def median_ns_per_step(fn: Callable[[], object], steps: int, repeats: int) -> float:
    fn()  # warm-up; for JAX this also keeps compilation out of the timing
    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        times.append(time.perf_counter() - t0)
    return statistics.median(times) / steps * 1e9


def peak_bytes(fn: Callable[[], object]) -> int:
    """Peak Python heap during one run. tracemalloc can't see JAX device buffers."""
    tracemalloc.start()
    fn()
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return peak


# ---- Per-step mode ---------------------------------------------------------------------------


def filterpy_step(sc: Scenario) -> np.ndarray:
    kf = FPKalmanFilter(dim_x=sc.n, dim_z=sc.m)
    kf.F, kf.H, kf.Q, kf.R = sc.F, sc.H, sc.Q, sc.R
    kf.x, kf.P = sc.x0.reshape(-1, 1).copy(), sc.P0.copy()
    out = np.empty((len(sc.zs), sc.n))
    for k, z in enumerate(sc.zs):
        kf.predict()
        kf.update(z)
        out[k] = kf.x.ravel()
    return out


def ours_step(sc: Scenario) -> np.ndarray:
    kf = KalmanFilter(**sc.params)
    out = np.empty((len(sc.zs), sc.n))
    for k, z in enumerate(sc.zs):
        kf.predict()
        kf.update(z)
        out[k] = kf.x
    return out


# ---- Batch mode ------------------------------------------------------------------------------


def pykalman_batch(sc: Scenario) -> np.ndarray:
    # pykalman updates with the first measurement before its first prediction, while FilterPy
    # and our library predict first. Moving the prior forward one step aligns the conventions.
    kf = PKKalmanFilter(
        transition_matrices=sc.F,
        observation_matrices=sc.H,
        transition_covariance=sc.Q,
        observation_covariance=sc.R,
        initial_state_mean=sc.F @ sc.x0,
        initial_state_covariance=sc.F @ sc.P0 @ sc.F.T + sc.Q,
    )
    means, _ = kf.filter(sc.zs)
    return np.asarray(means)


def ours_batch_numpy(sc: Scenario) -> np.ndarray:
    return KalmanFilter(**sc.params).filter(sc.zs).means


def ours_batch_jax(sc: Scenario) -> np.ndarray:
    result = KalmanFilter(**sc.params).filter(sc.zs, backend="jax")
    # block_until_ready stops the timer only after JAX has actually finished
    return np.asarray(jax.block_until_ready(result.means))


# (library, filter label, function, is JAX)
RUNS: list[tuple[str, str, Run, bool]] = [
    (OURS, "KF per-step", ours_step, False),
    ("filterpy", "KF per-step", filterpy_step, False),
    (OURS, "KF batch", ours_batch_jax, True),
    (f"{OURS}-numpy", "KF batch", ours_batch_numpy, False),
    ("pykalman", "KF batch", pykalman_batch, False),
]
# max_abs_diff rows: (ours, baseline) pairs per mode, on the scenarios that check equivalence.
DIFF_PAIRS = {"KF per-step": (ours_step, "filterpy"), "KF batch": (ours_batch_jax, "pykalman")}
DIFF_SCENARIOS = {"S1", "S2"}


def library_version(library: str) -> str:
    return version(OURS) if library.startswith(OURS) else version(library)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("env", help='"commit,cpu,os,toolchain,date"')
    parser.add_argument("--repeats", type=int, default=30)
    parser.add_argument("--scenarios", default="S1,S2,S5")
    parser.add_argument("--out", type=Path, default=Path("results/results.csv"))
    args = parser.parse_args()
    env = args.env.split(",")
    if len(env) != 5:
        parser.error('env must have 5 comma-separated fields: "commit,cpu,os,toolchain,date"')

    rows = []
    for sid in args.scenarios.split(","):
        sc = load(sid)
        truth = sc.truth_single
        estimates: dict[tuple[str, str], np.ndarray] = {}
        for library, filt, fn, is_jax in RUNS:
            est = fn(sc)
            estimates[(library, filt)] = est
            ns = median_ns_per_step(partial(fn, sc), len(sc.zs), args.repeats)
            rmse = float(np.sqrt(np.mean((est - truth) ** 2)))
            row = [library, library_version(library), sid, filt, "float64"]
            rows.append([*row, "time_per_step", f"{ns:.1f}", "ns", *env])
            rows.append([*row, "rmse", f"{rmse:.6g}", "state", *env])
            if not is_jax:
                rows.append([*row, "peak_memory", str(peak_bytes(partial(fn, sc))), "bytes", *env])
            print(f"{sid} {filt:12s} {library:16s} {ns:10.1f} ns/step  rmse {rmse:.6g}", flush=True)
        if sid in DIFF_SCENARIOS:
            for filt, (ours_fn, baseline) in DIFF_PAIRS.items():
                ours_est = estimates[(OURS, filt)]
                diff = float(np.max(np.abs(ours_est - estimates[(baseline, filt)])))
                row = [baseline, library_version(baseline), sid, filt, "float64"]
                rows.append([*row, "max_abs_diff", f"{diff:.3g}", "state", *env])
    append_rows(args.out, rows)
    add_note(
        args.out,
        f"KF per-step rows: `{OURS}` is the step-by-step API (NumPy). KF batch rows: `{OURS}` is "
        f"the JAX backend on CPU and `{OURS}-numpy` the NumPy backend; JAX has no peak_memory "
        "because tracemalloc can't see its buffers. max_abs_diff is the largest difference "
        f"between `{OURS}` and that library's estimates.",
    )
    print(f"appended {len(rows)} rows to {args.out}")


if __name__ == "__main__":
    main()
