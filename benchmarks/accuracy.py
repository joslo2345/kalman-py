"""Nonlinear accuracy (S3) and numerical stability (S4) against FilterPy and a naive baseline.

Usage: python benchmarks/accuracy.py "commit,cpu,os,toolchain,date" [--scenarios S3,S4]

S3: EKF and UKF position+velocity RMSE and average NEES over 200 seeds, with the same
sigma-point parameters and bearing handling in every library.
S4 (float32), steps completed before the first covariance that
- steps_to_failure: is non-finite or whose symmetric part has no Cholesky factorization in
  float32, i.e. is no longer usable as a covariance at working precision (the criterion S4's
  selection rule uses);
- steps_to_indefinite: is non-finite or whose symmetric part has a non-positive eigenvalue when
  evaluated exactly (in float64): genuinely indefinite, not just near-singular for float32.
1,000,000 means the filter survived every step.
"""

from __future__ import annotations

import argparse
import sys
import warnings
from collections.abc import Callable
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import jax

from benchmarks.baselines.naive import naive_filter
from benchmarks.results import add_note, append_rows
from benchmarks.run import OURS, library_version
from benchmarks.scenarios import Scenario, load
from kalman_py import ExtendedKalmanFilter, KalmanFilter, UnscentedKalmanFilter
from kalman_py.diagnostics import nees
from tests.nonlinear_scenarios import cv_f, range_bearing_h, wrap_bearing_residual

warnings.filterwarnings("ignore", category=SyntaxWarning)  # FilterPy's docstrings
from filterpy.kalman import ExtendedKalmanFilter as FPExtendedKalmanFilter
from filterpy.kalman import KalmanFilter as FPKalmanFilter
from filterpy.kalman import MerweScaledSigmaPoints
from filterpy.kalman import UnscentedKalmanFilter as FPUnscentedKalmanFilter

jax.config.update("jax_enable_x64", True)

SIGMA_PARAMS = {"alpha": 0.1, "beta": 2.0, "kappa": -1.0}  # S3, every library


# ---- S3 model functions for FilterPy (NumPy) --------------------------------------------------


def _fx(x: np.ndarray, dt: float) -> np.ndarray:
    return np.array([x[0] + dt * x[2], x[1] + dt * x[3], x[2], x[3]])


def _hx(x: np.ndarray) -> np.ndarray:
    x = np.ravel(x)
    return np.array([np.hypot(x[0], x[1]), np.arctan2(x[1], x[0])])


def _h_jacobian(x: np.ndarray) -> np.ndarray:
    px, py = np.ravel(x)[:2]
    r2 = px**2 + py**2
    r = np.sqrt(r2)
    return np.array([[px / r, py / r, 0.0, 0.0], [-py / r2, px / r2, 0.0, 0.0]])


def _residual(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    d = np.asarray(np.subtract(a, b), dtype=float)
    d[1] = (d[1] + np.pi) % (2 * np.pi) - np.pi
    return d


def _circular_z_mean(sigmas: np.ndarray, Wm: np.ndarray) -> np.ndarray:
    bearing = np.arctan2(Wm @ np.sin(sigmas[:, 1]), Wm @ np.cos(sigmas[:, 1]))
    return np.array([Wm @ sigmas[:, 0], bearing])


# ---- S3 runs: each returns posterior means (T, n) and covariances (T, n, n) --------------------

Run3 = Callable[[Scenario, np.ndarray], tuple[np.ndarray, np.ndarray]]


def ours_ekf(sc: Scenario, zs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    ekf = ExtendedKalmanFilter(
        f=cv_f,
        h=range_bearing_h,
        Q=sc.Q,
        R=sc.R,
        x0=sc.x0,
        P0=sc.P0,
        residual_z=wrap_bearing_residual,
    )
    res = ekf.filter(zs, dt=sc.dt, backend="jax")
    return np.asarray(res.means), np.asarray(res.covs)


def ours_ukf(sc: Scenario, zs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    ukf = UnscentedKalmanFilter(
        f=cv_f,
        h=range_bearing_h,
        Q=sc.Q,
        R=sc.R,
        x0=sc.x0,
        P0=sc.P0,
        residual_z=wrap_bearing_residual,
        **SIGMA_PARAMS,
    )
    res = ukf.filter(zs, dt=sc.dt, backend="jax")
    return np.asarray(res.means), np.asarray(res.covs)


def filterpy_ekf(sc: Scenario, zs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    ekf = FPExtendedKalmanFilter(dim_x=sc.n, dim_z=sc.m)
    ekf.F, ekf.Q, ekf.R = sc.F, sc.Q, sc.R
    ekf.x, ekf.P = sc.x0.reshape(-1, 1).copy(), sc.P0.copy()
    means, covs = np.empty((len(zs), sc.n)), np.empty((len(zs), sc.n, sc.n))

    def hx(x: np.ndarray) -> np.ndarray:
        return _hx(x).reshape(-1, 1)

    def residual(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        return _residual(np.ravel(a), np.ravel(b)).reshape(-1, 1)

    for k, z in enumerate(zs):
        ekf.predict()  # linear constant-velocity transition: x = F x
        ekf.update(z.reshape(-1, 1), _h_jacobian, hx, residual=residual)
        means[k], covs[k] = ekf.x.ravel(), ekf.P
    return means, covs


def filterpy_ukf(sc: Scenario, zs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    points = MerweScaledSigmaPoints(n=sc.n, **SIGMA_PARAMS)
    ukf = FPUnscentedKalmanFilter(
        dim_x=sc.n,
        dim_z=sc.m,
        dt=sc.dt,
        fx=_fx,
        hx=_hx,
        points=points,
        residual_z=_residual,
        z_mean_fn=_circular_z_mean,
    )
    ukf.x, ukf.P, ukf.Q, ukf.R = sc.x0.copy(), sc.P0.copy(), sc.Q, sc.R
    means, covs = np.empty((len(zs), sc.n)), np.empty((len(zs), sc.n, sc.n))
    for k, z in enumerate(zs):
        ukf.predict()
        ukf.update(z)
        means[k], covs[k] = ukf.x, ukf.P
    return means, covs


S3_RUNS: list[tuple[str, str, Run3]] = [
    (OURS, "EKF", ours_ekf),
    ("filterpy", "EKF", filterpy_ekf),
    (OURS, "UKF", ours_ukf),
    ("filterpy", "UKF", filterpy_ukf),
]


def run_s3(env: list[str], out: Path) -> list[list[str]]:
    sc = load("S3")
    assert sc.truth is not None
    rows = []
    for library, filt, fn in S3_RUNS:
        sq_err, nees_values = [], []
        for seed in range(sc.measurements.shape[0]):
            means, covs = fn(sc, sc.measurements[seed])
            truth = sc.truth[seed]
            sq_err.append((means - truth) ** 2)
            nees_values.append(nees(truth, means, covs))
        rmse = float(np.sqrt(np.mean(sq_err)))
        avg_nees = float(np.mean(nees_values))
        row = [library, library_version(library), "S3", filt, "float64"]
        rows.append([*row, "rmse", f"{rmse:.6g}", "state", *env])
        rows.append([*row, "nees", f"{avg_nees:.4g}", "-", *env])
        print(f"S3 {filt} {library:10s} rmse {rmse:.6g}  average NEES {avg_nees:.4g}", flush=True)
    add_note(
        out,
        "S3 UKF: kalman-py redraws sigma points from the predicted distribution before each "
        "update, while FilterPy reuses the propagated ones, so their estimates differ slightly; "
        "the EKFs agree to ~1e-13. A consistent filter has average NEES = 4.",
    )
    return rows


# ---- S4 ----------------------------------------------------------------------------------------


def steps_to_failure(covs: np.ndarray, chunk: int = 10_000) -> int:
    """Steps before the first covariance with no float32 Cholesky factor (or non-finite)."""
    for start in range(0, len(covs), chunk):
        block = covs[start : start + chunk]
        block = 0.5 * (block + np.swapaxes(block, 1, 2))
        try:  # fast path: the whole block factors
            if np.isfinite(block).all():
                np.linalg.cholesky(block)
                continue
        except np.linalg.LinAlgError:
            pass
        for i, P in enumerate(block):
            try:
                if not np.isfinite(P).all():
                    return start + i
                np.linalg.cholesky(P)
            except np.linalg.LinAlgError:
                return start + i
    return len(covs)


def steps_to_indefinite(covs: np.ndarray, chunk: int = 100_000) -> int:
    """Steps before the first covariance whose symmetric part has an eigenvalue <= 0 exactly."""
    for start in range(0, len(covs), chunk):
        block = np.asarray(covs[start : start + chunk], dtype=np.float64)
        finite = np.isfinite(block).all(axis=(1, 2))
        min_eig = np.full(len(block), -np.inf)
        min_eig[finite] = np.linalg.eigvalsh(
            0.5 * (block[finite] + np.swapaxes(block[finite], 1, 2))
        ).min(axis=1)
        bad = np.flatnonzero(min_eig <= 0)
        if bad.size:
            return start + int(bad[0])
    return len(covs)


def filterpy_keeps_float32(sc: Scenario) -> bool:
    """Run a few float32 steps and check whether FilterPy's covariance stayed float32."""
    kf = FPKalmanFilter(dim_x=sc.n, dim_z=sc.m)
    kf.F, kf.H, kf.Q, kf.R = sc.F, sc.H, sc.Q, sc.R
    kf.x, kf.P = sc.x0.reshape(-1, 1).copy(), sc.P0.copy()
    for z in sc.zs[:3]:
        kf.predict()
        kf.update(z)
    return bool(kf.P.dtype == np.float32)


def run_s4(env: list[str], out: Path) -> list[list[str]]:
    sc = load("S4")
    assert sc.dtype == np.float32
    p = sc.params
    runs: list[tuple[str, Callable[[], np.ndarray]]] = [
        (OURS, lambda: KalmanFilter(**p).filter(sc.zs).covs),
        (f"{OURS}-sqrt", lambda: KalmanFilter(**p, square_root=True).filter(sc.zs).covs),
        (f"{OURS}-jax", lambda: np.asarray(KalmanFilter(**p).filter(sc.zs, backend="jax").covs)),
        ("naive", lambda: naive_filter(sc.F, sc.H, sc.Q, sc.R, sc.x0, sc.P0, sc.zs)[1]),
    ]
    rows = []
    for library, fn in runs:
        covs = fn()
        if covs.dtype != np.float32:
            raise RuntimeError(f"{library} returned {covs.dtype} covariances on a float32 problem")
        version = "textbook" if library == "naive" else library_version(library)
        row = [library, version, "S4", "KF", "float32"]
        failure, indefinite = steps_to_failure(covs), steps_to_indefinite(covs)
        rows.append([*row, "steps_to_failure", str(failure), "steps", *env])
        rows.append([*row, "steps_to_indefinite", str(indefinite), "steps", *env])
        print(
            f"S4 {library:16s} steps_to_failure {failure:8d}  steps_to_indefinite {indefinite:8d}"
        )
    if filterpy_keeps_float32(sc):
        raise NotImplementedError("FilterPy kept float32: add it to the S4 runs")
    add_note(
        out,
        "S4 (float32): FilterPy is n/a because it converts float32 inputs to float64 internally "
        "(its identity matrix is float64), so it can't run S4 as specified.",
    )
    add_note(
        out,
        f"S4: `{OURS}` is the default configuration (NumPy backend, Joseph form); "
        f"`{OURS}-sqrt` uses `square_root=True`; `{OURS}-jax` is the JAX backend. "
        "1,000,000 means the filter never failed.",
    )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("env", help='"commit,cpu,os,toolchain,date"')
    parser.add_argument("--scenarios", default="S3,S4")
    parser.add_argument("--out", type=Path, default=Path("results/results.csv"))
    args = parser.parse_args()
    env = args.env.split(",")
    if len(env) != 5:
        parser.error('env must have 5 comma-separated fields: "commit,cpu,os,toolchain,date"')
    rows = []
    for sid in args.scenarios.split(","):
        rows += run_s3(env, args.out) if sid == "S3" else run_s4(env, args.out)
    append_rows(args.out, rows)
    print(f"appended {len(rows)} rows to {args.out}")


if __name__ == "__main__":
    main()
