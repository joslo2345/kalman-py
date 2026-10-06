"""Guide scenario S3: nonlinear range-bearing tracking, 200 seeds x 500 steps.

Accuracy can't beat a correct UKF by much, so the bar is: match FilterPy's RMSE with the same
sigma-point parameters, and be statistically consistent (NEES within chi-squared bounds).
"""

from typing import Any

import numpy as np
import pytest

from kalman_py import UnscentedKalmanFilter

pytest.importorskip("jax")
filterpy_kalman = pytest.importorskip("filterpy.kalman")
stats = pytest.importorskip("scipy.stats")

from tests.nonlinear_scenarios import (
    cv_f,
    make_range_bearing,
    range_bearing_h,
    wrap_bearing_residual,
)

SEEDS, STEPS, N_STATE = 200, 500, 4
SIGMA_PARAMS: dict[str, Any] = {"alpha": 0.1, "beta": 2.0, "kappa": -1.0}  # same for both libraries


def _fx(x: np.ndarray, dt: float) -> np.ndarray:
    return np.array([x[0] + dt * x[2], x[1] + dt * x[3], x[2], x[3]])


def _hx(x: np.ndarray) -> np.ndarray:
    return np.array([np.hypot(x[0], x[1]), np.arctan2(x[1], x[0])])


def _residual(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    d = np.asarray(np.subtract(a, b), dtype=float)
    d[1] = (d[1] + np.pi) % (2 * np.pi) - np.pi
    return d


def _circular_z_mean(sigmas: np.ndarray, Wm: np.ndarray) -> np.ndarray:
    # FilterPy averages raw sigma-point measurements, so bearings need a circular mean.
    bearing = np.arctan2(Wm @ np.sin(sigmas[:, 1]), Wm @ np.cos(sigmas[:, 1]))
    return np.array([Wm @ sigmas[:, 0], bearing])


def run_filterpy(sc) -> tuple[np.ndarray, np.ndarray]:  # type: ignore[no-untyped-def]
    points = filterpy_kalman.MerweScaledSigmaPoints(n=N_STATE, **SIGMA_PARAMS)
    ukf = filterpy_kalman.UnscentedKalmanFilter(
        dim_x=N_STATE,
        dim_z=2,
        dt=1.0,
        fx=_fx,
        hx=_hx,
        points=points,
        residual_z=_residual,
        z_mean_fn=_circular_z_mean,
    )
    ukf.x, ukf.P, ukf.Q, ukf.R = sc.x0.copy(), sc.P0.copy(), sc.Q, sc.R
    means, covs = np.empty((STEPS, N_STATE)), np.empty((STEPS, N_STATE, N_STATE))
    for k, z in enumerate(sc.zs):
        ukf.predict()
        ukf.update(z)
        means[k], covs[k] = ukf.x, ukf.P
    return means, covs


def run_ours(sc) -> tuple[np.ndarray, np.ndarray]:  # type: ignore[no-untyped-def]
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
    res = ukf.filter(sc.zs, dt=1.0, backend="jax")
    return np.asarray(res.means), np.asarray(res.covs)


def nees(means: np.ndarray, covs: np.ndarray, truth: np.ndarray) -> np.ndarray:
    e = truth - means
    return np.asarray(np.einsum("ki,ki->k", e, np.linalg.solve(covs, e[..., None])[..., 0]))


@pytest.mark.slow
def test_ukf_matches_filterpy_accuracy_and_is_consistent() -> None:
    results: dict[str, dict[str, list[np.ndarray]]] = {
        "ours": {"sq_err": [], "nees": []},
        "filterpy": {"sq_err": [], "nees": []},
    }
    for seed in range(SEEDS):
        sc = make_range_bearing(seed=seed, steps=STEPS)
        for name, run in (("ours", run_ours), ("filterpy", run_filterpy)):
            means, covs = run(sc)
            results[name]["sq_err"].append(((means[:, :2] - sc.truth[:, :2]) ** 2).sum(axis=1))
            results[name]["nees"].append(nees(means, covs, sc.truth))

    rmse = {name: float(np.sqrt(np.mean(r["sq_err"]) / 2)) for name, r in results.items()}
    # Average NEES per time step over the runs vs. its 95% chi-squared interval (Bar-Shalom).
    lo, hi = stats.chi2.ppf([0.025, 0.975], SEEDS * N_STATE) / SEEDS
    inside = {
        name: float(np.mean((lo <= (a := np.mean(r["nees"], axis=0))) & (a <= hi)))
        for name, r in results.items()
    }
    print(
        f"[report] S3 UKF position rmse: ours {rmse['ours']:.4f}, FilterPy {rmse['filterpy']:.4f}; "
        f"steps with ANEES in 95% bounds: ours {inside['ours']:.1%}, "
        f"FilterPy {inside['filterpy']:.1%}"
    )
    assert rmse["ours"] <= 1.01 * rmse["filterpy"]
    assert inside["ours"] >= 0.90
