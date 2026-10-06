"""Guide Step 6 'Consistency': Monte Carlo runs with the true state known; average NEES and
NIS must fall inside their 95% chi-squared bounds."""

import numpy as np
import pytest

from kalman_py import KalmanFilter
from kalman_py._common import Backend
from kalman_py.diagnostics import consistency_check, nees
from tests.scenarios import LinearScenario, make_cv_2d

RUNS, STEPS = 500, 100


@pytest.fixture(scope="module")
def monte_carlo_runs() -> list[LinearScenario]:
    return [make_cv_2d(seed=seed, steps=STEPS) for seed in range(RUNS)]


def run_filters(runs: list[LinearScenario], backend: Backend, Q_scale: float = 1.0) -> tuple:  # type: ignore[type-arg]
    nees_runs, nis_runs = [], []
    for sc in runs:
        kf = KalmanFilter(sc.F, sc.H, Q_scale * sc.Q, sc.R, sc.x0, sc.P0)
        res = kf.filter(sc.zs, backend=backend)
        nees_runs.append(nees(sc.truth, np.asarray(res.means), np.asarray(res.covs)))
        nis_runs.append(np.asarray(res.nis))
    return np.array(nees_runs), np.array(nis_runs)


@pytest.mark.parametrize("backend", ["numpy", "jax"])
def test_kalman_filter_is_consistent(
    monte_carlo_runs: list[LinearScenario], backend: Backend
) -> None:
    if backend == "jax":
        pytest.importorskip("jax")
    nees_runs, nis_runs = run_filters(monte_carlo_runs, backend)
    sc = monte_carlo_runs[0]
    anees = consistency_check(nees_runs, dof=sc.n)
    anis = consistency_check(nis_runs, dof=sc.m)
    print(
        f"[report] {backend} KF over {RUNS} runs: ANEES inside 95% bounds at "
        f"{anees.fraction_inside:.1%} of steps, ANIS at {anis.fraction_inside:.1%}"
    )
    # NEES is strongly autocorrelated over time (lag-1 ~0.74 here), so excursions come in clusters
    # and the fraction inside varies far more than binomially (90-92% across seed sets). The time
    # average is the robust check: averaging over steps can only shrink the variance, so the
    # single-step bounds still hold for it, conservatively. NIS is white and needs no slack.
    for check in (anees, anis):
        assert check.lower <= check.average.mean() <= check.upper
    assert anees.fraction_inside >= 0.85
    assert anis.fraction_inside >= 0.9


def test_mis_specified_filter_is_detected(monte_carlo_runs: list[LinearScenario]) -> None:
    # Telling the filter the process noise is 10x smaller than it is makes it overconfident.
    nees_runs, nis_runs = run_filters(monte_carlo_runs, "numpy", Q_scale=0.1)
    sc = monte_carlo_runs[0]
    anees = consistency_check(nees_runs, dof=sc.n)
    assert anees.fraction_inside < 0.5
    assert np.median(anees.average) > anees.upper
    assert consistency_check(nis_runs, dof=sc.m).fraction_inside < 0.5
