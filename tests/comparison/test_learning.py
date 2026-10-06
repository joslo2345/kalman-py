"""Guide Step 6: noise learning vs pykalman's EM.

pykalman treats its initial state as the first state (no prediction before the first
measurement), so its EM maximizes a slightly different likelihood. Both fits are therefore
scored with our filter, under one convention.
"""

import time

import numpy as np
import pytest

from kalman_py import KalmanFilter
from kalman_py.learning import fit_noise
from tests.scenarios import LinearScenario, make_cv_2d

pykalman = pytest.importorskip("pykalman")

EM_VARS = ["transition_covariance", "observation_covariance"]


@pytest.fixture(scope="module")
def cv_scenario_unknown_noise() -> LinearScenario:
    return make_cv_2d(seed=7, steps=1000)


def pykalman_em(sc: LinearScenario, n_iter: int) -> tuple[np.ndarray, np.ndarray, float]:
    pk = pykalman.KalmanFilter(
        transition_matrices=sc.F,
        observation_matrices=sc.H,
        initial_state_mean=sc.x0,
        initial_state_covariance=sc.P0,
    )
    t0 = time.perf_counter()
    pk = pk.em(sc.zs, n_iter=n_iter, em_vars=EM_VARS)
    return pk.transition_covariance, pk.observation_covariance, time.perf_counter() - t0


def log_likelihood(sc: LinearScenario, Q: np.ndarray, R: np.ndarray) -> float:
    return float(KalmanFilter(sc.F, sc.H, Q, R, sc.x0, sc.P0).filter(sc.zs).log_likelihood)


@pytest.mark.slow
def test_noise_learning_beats_pykalman_em(cv_scenario_unknown_noise: LinearScenario) -> None:
    sc = cv_scenario_unknown_noise
    t0 = time.perf_counter()
    ours = fit_noise(sc.F, sc.H, sc.zs, sc.x0, sc.P0)  # gradient when JAX is installed
    t_ours = time.perf_counter() - t0
    Q_pk, R_pk, t_pk = pykalman_em(sc, n_iter=50)
    ll_pk = log_likelihood(sc, Q_pk, R_pk)

    print(
        f"[report] noise learning: ours {ours.log_likelihood:.3f} in {t_ours:.2f}s "
        f"(converged={ours.converged}), pykalman EM x50 {ll_pk:.3f} in {t_pk:.2f}s"
    )
    assert ours.log_likelihood >= ll_pk - 1e-6
    assert t_ours < t_pk
    # Recovers the truth: variances within 10%, and the fitted cross-term (true value 0) is a
    # negligible correlation. (A relative tolerance on R itself fails on any nonzero estimate of
    # an off-diagonal zero.)
    np.testing.assert_allclose(np.diagonal(ours.R), np.diagonal(sc.R), rtol=0.1)
    assert abs(ours.R[0, 1]) / np.sqrt(ours.R[0, 0] * ours.R[1, 1]) < 0.1


@pytest.mark.slow
def test_em_iterations_match_pykalman_faster(cv_scenario_unknown_noise: LinearScenario) -> None:
    sc = cv_scenario_unknown_noise
    t0 = time.perf_counter()
    ours = fit_noise(sc.F, sc.H, sc.zs, sc.x0, sc.P0, method="em", max_iter=50, tol=0)
    t_ours = time.perf_counter() - t0
    Q_pk, R_pk, t_pk = pykalman_em(sc, n_iter=50)

    print(f"[report] EM x50: ours {t_ours:.2f}s vs pykalman {t_pk:.2f}s ({t_pk / t_ours:.1f}x)")
    assert ours.log_likelihood >= log_likelihood(sc, Q_pk, R_pk) - 1e-6
    np.testing.assert_allclose(ours.R, R_pk, rtol=0.05)  # same algorithm, same answer
    assert t_ours < t_pk
