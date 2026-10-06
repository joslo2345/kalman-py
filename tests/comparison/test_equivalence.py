"""Linear-Gaussian filtering is optimal, so we must match the reference libraries exactly."""

import numpy as np
import pytest

from kalman_py import KalmanFilter
from tests.scenarios import LinearScenario

filterpy_kalman = pytest.importorskip("filterpy.kalman")
pykalman = pytest.importorskip("pykalman")


def test_filter_matches_filterpy(cv_scenario: LinearScenario) -> None:
    sc = cv_scenario
    ours = KalmanFilter(**sc.params).filter(sc.zs)

    fp = filterpy_kalman.KalmanFilter(dim_x=sc.n, dim_z=sc.m)
    fp.F, fp.H, fp.Q, fp.R = sc.F, sc.H, sc.Q, sc.R
    fp.x, fp.P = sc.x0.reshape(-1, 1).copy(), sc.P0.copy()

    for k, z in enumerate(sc.zs):
        fp.predict()
        fp.update(z)
        np.testing.assert_allclose(ours.means[k], fp.x.ravel(), rtol=1e-9, atol=1e-12)
        np.testing.assert_allclose(ours.covs[k], fp.P, rtol=1e-9, atol=1e-12)


def _pykalman_with_aligned_prior(sc: LinearScenario) -> object:
    # pykalman updates with the first measurement before predicting; we predict first.
    # Moving the prior forward one step makes the two conventions identical.
    return pykalman.KalmanFilter(
        transition_matrices=sc.F,
        observation_matrices=sc.H,
        transition_covariance=sc.Q,
        observation_covariance=sc.R,
        initial_state_mean=sc.F @ sc.x0,
        initial_state_covariance=sc.F @ sc.P0 @ sc.F.T + sc.Q,
    )


def test_filter_matches_pykalman(cv_scenario: LinearScenario) -> None:
    sc = cv_scenario
    ours = KalmanFilter(**sc.params).filter(sc.zs)
    means, covs = _pykalman_with_aligned_prior(sc).filter(sc.zs)  # type: ignore[attr-defined]

    np.testing.assert_allclose(ours.means, means, rtol=1e-9, atol=1e-12)
    np.testing.assert_allclose(ours.covs, covs, rtol=1e-9, atol=1e-12)


def test_log_likelihood_matches_pykalman(cv_scenario: LinearScenario) -> None:
    sc = cv_scenario
    ours = KalmanFilter(**sc.params).filter(sc.zs)
    theirs = _pykalman_with_aligned_prior(sc).loglikelihood(sc.zs)  # type: ignore[attr-defined]
    np.testing.assert_allclose(ours.log_likelihood, theirs, rtol=1e-9)
