from typing import Any

import numpy as np
import pytest

from kalman_py import KalmanFilter, UnscentedKalmanFilter
from kalman_py._common import Backend
from kalman_py.backends import numpy_backend
from tests.scenarios import LinearScenario

BACKENDS: list[Backend] = ["numpy", "jax"]
GUIDE_SIGMA_PARAMS = {"alpha": 0.1, "beta": 2.0, "kappa": -1.0}  # FilterPy comparison (S3)


def skip_without_jax(backend: Backend) -> None:
    if backend == "jax":
        pytest.importorskip("jax")


def linear_ukf(sc: LinearScenario, **kwargs: Any) -> UnscentedKalmanFilter:
    F, H = sc.F, sc.H
    return UnscentedKalmanFilter(
        f=lambda x, dt: F @ x, h=lambda x: H @ x, Q=sc.Q, R=sc.R, x0=sc.x0, P0=sc.P0, **kwargs
    )


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("params", [{}, GUIDE_SIGMA_PARAMS], ids=["default", "guide-params"])
def test_linear_model_reduces_to_kalman_filter(
    cv_scenario: LinearScenario, backend: Backend, params: dict[str, float]
) -> None:
    # The unscented transform is exact for linear functions, whatever the sigma-point weights.
    skip_without_jax(backend)
    sc = cv_scenario
    ukf = linear_ukf(sc, **params)
    kf = KalmanFilter(**sc.params)
    ours, expected = ukf.filter(sc.zs, dt=0.1, backend=backend), kf.filter(sc.zs)

    np.testing.assert_allclose(np.asarray(ours.means), expected.means, rtol=1e-9, atol=1e-10)
    np.testing.assert_allclose(np.asarray(ours.covs), expected.covs, rtol=1e-8, atol=1e-10)
    np.testing.assert_allclose(ours.log_likelihood, expected.log_likelihood, rtol=1e-9)
    np.testing.assert_allclose(np.asarray(ours.nis), expected.nis, rtol=1e-7, atol=1e-10)

    smoothed, expected_smoothed = ukf.smooth(ours), kf.smooth(expected)
    np.testing.assert_allclose(
        np.asarray(smoothed.means), expected_smoothed.means, rtol=1e-8, atol=1e-10
    )
    np.testing.assert_allclose(
        np.asarray(smoothed.covs), expected_smoothed.covs, rtol=1e-7, atol=1e-10
    )


@pytest.mark.parametrize(("mu", "sigma"), [(0.0, 1.0), (2.0, 0.5), (-3.0, 4.0)])
def test_unscented_transform_matches_gaussian_moments_of_square(mu: float, sigma: float) -> None:
    # For x ~ N(mu, sigma^2): E[x^2] = mu^2 + sigma^2 and Var[x^2] = 4 mu^2 sigma^2 + 2 sigma^4.
    # With n = 1, kappa = 3 - n, alpha = 1, beta = 0 the sigma points match the Gaussian's
    # moments up to fourth order, so both are exact.
    w = numpy_backend.sigma_weights(1, alpha=1.0, beta=0.0, kappa=2.0, dtype=np.dtype(np.float64))
    mean, var, _ = numpy_backend.ukf_predict(
        np.array([mu]), np.array([[sigma**2]]), np.zeros((1, 1)), 0.0, lambda x, dt: x**2, w
    )
    np.testing.assert_allclose(mean, [mu**2 + sigma**2], rtol=1e-12)
    np.testing.assert_allclose(var, [[4 * mu**2 * sigma**2 + 2 * sigma**4]], rtol=1e-12)


def test_step_by_step_matches_batch(cv_scenario: LinearScenario) -> None:
    sc = cv_scenario
    ukf = linear_ukf(sc, **GUIDE_SIGMA_PARAMS)
    res = ukf.filter(sc.zs[:50], dt=0.1)
    for k, z in enumerate(sc.zs[:50]):
        ukf.predict(dt=0.1)
        np.testing.assert_array_equal(res.predicted_means[k], ukf.x)
        ukf.update(z)
        np.testing.assert_array_equal(res.means[k], ukf.x)
        np.testing.assert_array_equal(res.covs[k], ukf.P)


@pytest.mark.parametrize("backend", BACKENDS)
def test_float32_stays_float32(cv_scenario: LinearScenario, backend: Backend) -> None:
    skip_without_jax(backend)
    sc = cv_scenario
    F, H = sc.F.astype(np.float32), sc.H.astype(np.float32)
    ukf = UnscentedKalmanFilter(
        f=lambda x, dt: F @ x,
        h=lambda x: H @ x,
        **{k: v.astype(np.float32) for k, v in sc.params.items() if k in ("Q", "R", "x0", "P0")},
    )
    res = ukf.filter(sc.zs.astype(np.float32), dt=0.1, backend=backend)
    assert res.covs.dtype == np.float32
    assert ukf.smooth(res).covs.dtype == np.float32


def test_rejects_degenerate_sigma_point_parameters(cv_scenario: LinearScenario) -> None:
    with pytest.raises(ValueError, match="n \\+ lambda"):
        linear_ukf(cv_scenario, alpha=1.0, kappa=-4.0)
