"""UKF square-root form, invalid-covariance errors on both backends, and vectorized models."""

from typing import Any

import numpy as np
import pytest

from kalman_py import CovarianceDowndateError, KalmanFilter, UnscentedKalmanFilter
from kalman_py._common import Backend
from kalman_py.backends import numpy_backend
from tests.scenarios import LinearScenario

BACKENDS: list[Backend] = ["numpy", "jax"]
SIGMA_PARAMS = [{}, {"alpha": 0.1, "beta": 2.0, "kappa": -1.0}]  # default, negative center weight
SIGMA_IDS = ["default", "negative-center-weight"]


def skip_without_jax(backend: Backend) -> None:
    if backend == "jax":
        pytest.importorskip("jax")


def linear_ukf(sc: LinearScenario, **kwargs: Any) -> UnscentedKalmanFilter:
    F, H = sc.F, sc.H
    return UnscentedKalmanFilter(
        f=lambda x, dt: F @ x, h=lambda x: H @ x, Q=sc.Q, R=sc.R, x0=sc.x0, P0=sc.P0, **kwargs
    )


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("params", SIGMA_PARAMS, ids=SIGMA_IDS)
def test_square_root_matches_kalman_filter_on_linear_model(
    cv_scenario: LinearScenario, backend: Backend, params: dict[str, float]
) -> None:
    skip_without_jax(backend)
    sc = cv_scenario
    ukf = linear_ukf(sc, square_root=True, **params)
    kf = KalmanFilter(**sc.params)
    ours, expected = ukf.filter(sc.zs, dt=0.1, backend=backend), kf.filter(sc.zs)
    np.testing.assert_allclose(np.asarray(ours.means), expected.means, rtol=1e-9, atol=1e-10)
    np.testing.assert_allclose(np.asarray(ours.covs), expected.covs, rtol=1e-8, atol=1e-10)
    np.testing.assert_allclose(ours.log_likelihood, expected.log_likelihood, rtol=1e-9)
    np.testing.assert_allclose(
        np.asarray(ukf.smooth(ours).means), kf.smooth(expected).means, rtol=1e-8, atol=1e-10
    )


def test_square_root_step_by_step_matches_batch(cv_scenario: LinearScenario) -> None:
    sc = cv_scenario
    ukf = linear_ukf(sc, square_root=True)
    res = ukf.filter(sc.zs[:50], dt=0.1)
    for k, z in enumerate(sc.zs[:50]):
        ukf.predict(dt=0.1)
        ukf.update(z)
        np.testing.assert_array_equal(res.means[k], ukf.x)
        np.testing.assert_allclose(res.covs[k], ukf.P, rtol=1e-14, atol=1e-15)


@pytest.mark.parametrize("sign", [1.0, -1.0], ids=["update", "downdate"])
def test_chol_update_matches_refactorization(sign: float) -> None:
    rng = np.random.default_rng(0)
    A = rng.normal(size=(5, 5))
    P = A @ A.T + 5 * np.eye(5)
    v = 0.3 * rng.normal(size=5)
    L = numpy_backend.chol_update(np.linalg.cholesky(P), v, sign)
    np.testing.assert_allclose(L, np.linalg.cholesky(P + sign * np.outer(v, v)), rtol=1e-12)


def test_chol_downdate_that_would_lose_definiteness_raises() -> None:
    with pytest.raises(CovarianceDowndateError):
        numpy_backend.chol_update(np.eye(2), np.array([1.5, 0.0]), -1.0)


def _variance_goes_negative(square_root: bool) -> UnscentedKalmanFilter:
    # For f(x) = x^2 the sigma-point variance estimate is ~2 - 0.5 * 2 * 0.99^2 * ... : with
    # beta = -0.5 the central covariance weight outweighs the rest and it turns negative.
    return UnscentedKalmanFilter(
        f=lambda x, dt: x**2,
        h=lambda x: x,
        Q=[[0.0]],
        R=[[1.0]],
        x0=[0.0],
        P0=[[1.0]],
        alpha=0.1,
        beta=-0.5,
        square_root=square_root,
    )


@pytest.mark.parametrize("backend", BACKENDS)
def test_indefinite_covariance_raises_instead_of_nan(backend: Backend) -> None:
    skip_without_jax(backend)
    with pytest.raises(ValueError, match="state covariance must be positive semi-definite"):
        _variance_goes_negative(square_root=False).filter([[1.0]], dt=1.0, backend=backend)


@pytest.mark.parametrize("backend", BACKENDS)
def test_failed_square_root_downdate_raises(backend: Backend) -> None:
    skip_without_jax(backend)
    with pytest.raises(CovarianceDowndateError, match="downdate failed"):
        _variance_goes_negative(square_root=True).filter([[1.0]], dt=1.0, backend=backend)


@pytest.mark.parametrize("backend", BACKENDS)
def test_rounding_level_indefiniteness_is_tolerated(
    cv_scenario: LinearScenario, backend: Backend
) -> None:
    # A P0 with a -1e-17 eigenvalue has no Cholesky factor; both backends fall back to an
    # eigendecomposition (JAX's Cholesky would otherwise produce NaN) and agree.
    skip_without_jax(backend)
    V = np.linalg.qr(np.random.default_rng(1).normal(size=(4, 4)))[0]
    P0 = V @ np.diag([1.0, 2.0, 0.5, -1e-17]) @ V.T
    sc = cv_scenario
    F, H = sc.F, sc.H
    ukf = UnscentedKalmanFilter(
        f=lambda x, dt: F @ x, h=lambda x: H @ x, Q=sc.Q, R=sc.R, x0=sc.x0, P0=P0
    )
    expected = ukf.filter(sc.zs[:20], dt=0.1)
    actual = ukf.filter(sc.zs[:20], dt=0.1, backend=backend)
    assert np.isfinite(np.asarray(actual.covs)).all()
    np.testing.assert_allclose(np.asarray(actual.means), expected.means, rtol=1e-9, atol=1e-10)


def test_square_root_requires_positive_definite_p0(cv_scenario: LinearScenario) -> None:
    with pytest.raises(ValueError, match="P0 must be positive-definite"):
        UnscentedKalmanFilter(
            f=lambda x, dt: x,
            h=lambda x: x[:2],
            Q=cv_scenario.Q,
            R=cv_scenario.R,
            x0=cv_scenario.x0,
            P0=np.diag([1.0, 1.0, 1.0, 0.0]),
            square_root=True,
        )


def test_vectorized_function_with_wrong_output_shape_is_rejected(
    cv_scenario: LinearScenario,
) -> None:
    sc = cv_scenario
    ukf = UnscentedKalmanFilter(
        f=lambda X, dt: X[0],  # per-point code applied to a stack: wrong shape
        h=lambda X: X[..., :2],
        Q=sc.Q,
        R=sc.R,
        x0=sc.x0,
        P0=sc.P0,
        vectorized=True,
    )
    with pytest.raises(ValueError, match="vectorized model function must map"):
        ukf.filter(sc.zs[:5], dt=0.1)
