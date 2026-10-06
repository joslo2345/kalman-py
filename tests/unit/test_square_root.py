from dataclasses import fields
from pathlib import Path

import numpy as np
import pytest

from kalman_py import FilterResult, KalmanFilter
from kalman_py._common import Backend
from tests.scenarios import LinearScenario, cv_model_2d, simulate

BACKENDS: list[Backend] = ["numpy", "jax"]


def skip_without_jax(backend: Backend) -> None:
    if backend == "jax":
        pytest.importorskip("jax")


def assert_results_close(actual: FilterResult, expected: FilterResult, rtol: float) -> None:  # type: ignore[type-arg]
    for field in fields(FilterResult):
        np.testing.assert_allclose(
            np.asarray(getattr(actual, field.name)),
            np.asarray(getattr(expected, field.name)),
            rtol=rtol,
            atol=1e-12,
            err_msg=field.name,
        )


@pytest.mark.parametrize("backend", BACKENDS)
def test_matches_joseph_form(cv_scenario: LinearScenario, backend: Backend) -> None:
    skip_without_jax(backend)
    sc = cv_scenario
    joseph = KalmanFilter(**sc.params)
    sqrt = KalmanFilter(**sc.params, square_root=True)
    expected = joseph.filter(sc.zs)
    actual = sqrt.filter(sc.zs, backend=backend)
    assert_results_close(actual, expected, rtol=1e-9)

    smoothed, expected_smoothed = sqrt.smooth(actual), joseph.smooth(expected)
    np.testing.assert_allclose(
        np.asarray(smoothed.means), expected_smoothed.means, rtol=1e-9, atol=1e-12
    )
    np.testing.assert_allclose(
        np.asarray(smoothed.covs), expected_smoothed.covs, rtol=1e-9, atol=1e-12
    )


def test_step_by_step_matches_batch(cv_scenario: LinearScenario) -> None:
    sc = cv_scenario
    kf = KalmanFilter(**sc.params, square_root=True)
    res = kf.filter(sc.zs)
    for k, z in enumerate(sc.zs):
        kf.predict()
        np.testing.assert_allclose(res.predicted_covs[k], kf.P, rtol=1e-12, atol=1e-15)
        kf.update(z)
        np.testing.assert_allclose(res.means[k], kf.x, rtol=1e-12, atol=1e-15)
        np.testing.assert_allclose(res.covs[k], kf.P, rtol=1e-12, atol=1e-15)


@pytest.mark.parametrize("backend", BACKENDS)
def test_handles_singular_process_noise(backend: Backend) -> None:
    # q = 0 makes Q the zero matrix, which has no Cholesky factor.
    skip_without_jax(backend)
    F, Q = cv_model_2d(dt=0.1, q=0.0)
    H = np.array([[1.0, 0, 0, 0], [0, 1.0, 0, 0]])
    sc = simulate(F, H, Q, 0.3 * np.eye(2), np.zeros(4), np.eye(4), steps=100, seed=0)
    expected = KalmanFilter(**sc.params).filter(sc.zs)
    actual = KalmanFilter(**sc.params, square_root=True).filter(sc.zs, backend=backend)
    assert_results_close(actual, expected, rtol=1e-8)


def test_rejects_indefinite_noise_covariance(cv_scenario: LinearScenario) -> None:
    with pytest.raises(ValueError, match="R must be positive semi-definite"):
        KalmanFilter(**{**cv_scenario.params, "R": np.diag([1.0, -1.0])}, square_root=True)


@pytest.mark.parametrize("backend", BACKENDS)
def test_float32_stays_float32(cv_scenario: LinearScenario, backend: Backend) -> None:
    skip_without_jax(backend)
    params = {k: v.astype(np.float32) for k, v in cv_scenario.params.items()}
    kf = KalmanFilter(**params, square_root=True)
    res = kf.filter(cv_scenario.zs.astype(np.float32), backend=backend)
    assert res.covs.dtype == np.float32
    assert res.means.dtype == np.float32


ILL_CONDITIONED = Path(__file__).resolve().parents[1] / "data" / "ill_conditioned_r1e-9.npz"


@pytest.mark.parametrize("backend", BACKENDS)
def test_stays_positive_semidefinite_on_ill_conditioned_problem(backend: Backend) -> None:
    # Found by the stability property test (seed 3606, R ~ 1e-9). The problem is stored, since
    # NumPy versions draw different random numbers. On macOS (Accelerate) the NumPy Joseph form
    # turns indefinite here (min eigenvalue ~ -70 eps relative to the max at 16 of 2000 steps);
    # on Linux and Windows (OpenBLAS) rounding differs and it may not. The square-root form must
    # stay positive semi-definite everywhere.
    skip_without_jax(backend)
    data = np.load(ILL_CONDITIONED)
    params = {k: data[k] for k in ("F", "H", "Q", "R", "x0", "P0")}
    tol = 10 * 4 * np.finfo(np.float64).eps

    def worst_relative_eigenvalue(P: np.ndarray) -> float:
        eig = np.linalg.eigvalsh(P)
        return float((eig.min(axis=1) / eig.max(axis=1)).min())

    covs = np.asarray(
        KalmanFilter(**params, square_root=True).filter(data["zs"], backend=backend).covs
    )
    np.testing.assert_array_equal(covs, np.swapaxes(covs, -1, -2))
    assert worst_relative_eigenvalue(covs) >= -tol
