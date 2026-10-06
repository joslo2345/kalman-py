import sys
from dataclasses import fields

import numpy as np
import pytest

import kalman_py.backends
from kalman_py import FilterResult, KalmanFilter
from tests.scenarios import LinearScenario, make_cv_2d, simulate

jax = pytest.importorskip("jax")
jnp = pytest.importorskip("jax.numpy")
jax_backend = pytest.importorskip("kalman_py.backends.jax_backend")


def test_filter_matches_numpy_backend(cv_scenario: LinearScenario) -> None:
    kf = KalmanFilter(**cv_scenario.params)
    expected = kf.filter(cv_scenario.zs)
    actual = kf.filter(cv_scenario.zs, backend="jax")
    for field in fields(FilterResult):
        np.testing.assert_allclose(
            np.asarray(getattr(actual, field.name)),
            getattr(expected, field.name),
            rtol=1e-10,
            atol=1e-12,
            err_msg=field.name,
        )


@pytest.mark.parametrize(("n", "m"), [(3, 1), (20, 10)], ids=["fused", "lapack-fallback"])
def test_matches_numpy_backend_on_both_kernel_paths(n: int, m: int) -> None:
    # Small shapes use fused products and an unrolled Cholesky; large ones use @ and LAPACK.
    rng = np.random.default_rng(n)
    F = np.eye(n) + 0.05 * rng.normal(size=(n, n))
    H = rng.normal(size=(m, n))
    A, B = rng.normal(size=(n, n)), rng.normal(size=(m, m))
    sc = simulate(F, H, 0.1 * A @ A.T, B @ B.T + np.eye(m), np.zeros(n), np.eye(n), 50, seed=n)
    kf = KalmanFilter(**sc.params)
    expected = kf.smooth(kf.filter(sc.zs))
    filtered = kf.filter(sc.zs, backend="jax")
    actual = kf.smooth(filtered)
    np.testing.assert_allclose(filtered.log_likelihood, kf.filter(sc.zs).log_likelihood, rtol=1e-10)
    np.testing.assert_allclose(np.asarray(actual.means), expected.means, rtol=1e-8, atol=1e-10)
    np.testing.assert_allclose(np.asarray(actual.covs), expected.covs, rtol=1e-8, atol=1e-10)


def test_smoother_matches_numpy_backend(cv_scenario: LinearScenario) -> None:
    kf = KalmanFilter(**cv_scenario.params)
    expected = kf.smooth(kf.filter(cv_scenario.zs))
    actual = kf.smooth(kf.filter(cv_scenario.zs, backend="jax"))
    np.testing.assert_allclose(np.asarray(actual.means), expected.means, rtol=1e-10, atol=1e-12)
    np.testing.assert_allclose(np.asarray(actual.covs), expected.covs, rtol=1e-10, atol=1e-12)


def test_results_are_jax_arrays(cv_scenario: LinearScenario) -> None:
    kf = KalmanFilter(**cv_scenario.params)
    filtered = kf.filter(jnp.asarray(cv_scenario.zs), backend="jax")
    smoothed = kf.smooth(filtered)
    assert isinstance(filtered.means, jax.Array)
    assert isinstance(filtered.log_likelihood, jax.Array)
    assert isinstance(smoothed.covs, jax.Array)


def test_float32_stays_float32(cv_scenario: LinearScenario) -> None:
    params = {k: v.astype(np.float32) for k, v in cv_scenario.params.items()}
    kf = KalmanFilter(**params)
    filtered = kf.filter(cv_scenario.zs.astype(np.float32), backend="jax")
    smoothed = kf.smooth(filtered)
    assert filtered.covs.dtype == jnp.float32
    assert smoothed.covs.dtype == jnp.float32


def test_compiled_code_is_shared_across_filter_instances() -> None:
    first, second = make_cv_2d(seed=10, steps=50), make_cv_2d(seed=11, steps=50)
    KalmanFilter(**first.params).filter(first.zs, backend="jax")
    compiled = jax_backend._kalman_filter._cache_size()

    other_noise = {**second.params, "R": 3.0 * second.R}
    KalmanFilter(**other_noise).filter(second.zs, backend="jax")
    assert jax_backend._kalman_filter._cache_size() == compiled


def test_scalar_sensor_accepts_one_dimensional_measurements() -> None:
    kf = KalmanFilter(F=[[1.0]], H=[[1.0]], Q=[[0.1]], R=[[1.0]], x0=[0.0], P0=[[1.0]])
    filtered = kf.filter(jnp.array([1.0, 2.0, 3.0]), backend="jax")
    np.testing.assert_allclose(
        np.asarray(filtered.means), kf.filter([1.0, 2.0, 3.0]).means, rtol=1e-12
    )


def test_single_step_smoothing(cv_scenario: LinearScenario) -> None:
    kf = KalmanFilter(**cv_scenario.params)
    filtered = kf.filter(cv_scenario.zs[:1], backend="jax")
    np.testing.assert_array_equal(kf.smooth(filtered).means, filtered.means)


def test_rejects_wrong_measurement_shape(cv_scenario: LinearScenario) -> None:
    with pytest.raises(ValueError, match="zs must have shape"):
        KalmanFilter(**cv_scenario.params).filter(jnp.zeros((10, 3)), backend="jax")


def test_missing_jax_gives_install_hint(
    cv_scenario: LinearScenario, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Simulate an install without JAX: the backend module can't be imported.
    monkeypatch.setitem(sys.modules, "kalman_py.backends.jax_backend", None)
    monkeypatch.delattr(kalman_py.backends, "jax_backend")
    with pytest.raises(ImportError, match=r"pip install kalman-py\[jax\]"):
        KalmanFilter(**cv_scenario.params).filter(cv_scenario.zs, backend="jax")
