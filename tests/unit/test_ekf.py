import sys
from typing import Any

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st

import kalman_py.backends
from kalman_py import ExtendedKalmanFilter, KalmanFilter
from tests.scenarios import LinearScenario

jax = pytest.importorskip("jax")
jnp = pytest.importorskip("jax.numpy")
jax_backend = pytest.importorskip("kalman_py.backends.jax_backend")

from tests.nonlinear_scenarios import (
    cv_f,
    cv_f_jacobian,
    make_range_bearing,
    range_bearing_h,
    range_bearing_h_jacobian,
    wrap_bearing_residual,
)

BACKENDS = ["numpy", "jax"]
# Built once: each jacobian() call creates a new jitted function, which would recompile.
RANGE_BEARING_AUTODIFF_JACOBIAN = jax_backend.jacobian(range_bearing_h)


def linear_ekf(sc: LinearScenario, **kwargs: Any) -> ExtendedKalmanFilter:
    F, H = sc.F, sc.H
    return ExtendedKalmanFilter(
        f=lambda x, dt: F @ x, h=lambda x: H @ x, Q=sc.Q, R=sc.R, x0=sc.x0, P0=sc.P0, **kwargs
    )


def range_bearing_ekf(sc: LinearScenario, **kwargs: Any) -> ExtendedKalmanFilter:
    return ExtendedKalmanFilter(
        f=cv_f,
        h=range_bearing_h,
        Q=sc.Q,
        R=sc.R,
        x0=sc.x0,
        P0=sc.P0,
        residual_z=wrap_bearing_residual,
        **kwargs,
    )


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("jacobians", ["autodiff", "supplied"])
def test_linear_model_reduces_to_kalman_filter(
    cv_scenario: LinearScenario, backend: str, jacobians: str
) -> None:
    sc = cv_scenario
    kwargs: dict[str, Any] = {}
    if jacobians == "supplied":
        kwargs = {"jac_f": lambda x, dt: sc.F, "jac_h": lambda x: sc.H}
    ekf = linear_ekf(sc, **kwargs)
    kf = KalmanFilter(**sc.params)

    ours = ekf.filter(sc.zs, dt=0.1, backend=backend)  # type: ignore[call-overload]
    expected = kf.filter(sc.zs)
    np.testing.assert_allclose(np.asarray(ours.means), expected.means, rtol=1e-10, atol=1e-12)
    np.testing.assert_allclose(np.asarray(ours.covs), expected.covs, rtol=1e-10, atol=1e-12)
    np.testing.assert_allclose(ours.log_likelihood, expected.log_likelihood, rtol=1e-10)

    smoothed, expected_smoothed = ekf.smooth(ours), kf.smooth(expected)
    np.testing.assert_allclose(
        np.asarray(smoothed.means), expected_smoothed.means, rtol=1e-9, atol=1e-12
    )
    np.testing.assert_allclose(
        np.asarray(smoothed.covs), expected_smoothed.covs, rtol=1e-9, atol=1e-12
    )


def test_numpy_and_jax_backends_agree_on_nonlinear_problem() -> None:
    sc = make_range_bearing()
    ekf = range_bearing_ekf(sc)
    a, b = ekf.filter(sc.zs, dt=1.0), ekf.filter(sc.zs, dt=1.0, backend="jax")
    np.testing.assert_allclose(np.asarray(b.means), a.means, rtol=1e-9, atol=1e-9)
    np.testing.assert_allclose(np.asarray(b.covs), a.covs, rtol=1e-9, atol=1e-12)
    np.testing.assert_allclose(
        np.asarray(ekf.smooth(b).means), ekf.smooth(a).means, rtol=1e-9, atol=1e-9
    )


def test_supplied_and_autodiff_jacobians_agree() -> None:
    sc = make_range_bearing()
    auto = range_bearing_ekf(sc).filter(sc.zs, dt=1.0)
    supplied = range_bearing_ekf(sc, jac_f=cv_f_jacobian, jac_h=range_bearing_h_jacobian).filter(
        sc.zs, dt=1.0
    )
    np.testing.assert_allclose(supplied.means, auto.means, rtol=1e-10, atol=1e-10)
    np.testing.assert_allclose(supplied.covs, auto.covs, rtol=1e-10, atol=1e-12)


@given(x=st.lists(st.floats(-100, 100), min_size=4, max_size=4))
def test_autodiff_jacobian_matches_analytic(x: list[float]) -> None:
    xa = jnp.array(x)
    if jnp.hypot(xa[0], xa[1]) < 1e-3:  # range-bearing is undefined at the origin
        return
    np.testing.assert_allclose(
        RANGE_BEARING_AUTODIFF_JACOBIAN(xa),
        range_bearing_h_jacobian(xa),
        rtol=1e-10,
        atol=1e-10,
    )


def test_bearing_residual_wrapping_is_applied() -> None:
    sc = make_range_bearing()
    crossings = np.abs(np.diff(sc.zs[:, 1])) > np.pi
    assert crossings.any(), "scenario must cross the +-pi bearing boundary"

    def position_rmse(ekf: ExtendedKalmanFilter) -> float:
        means = ekf.filter(sc.zs, dt=1.0).means
        return float(np.sqrt(np.mean((means[:, :2] - sc.truth[:, :2]) ** 2)))

    unwrapped = ExtendedKalmanFilter(f=cv_f, h=range_bearing_h, Q=sc.Q, R=sc.R, x0=sc.x0, P0=sc.P0)
    assert position_rmse(range_bearing_ekf(sc)) < 5.0
    assert position_rmse(unwrapped) > 10 * position_rmse(range_bearing_ekf(sc))


def test_batch_matches_step_by_step_with_irregular_dt() -> None:
    sc = make_range_bearing(steps=40)
    dts = np.random.default_rng(0).uniform(0.5, 1.5, size=40)
    ekf = range_bearing_ekf(sc)
    res = ekf.filter(sc.zs, dt=dts)
    for k, z in enumerate(sc.zs):
        ekf.predict(dt=dts[k])
        np.testing.assert_allclose(res.predicted_means[k], ekf.x, rtol=1e-12)
        ekf.update(z)
        np.testing.assert_allclose(res.means[k], ekf.x, rtol=1e-12)
        np.testing.assert_allclose(res.covs[k], ekf.P, rtol=1e-12, atol=1e-15)

    jax_res = ekf.filter(sc.zs, dt=dts, backend="jax")
    np.testing.assert_allclose(np.asarray(jax_res.means), res.means, rtol=1e-9, atol=1e-9)


def test_rejects_dt_of_wrong_length() -> None:
    sc = make_range_bearing(steps=10)
    with pytest.raises(ValueError, match="dt must be a scalar or have shape"):
        range_bearing_ekf(sc).filter(sc.zs, dt=np.ones(9))


@pytest.mark.parametrize("backend", BACKENDS)
def test_float32_stays_float32(backend: str) -> None:
    sc = make_range_bearing(steps=20)
    ekf = ExtendedKalmanFilter(
        f=cv_f,
        h=range_bearing_h,
        Q=sc.Q.astype(np.float32),
        R=sc.R.astype(np.float32),
        x0=sc.x0.astype(np.float32),
        P0=sc.P0.astype(np.float32),
        residual_z=wrap_bearing_residual,
    )
    res = ekf.filter(sc.zs.astype(np.float32), dt=1.0, backend=backend)  # type: ignore[call-overload]
    assert res.covs.dtype == np.float32
    assert ekf.smooth(res).covs.dtype == np.float32


def test_jax_compiles_once_for_shared_model_functions() -> None:
    sc = make_range_bearing(steps=30)
    range_bearing_ekf(sc).filter(sc.zs, dt=1.0, backend="jax")
    compiled = jax_backend._ekf_filter._cache_size()
    other = make_range_bearing(seed=5, steps=30)
    range_bearing_ekf(other).filter(other.zs, dt=1.0, backend="jax")
    assert jax_backend._ekf_filter._cache_size() == compiled


def test_missing_jacobians_without_jax_gives_install_hint(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setitem(sys.modules, "kalman_py.backends.jax_backend", None)
    monkeypatch.delattr(kalman_py.backends, "jax_backend")
    with pytest.raises(ImportError, match=r"pass jac_f and jac_h, or install JAX"):
        ExtendedKalmanFilter(
            f=lambda x, dt: x, h=lambda x: x, Q=np.eye(2), R=np.eye(2), x0=np.zeros(2), P0=np.eye(2)
        )


def test_supplied_jacobians_work_without_jax(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "kalman_py.backends.jax_backend", None)
    monkeypatch.delattr(kalman_py.backends, "jax_backend")
    ekf = ExtendedKalmanFilter(
        f=lambda x, dt: x,
        h=lambda x: x,
        Q=np.eye(2),
        R=np.eye(2),
        x0=np.zeros(2),
        P0=np.eye(2),
        jac_f=lambda x, dt: np.eye(2),
        jac_h=lambda x: np.eye(2),
    )
    assert ekf.smooth(ekf.filter(np.ones((5, 2)), dt=1.0)).means.shape == (5, 2)


@pytest.mark.parametrize("backend", BACKENDS)
def test_square_root_form_matches_joseph_form(backend: str) -> None:
    sc = make_range_bearing()
    expected = range_bearing_ekf(sc).filter(sc.zs, dt=1.0)
    sqrt_ekf = range_bearing_ekf(sc, square_root=True)
    actual = sqrt_ekf.filter(sc.zs, dt=1.0, backend=backend)  # type: ignore[call-overload]
    np.testing.assert_allclose(np.asarray(actual.means), expected.means, rtol=1e-9, atol=1e-9)
    np.testing.assert_allclose(np.asarray(actual.covs), expected.covs, rtol=1e-8, atol=1e-12)
    np.testing.assert_allclose(actual.log_likelihood, expected.log_likelihood, rtol=1e-10)
    np.testing.assert_allclose(
        np.asarray(sqrt_ekf.smooth(actual).means),
        range_bearing_ekf(sc).smooth(expected).means,
        rtol=1e-9,
        atol=1e-9,
    )


def test_square_root_step_by_step_matches_batch() -> None:
    sc = make_range_bearing(steps=40)
    ekf = range_bearing_ekf(sc, square_root=True)
    res = ekf.filter(sc.zs, dt=1.0)
    for k, z in enumerate(sc.zs):
        ekf.predict(dt=1.0)
        ekf.update(z)
        np.testing.assert_allclose(res.means[k], ekf.x, rtol=1e-12)
        np.testing.assert_allclose(res.covs[k], ekf.P, rtol=1e-12, atol=1e-15)
