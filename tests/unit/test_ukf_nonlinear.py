from typing import Any

import numpy as np
import pytest

from kalman_py import UnscentedKalmanFilter
from kalman_py.backends import numpy_backend
from tests.scenarios import LinearScenario

jax = pytest.importorskip("jax")
jax_backend = pytest.importorskip("kalman_py.backends.jax_backend")

from tests.nonlinear_scenarios import (
    SENSOR_R,
    cv_f,
    make_range_bearing,
    range_bearing_h,
    wrap_bearing_residual,
)


def range_bearing_ukf(sc: LinearScenario, **kwargs: object) -> UnscentedKalmanFilter:
    return UnscentedKalmanFilter(
        f=cv_f,
        h=range_bearing_h,
        Q=sc.Q,
        R=sc.R,
        x0=sc.x0,
        P0=sc.P0,
        **kwargs,  # type: ignore[arg-type]
    )


def position_rmse(means: Any, sc: LinearScenario) -> float:
    return float(np.sqrt(np.mean((np.asarray(means)[:, :2] - sc.truth[:, :2]) ** 2)))


def test_numpy_and_jax_backends_agree() -> None:
    sc = make_range_bearing(steps=150)
    ukf = range_bearing_ukf(sc, residual_z=wrap_bearing_residual)
    a, b = ukf.filter(sc.zs, dt=1.0), ukf.filter(sc.zs, dt=1.0, backend="jax")
    np.testing.assert_allclose(np.asarray(b.means), a.means, rtol=1e-9, atol=1e-9)
    np.testing.assert_allclose(np.asarray(b.covs), a.covs, rtol=1e-8, atol=1e-11)
    np.testing.assert_allclose(b.log_likelihood, a.log_likelihood, rtol=1e-9)
    np.testing.assert_allclose(
        np.asarray(ukf.smooth(b).means), ukf.smooth(a).means, rtol=1e-9, atol=1e-9
    )


def test_bearing_wrap_is_handled_in_mean_and_residual() -> None:
    # Target just above the negative x-axis (bearing ~ +pi). With P's spread in y the sigma
    # points straddle the +-pi boundary, and the measurement lands just across it.
    x = np.array([-100.0, 0.05, 0.0, 0.0])
    P = np.eye(4)
    z = np.array([100.0, -np.pi + 0.001])
    w = numpy_backend.sigma_weights(4, 1.0, 2.0, 0.0, np.dtype(np.float64))

    wrapped = numpy_backend.ukf_update(x, P, z, SENSOR_R, range_bearing_h, wrap_bearing_residual, w)
    # Bearing spread is ~ position spread / range ~ 0.01 rad, so its variance is ~1e-4.
    assert wrapped.S[1, 1] < 1e-3
    assert abs(wrapped.y[1]) < 0.01
    assert np.linalg.norm(wrapped.x[:2] - x[:2]) < 1.0

    # Plain subtraction sees bearings of +pi and -pi among the sigma points: the predicted
    # bearing variance explodes (the UKF then ignores the bearing rather than diverging).
    plain = numpy_backend.ukf_update(x, P, z, SENSOR_R, range_bearing_h, np.subtract, w)
    assert plain.S[1, 1] > 1.0


def test_wrapped_filter_tracks_through_the_crossing() -> None:
    sc = make_range_bearing()
    assert (np.abs(np.diff(sc.zs[:, 1])) > np.pi).any(), "scenario must cross +-pi"
    ukf = range_bearing_ukf(sc, residual_z=wrap_bearing_residual)
    assert position_rmse(ukf.filter(sc.zs, dt=1.0, backend="jax").means, sc) < 5.0


def test_smoothing_improves_accuracy() -> None:
    sc = make_range_bearing()
    ukf = range_bearing_ukf(sc, residual_z=wrap_bearing_residual)
    filtered = ukf.filter(sc.zs, dt=1.0, backend="jax")
    assert position_rmse(ukf.smooth(filtered).means, sc) < position_rmse(filtered.means, sc)


def test_jax_compiles_once_for_shared_model_functions() -> None:
    sc, other = make_range_bearing(steps=30), make_range_bearing(seed=5, steps=30)
    range_bearing_ukf(sc, residual_z=wrap_bearing_residual).filter(sc.zs, dt=1.0, backend="jax")
    compiled = jax_backend._ukf_filter._cache_size()
    range_bearing_ukf(other, residual_z=wrap_bearing_residual).filter(
        other.zs, dt=1.0, backend="jax"
    )
    assert jax_backend._ukf_filter._cache_size() == compiled
