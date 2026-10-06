from typing import Any

import numpy as np
import pytest

from kalman_py import UnscentedKalmanFilter
from kalman_py._common import Backend

pytest.importorskip("jax")

from tests.nonlinear_scenarios import (
    cv_f,
    cv_f_vectorized,
    make_range_bearing,
    range_bearing_h,
    range_bearing_h_vectorized,
    wrap_bearing_residual,
    wrap_bearing_residual_vectorized,
)


@pytest.mark.parametrize("backend", ["numpy", "jax"])
@pytest.mark.parametrize("square_root", [False, True], ids=["standard", "square-root"])
@pytest.mark.parametrize(
    "params", [{}, {"alpha": 0.1, "beta": 2.0, "kappa": -1.0}], ids=["default", "guide"]
)
def test_vectorized_models_match_per_point_models(
    backend: Backend, square_root: bool, params: dict[str, float]
) -> None:
    sc = make_range_bearing(steps=120)
    common: dict[str, Any] = {
        "Q": sc.Q,
        "R": sc.R,
        "x0": sc.x0,
        "P0": sc.P0,
        "square_root": square_root,
        **params,
    }
    per_point = UnscentedKalmanFilter(
        f=cv_f, h=range_bearing_h, residual_z=wrap_bearing_residual, **common
    )
    vectorized = UnscentedKalmanFilter(
        f=cv_f_vectorized,
        h=range_bearing_h_vectorized,
        residual_z=wrap_bearing_residual_vectorized,
        vectorized=True,
        **common,
    )
    expected = per_point.filter(sc.zs, dt=1.0, backend=backend)
    actual = vectorized.filter(sc.zs, dt=1.0, backend=backend)
    np.testing.assert_allclose(np.asarray(actual.means), np.asarray(expected.means), rtol=1e-12)
    np.testing.assert_allclose(
        np.asarray(actual.covs), np.asarray(expected.covs), rtol=1e-11, atol=1e-14
    )


@pytest.mark.parametrize("backend", ["numpy", "jax"])
@pytest.mark.parametrize("params", [{}, {"alpha": 0.1, "beta": 2.0, "kappa": -1.0}])
def test_square_root_matches_standard_form_on_nonlinear_problem(
    backend: Backend, params: dict[str, float]
) -> None:
    sc = make_range_bearing(steps=150)  # crosses the +-pi bearing boundary at step 108
    kwargs: dict[str, Any] = {
        "f": cv_f_vectorized,
        "h": range_bearing_h_vectorized,
        "residual_z": wrap_bearing_residual_vectorized,
        "vectorized": True,
        "Q": sc.Q,
        "R": sc.R,
        "x0": sc.x0,
        "P0": sc.P0,
        **params,
    }
    a = UnscentedKalmanFilter(**kwargs).filter(sc.zs, dt=1.0, backend=backend)
    b = UnscentedKalmanFilter(**kwargs, square_root=True).filter(sc.zs, dt=1.0, backend=backend)
    np.testing.assert_allclose(np.asarray(b.means), np.asarray(a.means), rtol=1e-9, atol=1e-9)
    np.testing.assert_allclose(np.asarray(b.covs), np.asarray(a.covs), rtol=1e-8, atol=1e-11)
    np.testing.assert_allclose(b.log_likelihood, a.log_likelihood, rtol=1e-9)
