import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from kalman_py import KalmanFilter
from kalman_py._common import Backend
from tests.scenarios import make_random_linear

JOSEPH_LIMIT = pytest.mark.xfail(
    reason="Joseph form can lose definiteness under rounding when R is tiny (e.g. seed=3606, "
    "R ~ 1e-9 on NumPy); square_root=True is the guaranteed mode",
    strict=False,
)


@pytest.mark.slow
@pytest.mark.parametrize(
    ("backend", "square_root"),
    [
        pytest.param("numpy", True, id="numpy-sqrt"),
        pytest.param("jax", True, id="jax-sqrt"),
        pytest.param("numpy", False, id="numpy-joseph", marks=JOSEPH_LIMIT),
        pytest.param("jax", False, id="jax-joseph", marks=JOSEPH_LIMIT),
    ],
)
@given(seed=st.integers(0, 2**32 - 1), log_r=st.floats(-12, 2))
@settings(max_examples=200, deadline=None)
def test_covariance_stays_symmetric_positive_semidefinite(
    backend: Backend, square_root: bool, seed: int, log_r: float
) -> None:
    if backend == "jax":
        pytest.importorskip("jax")
    sc = make_random_linear(seed=seed, r_scale=10.0**log_r, steps=2000)
    kf = KalmanFilter(**sc.params, square_root=square_root)
    P = np.asarray(kf.filter(sc.zs, backend=backend).covs)

    np.testing.assert_array_equal(P, np.swapaxes(P, -1, -2))
    # Tiny R drives cond(P) to ~1/eps, where eigvalsh can't resolve the smallest eigenvalue's
    # sign (its error is ~eps * max eigenvalue). So require PSD to working precision.
    eig = np.linalg.eigvalsh(P)
    tol = 10 * sc.n * np.finfo(P.dtype).eps
    assert (eig.min(axis=1) >= -tol * eig.max(axis=1)).all()
