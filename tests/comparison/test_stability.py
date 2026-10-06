import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from kalman_py import KalmanFilter
from tests.scenarios import make_random_linear


@pytest.mark.slow
@given(seed=st.integers(0, 2**32 - 1), log_r=st.floats(-12, 2))
@settings(max_examples=200, deadline=None)
def test_covariance_stays_symmetric_positive_semidefinite(seed: int, log_r: float) -> None:
    sc = make_random_linear(seed=seed, r_scale=10.0**log_r, steps=2000)
    P = KalmanFilter(**sc.params).filter(sc.zs).covs

    np.testing.assert_array_equal(P, np.swapaxes(P, -1, -2))
    # Tiny R drives cond(P) to ~1/eps, where eigvalsh can't resolve the smallest eigenvalue's
    # sign (its error is ~eps * max eigenvalue). So require PSD to working precision.
    eig = np.linalg.eigvalsh(P)
    tol = 10 * sc.n * np.finfo(P.dtype).eps
    assert (eig.min(axis=1) >= -tol * eig.max(axis=1)).all()
