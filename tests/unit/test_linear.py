import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st

from kalman_py import KalmanFilter
from tests.scenarios import LinearScenario

stats = pytest.importorskip("scipy.stats")


@given(
    p=st.floats(1e-3, 1e3),
    q=st.floats(1e-3, 1e3),
    r=st.floats(1e-3, 1e3),
    z=st.floats(-1e3, 1e3),
)
def test_scalar_random_walk_matches_closed_form(p: float, q: float, r: float, z: float) -> None:
    kf = KalmanFilter(F=[[1.0]], H=[[1.0]], Q=[[q]], R=[[r]], x0=[0.0], P0=[[p]])
    kf.predict()
    kf.update(z)

    gain = (p + q) / (p + q + r)
    np.testing.assert_allclose(kf.x, [gain * z], rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(kf.P, [[(1 - gain) * (p + q)]], rtol=1e-9)


def test_batch_matches_step_by_step(cv_scenario: LinearScenario) -> None:
    sc = cv_scenario
    kf = KalmanFilter(**sc.params)
    res = kf.filter(sc.zs)

    for k, z in enumerate(sc.zs):
        kf.predict()
        np.testing.assert_array_equal(res.predicted_means[k], kf.x)
        np.testing.assert_array_equal(res.predicted_covs[k], kf.P)
        kf.update(z)
        np.testing.assert_array_equal(res.means[k], kf.x)
        np.testing.assert_array_equal(res.covs[k], kf.P)


def test_filter_leaves_step_state_untouched(cv_scenario: LinearScenario) -> None:
    kf = KalmanFilter(**cv_scenario.params)
    kf.filter(cv_scenario.zs)
    np.testing.assert_array_equal(kf.x, cv_scenario.x0)
    np.testing.assert_array_equal(kf.P, cv_scenario.P0)


def test_nis_and_log_likelihood_match_gaussian_innovations(cv_scenario: LinearScenario) -> None:
    sc = cv_scenario
    res = KalmanFilter(**sc.params).filter(sc.zs)

    expected_ll = 0.0
    for k, z in enumerate(sc.zs):
        mean = sc.H @ res.predicted_means[k]
        S = sc.H @ res.predicted_covs[k] @ sc.H.T + sc.R
        y = z - mean
        np.testing.assert_allclose(res.nis[k], y @ np.linalg.solve(S, y), rtol=1e-10)
        expected_ll += stats.multivariate_normal(mean, S).logpdf(z)
    np.testing.assert_allclose(res.log_likelihood, expected_ll, rtol=1e-10)


def test_float32_inputs_stay_float32(cv_scenario: LinearScenario) -> None:
    params = {k: v.astype(np.float32) for k, v in cv_scenario.params.items()}
    res = KalmanFilter(**params).filter(cv_scenario.zs.astype(np.float32))
    for arr in (res.means, res.covs, res.predicted_means, res.predicted_covs):
        assert arr.dtype == np.float32


def test_integer_inputs_become_float64() -> None:
    kf = KalmanFilter(F=[[1]], H=[[1]], Q=[[1]], R=[[1]], x0=[0], P0=[[1]])
    assert kf.P.dtype == np.float64


def test_one_dimensional_measurements_accepted_for_scalar_sensor() -> None:
    kf = KalmanFilter(F=[[1.0]], H=[[1.0]], Q=[[0.1]], R=[[1.0]], x0=[0.0], P0=[[1.0]])
    assert kf.filter([1.0, 2.0, 3.0]).means.shape == (3, 1)


@pytest.mark.parametrize(
    ("override", "message"),
    [
        ({"F": np.eye(3)}, "F must have shape"),
        ({"R": np.eye(3)}, "R must have shape"),
        ({"x0": np.zeros(3)}, "x0 must have shape"),
        ({"H": np.zeros(4)}, "H must be 2-D"),
    ],
)
def test_rejects_inconsistent_shapes(
    cv_scenario: LinearScenario, override: dict[str, np.ndarray], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        KalmanFilter(**{**cv_scenario.params, **override})


def test_rejects_wrong_measurement_shape(cv_scenario: LinearScenario) -> None:
    kf = KalmanFilter(**cv_scenario.params)
    with pytest.raises(ValueError, match="zs must have shape"):
        kf.filter(np.zeros((10, 3)))
    with pytest.raises(ValueError, match="z must have shape"):
        kf.update(np.zeros(3))


def test_unknown_backend_raises(cv_scenario: LinearScenario) -> None:
    with pytest.raises(ValueError, match="unknown backend"):
        KalmanFilter(**cv_scenario.params).filter(cv_scenario.zs, backend="torch")  # type: ignore[call-overload]
