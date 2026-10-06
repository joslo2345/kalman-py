import numpy as np
import pytest

from kalman_py import KalmanFilter
from tests.scenarios import LinearScenario, make_cv_2d


def joint_gaussian_posterior(sc: LinearScenario) -> tuple[np.ndarray, np.ndarray]:
    """Exact p(x_1..x_T | z_1..z_T) by conditioning the joint Gaussian of all states and
    measurements; independent of any recursion. Only feasible for short sequences."""
    T, n = len(sc.zs), sc.n
    mean_x = np.empty((T, n))
    cov_x = np.empty((T, n, T, n))  # cov_x[i, :, j, :] = Cov(x_i, x_j)
    x, P = sc.x0, sc.P0
    for k in range(T):
        x, P = sc.F @ x, sc.F @ P @ sc.F.T + sc.Q
        mean_x[k] = x
        cov_x[k, :, k, :] = P
        for j in range(k):  # Cov(x_k, x_j) = F Cov(x_{k-1}, x_j)
            cov_x[k, :, j, :] = sc.F @ cov_x[k - 1, :, j, :]
            cov_x[j, :, k, :] = cov_x[k, :, j, :].T
    Sxx = cov_x.reshape(T * n, T * n)

    Hbig = np.kron(np.eye(T), sc.H)
    Szz = Hbig @ Sxx @ Hbig.T + np.kron(np.eye(T), sc.R)
    Sxz = Sxx @ Hbig.T
    innovation = sc.zs.reshape(-1) - Hbig @ mean_x.reshape(-1)

    post_mean = mean_x.reshape(-1) + Sxz @ np.linalg.solve(Szz, innovation)
    post_cov = Sxx - Sxz @ np.linalg.solve(Szz, Sxz.T)
    covs = post_cov.reshape(T, n, T, n)[np.arange(T), :, np.arange(T), :]
    return post_mean.reshape(T, n), covs


def test_smoother_matches_exact_joint_gaussian_posterior() -> None:
    sc = make_cv_2d(seed=3, steps=12)
    kf = KalmanFilter(**sc.params)
    smoothed = kf.smooth(kf.filter(sc.zs))

    means, covs = joint_gaussian_posterior(sc)
    np.testing.assert_allclose(smoothed.means, means, rtol=1e-8, atol=1e-10)
    np.testing.assert_allclose(smoothed.covs, covs, rtol=1e-8, atol=1e-10)


def test_last_smoothed_estimate_equals_last_filtered(cv_scenario: LinearScenario) -> None:
    kf = KalmanFilter(**cv_scenario.params)
    filtered = kf.filter(cv_scenario.zs)
    smoothed = kf.smooth(filtered)
    np.testing.assert_array_equal(smoothed.means[-1], filtered.means[-1])
    np.testing.assert_array_equal(smoothed.covs[-1], filtered.covs[-1])


def test_smoothing_never_increases_uncertainty(cv_scenario: LinearScenario) -> None:
    kf = KalmanFilter(**cv_scenario.params)
    filtered = kf.filter(cv_scenario.zs)
    smoothed = kf.smooth(filtered)
    # P_filtered - P_smoothed must be positive semi-definite at every step.
    gap = np.linalg.eigvalsh(filtered.covs - smoothed.covs)
    assert gap.min() >= -1e-12 * np.abs(filtered.covs).max()
    np.testing.assert_array_equal(smoothed.covs, np.swapaxes(smoothed.covs, -1, -2))


def test_smoother_keeps_float32(cv_scenario: LinearScenario) -> None:
    params = {k: v.astype(np.float32) for k, v in cv_scenario.params.items()}
    kf = KalmanFilter(**params)
    smoothed = kf.smooth(kf.filter(cv_scenario.zs.astype(np.float32)))
    assert smoothed.means.dtype == np.float32
    assert smoothed.covs.dtype == np.float32


def test_smoother_handles_single_step(cv_scenario: LinearScenario) -> None:
    kf = KalmanFilter(**cv_scenario.params)
    filtered = kf.filter(cv_scenario.zs[:1])
    np.testing.assert_array_equal(kf.smooth(filtered).means, filtered.means)


def test_smooth_rejects_result_from_another_model(cv_scenario: LinearScenario) -> None:
    other = KalmanFilter(F=[[1.0]], H=[[1.0]], Q=[[1.0]], R=[[1.0]], x0=[0.0], P0=[[1.0]])
    result = other.filter([1.0, 2.0])
    with pytest.raises(ValueError, match="result does not match"):
        KalmanFilter(**cv_scenario.params).smooth(result)
