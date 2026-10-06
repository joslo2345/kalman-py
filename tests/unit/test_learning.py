import sys

import numpy as np
import pytest

import kalman_py.backends
from kalman_py import KalmanFilter
from kalman_py.learning import Method, fit_noise
from tests.scenarios import LinearScenario, make_cv_2d, simulate


def local_level(steps: int, seed: int = 0) -> LinearScenario:
    """Random walk observed in noise: both Q and R are well identified."""
    return simulate(
        np.eye(1),
        np.eye(1),
        np.array([[0.5]]),
        np.array([[2.0]]),
        np.zeros(1),
        np.eye(1),
        steps,
        seed,
    )


def our_log_likelihood(sc: LinearScenario, Q: np.ndarray, R: np.ndarray) -> float:
    return float(KalmanFilter(sc.F, sc.H, Q, R, sc.x0, sc.P0).filter(sc.zs).log_likelihood)


def test_em_never_decreases_the_likelihood(cv_scenario: LinearScenario) -> None:
    sc = cv_scenario
    fit = fit_noise(sc.F, sc.H, sc.zs, sc.x0, sc.P0, method="em", max_iter=40, tol=0)
    assert len(fit.history) == 41
    assert np.all(np.diff(fit.history) >= -1e-9 * abs(fit.history[-1]))
    assert fit.history[-1] > fit.history[0]


@pytest.mark.parametrize("method", ["em", "gradient"])
def test_reported_log_likelihood_matches_the_filter(
    cv_scenario: LinearScenario, method: Method
) -> None:
    if method == "gradient":
        pytest.importorskip("jax")
    sc = cv_scenario
    fit = fit_noise(sc.F, sc.H, sc.zs, sc.x0, sc.P0, method=method, max_iter=20)
    np.testing.assert_allclose(fit.log_likelihood, our_log_likelihood(sc, fit.Q, fit.R), rtol=1e-12)


def test_em_and_gradient_reach_the_same_maximum() -> None:
    pytest.importorskip("jax")
    sc = local_level(steps=300)
    em = fit_noise(sc.F, sc.H, sc.zs, sc.x0, sc.P0, method="em", max_iter=5000, tol=1e-13)
    grad = fit_noise(sc.F, sc.H, sc.zs, sc.x0, sc.P0, method="gradient")
    assert grad.converged
    np.testing.assert_allclose(em.log_likelihood, grad.log_likelihood, rtol=1e-9)
    np.testing.assert_allclose(em.Q, grad.Q, rtol=1e-3)
    np.testing.assert_allclose(em.R, grad.R, rtol=1e-3)


@pytest.mark.slow
@pytest.mark.parametrize("method", ["em", "gradient"])
def test_recovers_true_noise_of_identifiable_model(method: Method) -> None:
    if method == "gradient":
        pytest.importorskip("jax")
    sc = local_level(steps=5000, seed=3)
    fit = fit_noise(sc.F, sc.H, sc.zs, sc.x0, sc.P0, method=method, max_iter=2000)
    np.testing.assert_allclose(fit.Q, sc.Q, rtol=0.15)
    np.testing.assert_allclose(fit.R, sc.R, rtol=0.1)


@pytest.mark.slow
def test_gradient_fit_is_a_stationary_point_above_em(cv_scenario: LinearScenario) -> None:
    pytest.importorskip("jax")
    sc = make_cv_2d(seed=7, steps=500)
    grad = fit_noise(sc.F, sc.H, sc.zs, sc.x0, sc.P0, method="gradient")
    em = fit_noise(sc.F, sc.H, sc.zs, sc.x0, sc.P0, method="em", max_iter=100)
    assert grad.converged
    assert grad.log_likelihood >= em.log_likelihood
    assert np.all(np.linalg.eigvalsh(grad.Q) > 0) and np.all(np.linalg.eigvalsh(grad.R) > 0)


@pytest.mark.parametrize("method", ["em", "gradient"])
def test_only_listed_covariances_are_estimated(cv_scenario: LinearScenario, method: Method) -> None:
    if method == "gradient":
        pytest.importorskip("jax")
    sc = cv_scenario
    fit = fit_noise(
        sc.F,
        sc.H,
        sc.zs,
        sc.x0,
        sc.P0,
        Q0=sc.Q,
        estimate=("R",),
        method=method,
        max_iter=50,
    )
    np.testing.assert_array_equal(fit.Q, sc.Q)
    assert not np.allclose(fit.R, np.eye(2))


def test_auto_falls_back_to_em_without_jax(
    cv_scenario: LinearScenario, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setitem(sys.modules, "kalman_py.backends.jax_backend", None)
    if hasattr(kalman_py.backends, "jax_backend"):
        monkeypatch.delattr(kalman_py.backends, "jax_backend")
    sc = cv_scenario
    auto = fit_noise(sc.F, sc.H, sc.zs, sc.x0, sc.P0, max_iter=5)
    em = fit_noise(sc.F, sc.H, sc.zs, sc.x0, sc.P0, method="em", max_iter=5)
    np.testing.assert_array_equal(auto.R, em.R)
    with pytest.raises(ImportError, match="pip install kalman-py"):
        fit_noise(sc.F, sc.H, sc.zs, sc.x0, sc.P0, method="gradient")


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"estimate": ("S",)}, "estimate must be"),
        ({"estimate": ()}, "estimate must be"),
        ({"method": "newton"}, "unknown method"),
        ({"R0": np.eye(3)}, "R0 must have shape"),
    ],
)
def test_rejects_bad_arguments(
    cv_scenario: LinearScenario, kwargs: dict[str, object], message: str
) -> None:
    sc = cv_scenario
    with pytest.raises(ValueError, match=message):
        fit_noise(sc.F, sc.H, sc.zs, sc.x0, sc.P0, **kwargs)  # type: ignore[arg-type]
