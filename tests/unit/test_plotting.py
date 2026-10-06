from collections.abc import Iterator

import numpy as np
import pytest

from kalman_py import KalmanFilter
from kalman_py.diagnostics import consistency_check, nees
from tests.scenarios import LinearScenario

matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg")
plt = pytest.importorskip("matplotlib.pyplot")

from kalman_py.plotting import plot_consistency, plot_estimates


@pytest.fixture(autouse=True)
def close_figures() -> Iterator[None]:
    yield
    plt.close("all")


def test_plot_estimates_draws_estimate_band_and_truth(cv_scenario: LinearScenario) -> None:
    sc = cv_scenario
    res = KalmanFilter(**sc.params).filter(sc.zs)
    axes = plot_estimates(res, truth=sc.truth, labels=["px", "py", "vx", "vy"], sigmas=3)
    assert len(axes) == 4
    for ax, label in zip(axes, ["px", "py", "vx", "vy"], strict=True):
        assert ax.get_ylabel() == label
        assert len(ax.lines) == 2  # estimate and truth
        assert len(ax.collections) == 1  # the ±3σ band
    np.testing.assert_array_equal(axes[2].lines[0].get_ydata(), res.means[:, 2])


def test_plot_estimates_subset_on_given_axes(cv_scenario: LinearScenario) -> None:
    sc = cv_scenario
    res = KalmanFilter(**sc.params).filter(sc.zs)
    _, (ax0, ax1) = plt.subplots(2)
    assert plot_estimates(res, states=[0, 1], axes=[ax0, ax1]) == [ax0, ax1]
    assert len(ax0.lines) == 1  # no truth given
    with pytest.raises(ValueError, match="got 2 axes for 3 states"):
        plot_estimates(res, states=[0, 1, 2], axes=[ax0, ax1])


def test_plot_estimates_accepts_jax_results(cv_scenario: LinearScenario) -> None:
    pytest.importorskip("jax")
    sc = cv_scenario
    res = KalmanFilter(**sc.params).filter(sc.zs, backend="jax")
    assert len(plot_estimates(res)) == 4


def test_plot_consistency_shows_average_and_bounds(cv_scenario: LinearScenario) -> None:
    sc = cv_scenario
    res = KalmanFilter(**sc.params).filter(sc.zs)
    check = consistency_check(nees(sc.truth, res.means, res.covs), dof=4)
    ax = plot_consistency(check, name="NEES")
    assert len(ax.lines) == 4  # average, two bounds, expected value
    assert "of steps inside the 95% bounds" in ax.get_title()
