"""Plotting helpers (requires the ``plot`` extra: ``pip install kalman-py[plot]``)."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

import numpy as np
from numpy.typing import ArrayLike

from kalman_py.diagnostics import ConsistencyCheck

if TYPE_CHECKING:
    from matplotlib.axes import Axes


def _pyplot() -> Any:
    try:
        import matplotlib.pyplot as plt
    except ImportError as e:
        raise ImportError(
            "plotting needs matplotlib; install it with `pip install kalman-py[plot]`"
        ) from e
    return plt


def plot_estimates(
    result: Any,
    truth: ArrayLike | None = None,
    states: Sequence[int] | None = None,
    labels: Sequence[str] | None = None,
    times: ArrayLike | None = None,
    sigmas: float = 2.0,
    axes: Sequence[Axes] | None = None,
) -> list[Axes]:
    """Plot each state's estimate with a ``±sigmas`` standard-deviation band, and the truth if
    given, one subplot per state.

    ``result`` is anything with ``means`` (T, n) and ``covs`` (T, n, n): a filter or smoother
    result from either backend.
    """
    means = np.asarray(result.means)
    stds = np.sqrt(np.diagonal(np.asarray(result.covs), axis1=-2, axis2=-1))
    T, n = means.shape
    states = list(range(n)) if states is None else list(states)
    t = np.arange(T) if times is None else np.asarray(times)
    truth_ = None if truth is None else np.asarray(truth)
    if axes is None:
        _, created = _pyplot().subplots(len(states), 1, sharex=True, squeeze=False)
        axes = list(created[:, 0])
    if len(axes) != len(states):
        raise ValueError(f"got {len(axes)} axes for {len(states)} states")

    for ax, i in zip(axes, states, strict=True):
        label = labels[i] if labels is not None else f"x[{i}]"
        (line,) = ax.plot(t, means[:, i], label="estimate")
        ax.fill_between(
            t,
            means[:, i] - sigmas * stds[:, i],
            means[:, i] + sigmas * stds[:, i],
            color=line.get_color(),
            alpha=0.25,
            linewidth=0,
            label=f"±{sigmas:g}σ",
        )
        if truth_ is not None:
            ax.plot(t, truth_[:, i], color="black", linewidth=1, linestyle="--", label="truth")
        ax.set_ylabel(label)
    axes[0].legend(loc="best")
    axes[-1].set_xlabel("time" if times is not None else "step")
    return list(axes)


def plot_consistency(
    check: ConsistencyCheck,
    name: str = "NEES",
    times: ArrayLike | None = None,
    ax: Axes | None = None,
) -> Axes:
    """Plot the per-step average NIS or NEES against its chi-squared bounds."""
    if ax is None:
        _, ax = _pyplot().subplots()
    assert ax is not None
    t = np.arange(check.average.shape[0]) if times is None else np.asarray(times)
    average = "average " if check.n_runs > 1 else ""
    ax.plot(t, check.average, label=f"{average}{name} ({check.n_runs} runs)")
    for bound in (check.lower, check.upper):
        ax.axhline(bound, color="red", linestyle="--", linewidth=1)
    ax.axhline(check.dof, color="gray", linewidth=1, label=f"expected ({check.dof:g})")
    ax.set_title(
        f"{name}: {check.fraction_inside:.1%} of steps inside the "
        f"{check.confidence:.0%} bounds [{check.lower:.3g}, {check.upper:.3g}]"
    )
    ax.set_xlabel("time" if times is not None else "step")
    ax.legend(loc="best")
    return ax
