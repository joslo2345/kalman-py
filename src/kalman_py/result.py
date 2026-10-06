"""Result containers returned by batch filtering."""

from __future__ import annotations

from dataclasses import dataclass

from kalman_py._typing import Array


@dataclass(frozen=True)
class FilterResult:
    """Output of a batch filter run over ``T`` measurements.

    Row ``k`` of ``means``/``covs`` is the posterior after measurement ``k``; row ``k`` of
    ``predicted_means``/``predicted_covs`` is the prior just before it (needed by smoothers).
    """

    means: Array  # (T, n)
    covs: Array  # (T, n, n)
    predicted_means: Array  # (T, n)
    predicted_covs: Array  # (T, n, n)
    nis: Array  # (T,) normalized innovation squared
    log_likelihood: float  # log p(z_1, ..., z_T)


@dataclass(frozen=True)
class SmootherResult:
    """Smoothed estimates ``p(x_k | z_1, ..., z_T)`` for every step ``k``."""

    means: Array  # (T, n)
    covs: Array  # (T, n, n)
