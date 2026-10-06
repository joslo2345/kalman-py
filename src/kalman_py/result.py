"""Result containers returned by batch filtering and smoothing.

They are generic over the array type: the NumPy backend fills them with NumPy arrays, the JAX
backend with JAX arrays (left on the device, so computation can stay asynchronous).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Generic, TypeVar

A = TypeVar("A")


@dataclass(frozen=True)
class FilterResult(Generic[A]):
    """Output of a batch filter run over ``T`` measurements.

    Row ``k`` of ``means``/``covs`` is the posterior after measurement ``k``; row ``k`` of
    ``predicted_means``/``predicted_covs`` is the prior just before it (needed by smoothers).
    """

    means: A  # (T, n)
    covs: A  # (T, n, n)
    predicted_means: A  # (T, n)
    predicted_covs: A  # (T, n, n)
    nis: A  # (T,) normalized innovation squared
    log_likelihood: A  # 0-d array: log p(z_1, ..., z_T)


@dataclass(frozen=True)
class ExtendedFilterResult(FilterResult[A]):
    """:class:`FilterResult` plus the linearized models the EKF used, needed for smoothing."""

    transition_jacobians: A  # (T, n, n): row k is df/dx at the posterior before step k


@dataclass(frozen=True)
class SmootherResult(Generic[A]):
    """Smoothed estimates ``p(x_k | z_1, ..., z_T)`` for every step ``k``."""

    means: A  # (T, n)
    covs: A  # (T, n, n)
