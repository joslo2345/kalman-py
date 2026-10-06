"""NIS and NEES consistency checks.

A consistent filter's normalized errors follow chi-squared distributions: the NEES
``e' P^-1 e`` (estimation error ``e`` against its covariance ``P``) with ``n`` degrees of
freedom, and the NIS (``FilterResult.nis``) with ``m``. Averaged over ``N`` Monte Carlo runs at
each time step, ``N`` times the average is chi-squared with ``N * dof`` degrees of freedom, which
gives the standard bounds (Bar-Shalom, Li & Kirubarajan, 2001, sec. 5.4).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike

from kalman_py._typing import Array


def nees(truth: ArrayLike, means: ArrayLike, covs: ArrayLike) -> Array:
    """Normalized estimation error squared ``e' P^-1 e`` with ``e = truth - means``.

    Leading dimensions broadcast, so ``truth`` and ``means`` of shape ``(..., T, n)`` with
    ``covs`` of shape ``(..., T, n, n)`` give a ``(..., T)`` result, e.g. ``(runs, T)``.
    """
    e = np.asarray(truth) - np.asarray(means)
    covs_ = np.asarray(covs)
    if covs_.shape[-2:] != (e.shape[-1], e.shape[-1]):
        raise ValueError(f"covs must end in shape {(e.shape[-1], e.shape[-1])}, got {covs_.shape}")
    solved = np.linalg.solve(covs_, e[..., None])[..., 0]
    return np.asarray((e * solved).sum(axis=-1))


def chi2_bounds(dof: float, n_runs: int = 1, confidence: float = 0.95) -> tuple[float, float]:
    """Two-sided ``confidence`` interval for the average of ``n_runs`` independent
    chi-squared(``dof``) values."""
    if not 0 < confidence < 1:
        raise ValueError(f"confidence must be in (0, 1), got {confidence}")
    if dof <= 0 or n_runs < 1:
        raise ValueError(f"need dof > 0 and n_runs >= 1, got dof={dof}, n_runs={n_runs}")
    tail = (1 - confidence) / 2
    k = dof * n_runs
    return chi2_ppf(tail, k) / n_runs, chi2_ppf(1 - tail, k) / n_runs


@dataclass(frozen=True)
class ConsistencyCheck:
    """Per-step averages of NIS or NEES over Monte Carlo runs against chi-squared bounds."""

    average: Array  # (T,) average over runs at each step
    lower: float
    upper: float
    confidence: float
    n_runs: int
    dof: float

    @property
    def inside(self) -> Array:
        """Which steps' averages fall inside the bounds."""
        return np.asarray((self.average >= self.lower) & (self.average <= self.upper))

    @property
    def fraction_inside(self) -> float:
        """Should be close to ``confidence`` for a consistent filter. Clearly lower values with
        averages above ``upper`` mean an overconfident filter (P too small); below ``lower``,
        an overcautious one.

        NEES is usually strongly correlated from step to step, so excursions come in clusters
        and this fraction varies much more between Monte Carlo batches than a binomial
        proportion would (several points below ``confidence`` is normal). The time average of
        ``average`` lying inside ``[lower, upper]`` is a more robust check."""
        return float(self.inside.mean())


def consistency_check(values: ArrayLike, dof: float, confidence: float = 0.95) -> ConsistencyCheck:
    """Average NIS or NEES ``values`` of shape ``(runs, T)`` (or ``(T,)`` for one run) over the
    runs and compare each step with the chi-squared bounds.

    ``dof`` is the state dimension for NEES and the measurement dimension for NIS.
    """
    v = np.asarray(values, dtype=float)
    if v.ndim == 1:
        v = v[None]
    if v.ndim != 2:
        raise ValueError(f"values must have shape (runs, T) or (T,), got {v.shape}")
    lower, upper = chi2_bounds(dof, v.shape[0], confidence)
    return ConsistencyCheck(v.mean(axis=0), lower, upper, confidence, v.shape[0], dof)


# ---- Chi-squared quantiles without SciPy ---------------------------------------------------


def chi2_cdf(x: float, dof: float) -> float:
    """Chi-squared CDF: the regularized lower incomplete gamma function ``P(dof/2, x/2)``."""
    if x <= 0:
        return 0.0
    return _regularized_gamma_p(dof / 2, x / 2)


def chi2_ppf(p: float, dof: float) -> float:
    """Chi-squared quantile, by bisection on [`chi2_cdf`][kalman_py.diagnostics.chi2_cdf]
    (accurate to ~1e-12 relative)."""
    if not 0 < p < 1:
        raise ValueError(f"p must be in (0, 1), got {p}")
    lo, hi = 0.0, dof + 10 * math.sqrt(2 * dof) + 10
    while chi2_cdf(hi, dof) < p:
        lo, hi = hi, 2 * hi
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if chi2_cdf(mid, dof) < p:
            lo = mid
        else:
            hi = mid
        if hi - lo <= 1e-14 * hi:
            break
    return 0.5 * (lo + hi)


def _regularized_gamma_p(a: float, x: float) -> float:
    # Numerical Recipes (3rd ed.) sec. 6.2: series below a + 1, continued fraction above.
    log_prefactor = -x + a * math.log(x) - math.lgamma(a)
    eps, tiny = 1e-16, 1e-300
    if x < a + 1:
        term = total = 1 / a
        ap = a
        for _ in range(100_000):
            ap += 1
            term *= x / ap
            total += term
            if abs(term) < abs(total) * eps:
                break
        return min(1.0, total * math.exp(log_prefactor))
    # Modified Lentz evaluation of the continued fraction for Q(a, x) = 1 - P(a, x).
    b = x + 1 - a
    c = 1 / tiny
    d = 1 / b
    h = d
    for i in range(1, 100_000):
        an = -i * (i - a)
        b += 2
        d = an * d + b
        d = tiny if abs(d) < tiny else d
        c = b + an / c
        c = tiny if abs(c) < tiny else c
        d = 1 / d
        delta = d * c
        h *= delta
        if abs(delta - 1) < eps:
            break
    return max(0.0, 1 - math.exp(log_prefactor) * h)
