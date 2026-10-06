"""Maximum-likelihood estimation of the noise covariances Q and R of a linear model."""

from __future__ import annotations

from collections.abc import Collection
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
from numpy.typing import ArrayLike

from kalman_py._common import as_float_arrays, check_measurements, check_shape, jax_backend
from kalman_py._typing import Array
from kalman_py.backends import numpy_backend
from kalman_py.result import FilterResult

Method = Literal["auto", "em", "gradient"]

_DEFAULTS = {"em": (200, 1e-8), "gradient": (1000, 1e-6)}  # (max_iter, tol)


@dataclass(frozen=True)
class NoiseFit:
    """Fitted noise covariances and how the fit went."""

    Q: Array
    R: Array
    log_likelihood: float  # log p(z_1..z_T) under the fitted Q, R (KalmanFilter convention)
    n_iter: int
    converged: bool
    history: Array  # log-likelihood after each EM iteration or each BFGS run


def fit_noise(
    F: ArrayLike,
    H: ArrayLike,
    zs: ArrayLike,
    x0: ArrayLike,
    P0: ArrayLike,
    *,
    Q0: ArrayLike | None = None,
    R0: ArrayLike | None = None,
    estimate: Collection[str] = ("Q", "R"),
    method: Method = "auto",
    max_iter: int | None = None,
    tol: float | None = None,
) -> NoiseFit:
    """Estimate ``Q`` and/or ``R`` of ``x_k = F x_{k-1} + w``, ``z_k = H x_k + v`` by maximum
    likelihood, with ``(x0, P0)`` the prior before the first prediction (as in
    [`KalmanFilter`][kalman_py.KalmanFilter]).

    ``method="gradient"`` maximizes the exact log-likelihood with BFGS, differentiating through
    the JAX filter (needs JAX; enable ``jax_enable_x64`` for float64 accuracy). Q and R are
    parameterized by Cholesky factors with log-diagonals, so they stay positive-definite. The
    likelihood is often flat and badly conditioned (e.g. a full Q from position-only data), where
    BFGS line searches stall; it is restarted from where it stopped until the gradient of the
    per-step average log-likelihood is below ``tol`` (default 1e-6) or progress stops.
    ``max_iter`` (default 1000) caps the BFGS iterations of each run.

    ``method="em"`` is expectation-maximization (Shumway & Stoffer) on the NumPy backend: each
    iteration runs the filter and RTS smoother, then updates Q and R in closed form. It never
    decreases the likelihood and needs no JAX, but on such flat likelihoods it converges very
    slowly. It stops when an iteration improves the log-likelihood by less than
    ``tol * max(1, |log-likelihood|)`` (default 1e-8) or after ``max_iter`` (default 200).

    ``method="auto"`` uses the gradient method when JAX is installed, EM otherwise.

    ``Q0`` and ``R0`` are starting values (identity by default, like pykalman); covariances not
    listed in ``estimate`` stay fixed at them.
    """
    unknown = set(estimate) - {"Q", "R"}
    if unknown or not estimate:
        raise ValueError(f"estimate must be a non-empty subset of ('Q', 'R'), got {estimate!r}")
    F_, H_, x0_, P0_ = as_float_arrays(F, H, x0, P0)
    m, n = H_.shape
    Q_init = np.eye(n, dtype=P0_.dtype) if Q0 is None else np.asarray(Q0, dtype=P0_.dtype)
    R_init = np.eye(m, dtype=P0_.dtype) if R0 is None else np.asarray(R0, dtype=P0_.dtype)
    check_shape("F", F_, (n, n))
    check_shape("x0", x0_, (n,))
    check_shape("P0", P0_, (n, n))
    check_shape("Q0", Q_init, (n, n))
    check_shape("R0", R_init, (m, m))
    zs_ = check_measurements(np.asarray(zs, dtype=P0_.dtype), m)

    if method == "auto":
        try:
            jax_backend()
            method = "gradient"
        except ImportError:
            method = "em"
    if method not in _DEFAULTS:
        raise ValueError(f"unknown method {method!r}; expected 'auto', 'em' or 'gradient'")
    default_iter, default_tol = _DEFAULTS[method]
    args = (F_, H_, zs_, x0_, P0_, Q_init, R_init, set(estimate))
    limits = (default_iter if max_iter is None else max_iter, default_tol if tol is None else tol)
    return _fit_em(*args, *limits) if method == "em" else _fit_gradient(*args, *limits)


# ---- EM ------------------------------------------------------------------------------------


def _fit_em(
    F: Array,
    H: Array,
    zs: Array,
    x0: Array,
    P0: Array,
    Q: Array,
    R: Array,
    estimate: set[str],
    max_iter: int,
    tol: float,
) -> NoiseFit:
    history: list[float] = []
    converged = False
    for _ in range(max_iter + 1):
        filtered = numpy_backend.kalman_filter(F, H, Q, R, x0, P0, zs)
        ll = float(filtered.log_likelihood)
        history.append(ll)
        if len(history) > 1 and abs(ll - history[-2]) <= tol * max(1.0, abs(ll)):
            converged = True
            break
        if len(history) > max_iter:
            break
        Q, R = _em_step(F, H, zs, x0, P0, filtered, estimate, Q, R)
    return NoiseFit(Q, R, history[-1], len(history) - 1, converged, np.array(history))


def _em_step(
    F: Array,
    H: Array,
    zs: Array,
    x0: Array,
    P0: Array,
    filtered: Any,
    estimate: set[str],
    Q: Array,
    R: Array,
) -> tuple[Array, Array]:
    """One M-step from the smoothed moments of x_0 (the prior) through x_T."""
    smoothed = numpy_backend.rts_smoother(F, filtered)
    # Extend the smoother back to the prior state x_0, which precedes the first prediction.
    G0 = np.linalg.solve(filtered.predicted_covs[0], F @ P0).T
    m0 = x0 + G0 @ (smoothed.means[0] - filtered.predicted_means[0])
    C0 = P0 + G0 @ (smoothed.covs[0] - filtered.predicted_covs[0]) @ G0.T
    means = np.concatenate((m0[None], smoothed.means))  # (T + 1, n): x_0 .. x_T
    covs = np.concatenate((C0[None], smoothed.covs))
    gains = np.concatenate((G0[None], smoothed.gains))  # gains[j] links x_j and x_{j+1}
    # Lag-one covariances Cov(x_k, x_{k-1} | all data), k = 1..T.
    lag_one = covs[1:] @ np.swapaxes(gains, -1, -2)

    if "Q" in estimate:
        # Q = mean over k of E[(x_k - F x_{k-1})(x_k - F x_{k-1})'].
        d = means[1:] - means[:-1] @ F.T
        F_lag = F @ np.swapaxes(lag_one, -1, -2)  # F Cov(x_{k-1}, x_k)
        terms = (
            d[:, :, None] * d[:, None, :]
            + covs[1:]
            - F_lag
            - np.swapaxes(F_lag, -1, -2)
            + F @ covs[:-1] @ F.T
        )
        Q = numpy_backend._symmetrize(terms.mean(axis=0))
    if "R" in estimate:
        # R = mean over k of E[(z_k - H x_k)(z_k - H x_k)'].
        e = zs - means[1:] @ H.T
        terms = e[:, :, None] * e[:, None, :] + H @ covs[1:] @ H.T
        R = numpy_backend._symmetrize(terms.mean(axis=0))
    return Q, R


# ---- Gradient ------------------------------------------------------------------------------


def _fit_gradient(
    F: Array,
    H: Array,
    zs: Array,
    x0: Array,
    P0: Array,
    Q: Array,
    R: Array,
    estimate: set[str],
    max_iter: int,
    tol: float,
) -> NoiseFit:
    jb = jax_backend()
    import jax
    import jax.numpy as jnp
    from jax.scipy.optimize import minimize

    dtype = np.dtype(P0.dtype)
    F_, H_, x0_, P0_, zs_, Q_, R_ = (jb.asarray(a, dtype) for a in (F, H, x0, P0, zs, Q, R))
    n, m = F.shape[0], H.shape[0]
    rows = {"Q": np.tril_indices(n), "R": np.tril_indices(m)}
    names = [name for name in ("Q", "R") if name in estimate]

    def to_params(M: Array) -> Array:
        L = np.linalg.cholesky(M)
        L[np.diag_indices_from(L)] = np.log(np.diagonal(L))
        return L[np.tril_indices_from(L)]

    def from_params(theta: jax.Array, size: int) -> jax.Array:
        L = jnp.zeros((size, size), dtype=theta.dtype).at[np.tril_indices(size)].set(theta)
        L = L.at[np.diag_indices(size)].set(jnp.exp(jnp.diagonal(L)))
        return L @ L.T

    sizes = [len(rows[name][0]) for name in names]
    splits = np.cumsum(sizes)[:-1].tolist()
    theta0 = jnp.concatenate(
        [jb.asarray(to_params(Q if name == "Q" else R), dtype) for name in names]
    )

    def covariances(theta: jax.Array) -> tuple[jax.Array, jax.Array]:
        parts = dict(zip(names, jnp.split(theta, splits), strict=True))
        Q_t = from_params(parts["Q"], n) if "Q" in parts else Q_
        R_t = from_params(parts["R"], m) if "R" in parts else R_
        return Q_t, R_t

    T = zs.shape[0]

    def objective(theta: jax.Array) -> jax.Array:
        Q_t, R_t = covariances(theta)
        result: FilterResult[jax.Array] = jb._kalman_filter(
            F_, H_, Q_t, R_t, x0_, P0_, zs_, square_root=False
        )
        value: jax.Array = -result.log_likelihood / T  # per-step scale keeps BFGS well conditioned
        return value

    solve = jax.jit(
        lambda theta: minimize(
            objective, theta, method="BFGS", options={"maxiter": max_iter, "gtol": tol}
        )
    )
    gradient = jax.jit(jax.grad(objective))

    theta, best, history, n_iter = theta0, np.inf, [], 0
    converged = False
    for _ in range(_MAX_BFGS_RESTARTS):
        opt = solve(theta)
        n_iter += int(opt.nit)
        value = float(opt.fun)
        if not value < best:  # no progress: keep the previous point
            break
        theta, best = opt.x, value
        history.append(-value * T)
        if float(jnp.abs(gradient(theta)).max()) <= tol:
            converged = True
            break

    Q_fit, R_fit = (np.asarray(M) for M in covariances(theta))
    return NoiseFit(Q_fit, R_fit, history[-1], n_iter, converged, np.array(history, dtype=dtype))


# BFGS runs per fit; each restart resets the curvature estimate after a stalled line search.
_MAX_BFGS_RESTARTS = 20
