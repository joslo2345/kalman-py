"""NumPy backend.

All functions preserve the dtype of their inputs, so a float32 problem stays float32.

Covariances are carried in one of two forms:

- Joseph form (default): the covariance ``P`` itself, updated with the Joseph formula.
- Square-root form: a factor ``S`` with ``P = S S'``, propagated with QR decompositions.
  ``S S'`` is positive semi-definite by construction, so it survives problems where rounding
  makes the Joseph form indefinite (e.g. very precise sensors). It costs a QR per step.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from typing import NamedTuple

import numpy as np

from kalman_py._typing import Array
from kalman_py.result import ExtendedFilterResult, FilterResult, SmootherResult

TransitionFn = Callable[[Array, float], Array]  # f(x, dt) -> x_next, or its Jacobian
MeasurementFn = Callable[[Array], Array]  # h(x) -> z_pred, or its Jacobian
ResidualFn = Callable[[Array, Array], Array]  # residual(z, z_pred) -> innovation

_LOG_2PI = math.log(2 * math.pi)


class Correction(NamedTuple):
    """Result of a measurement update. In square-root form ``P`` and ``S`` are factors."""

    x: Array
    P: Array  # posterior covariance
    y: Array  # innovation
    S: Array  # innovation covariance


def _symmetrize(P: Array) -> Array:
    return 0.5 * (P + P.T)


def psd_factor(M: Array, name: str = "matrix") -> Array:
    """Return ``L`` with ``L L' = M`` for a symmetric positive semi-definite ``M``.

    Uses Cholesky when ``M`` is positive-definite and an eigendecomposition otherwise, since
    noise covariances are often singular (e.g. process noise driving only some states).
    """
    try:
        return np.linalg.cholesky(M)
    except np.linalg.LinAlgError:
        w, V = np.linalg.eigh(_symmetrize(M))
        if w.min() < -100 * np.finfo(M.dtype).eps * float(abs(w).max()):
            raise ValueError(f"{name} must be positive semi-definite") from None
        return V * np.sqrt(np.clip(w, 0, None))


# ---- Joseph form ---------------------------------------------------------------------------


def predict(x: Array, P: Array, F: Array, Q: Array) -> tuple[Array, Array]:
    """Propagate the state and covariance one step through a linear model."""
    return F @ x, _symmetrize(F @ P @ F.T + Q)


def update(
    x: Array, P: Array, z: Array, H: Array, R: Array, identity: Array | None = None
) -> Correction:
    """Condition the state on measurement ``z`` using the Joseph-form covariance update.

    Pass ``identity`` (``np.eye(n)`` in P's dtype) when calling in a loop, to skip rebuilding it.
    """
    return _correct(x, P, z - H @ x, H, R, identity)


def _correct(
    x: Array, P: Array, y: Array, H: Array, R: Array, identity: Array | None = None
) -> Correction:
    """Measurement update given the innovation ``y`` and (linearized) measurement matrix ``H``.

    Per-step cost is dominated by fixed call overhead on small matrices, so this does the
    minimum: NIS and likelihood are left to :func:`_innovation_stats`, which batches them.
    """
    if identity is None:
        identity = np.eye(x.shape[0], dtype=P.dtype)
    K, S = _joseph_gain(P, H, R)
    return Correction(x + K @ y, _joseph_covariance(P, K, H, R, identity), y, S)


def _joseph_gain(P: Array, H: Array, R: Array) -> tuple[Array, Array]:
    """Kalman gain and innovation covariance."""
    HP = H @ P
    # Not symmetrized: S only feeds the gain, and the Joseph form is valid for any gain.
    S = HP @ H.T + R
    # P and S are symmetric, so K = P H' S^-1 = (S^-1 H P)', without an explicit inverse.
    return np.linalg.solve(S, HP).T, S


def _joseph_covariance(P: Array, K: Array, H: Array, R: Array, identity: Array) -> Array:
    # Joseph form is far more robust than P = (I - KH) P, but can still turn indefinite under
    # rounding when cond(P) is extreme; the square-root form below cannot.
    I_KH = identity - K @ H
    return _symmetrize(I_KH @ P @ I_KH.T + K @ R @ K.T)


def _innovation_stats(ys: Array, Ss: Array, square_root: bool) -> tuple[Array, Array]:
    """Per-step NIS and total log-likelihood from innovations ``ys`` (T, m) and their
    covariances ``Ss`` (T, m, m), or lower-triangular factors of them in square-root form."""
    m = ys.shape[1]
    w = np.linalg.solve(Ss, ys[..., None])[..., 0]
    if square_root:
        nis = (w * w).sum(axis=-1)  # w = L^-1 y is the whitened innovation
        logdet = 2 * np.log(np.abs(np.diagonal(Ss, axis1=-2, axis2=-1))).sum(axis=-1)
    else:
        nis = (ys * w).sum(axis=-1)
        logdet = np.linalg.slogdet(Ss)[1]
    log_likelihood = -0.5 * (nis + logdet + m * _LOG_2PI)
    return nis, np.asarray(log_likelihood.sum(dtype=np.float64), dtype=ys.dtype)


# ---- Square-root form ----------------------------------------------------------------------


def sqrt_predict(x: Array, S: Array, F: Array, L_Q: Array) -> tuple[Array, Array]:
    """Predict with covariance factors: ``S`` and ``L_Q`` are factors of ``P`` and ``Q``."""
    # P_pred = [F S, L_Q] [F S, L_Q]'. If [F S, L_Q]' = Q R then P_pred = R' R.
    r = np.linalg.qr(np.vstack(((F @ S).T, L_Q.T)), mode="r")
    return F @ x, r.T


def sqrt_update(x: Array, S: Array, z: Array, H: Array, L_R: Array) -> Correction:
    """Square-root counterpart of :func:`update`; ``S`` and ``L_R`` factor ``P`` and ``R``."""
    return _sqrt_correct(x, S, z - H @ x, H, L_R)


def _sqrt_correct(x: Array, S: Array, y: Array, H: Array, L_R: Array) -> Correction:
    m, n = L_R.shape[0], S.shape[0]
    # Triangularizing the pre-array  [L_R  H S]   gives   [L_y  0     ]
    #                                [0    S  ]           [Kb   S_post]
    # with L_y L_y' = H P H' + R (innovation covariance), Kb = K L_y and S_post S_post' = P_post.
    pre = np.block([[L_R, H @ S], [np.zeros((n, m), dtype=S.dtype), S]])
    post = np.linalg.qr(pre.T, mode="r").T
    L_y, Kb, S_post = post[:m, :m], post[m:, :m], post[m:, m:]

    w = np.linalg.solve(L_y, y)  # whitened innovation: L_y^-1 y
    return Correction(x + Kb @ w, S_post, y, L_y)


# ---- Extended Kalman filter steps ----------------------------------------------------------


def ekf_predict(
    x: Array,
    P: Array,
    Q: Array,
    dt: float,
    f: TransitionFn,
    jac_f: TransitionFn,
    square_root: bool = False,
) -> tuple[Array, Array, Array]:
    """Propagate through ``f``, linearized at ``x``; also returns the Jacobian used.

    In square-root form ``P`` and ``Q`` are factors, and so is the returned covariance.
    """
    F = np.asarray(jac_f(x, dt), dtype=P.dtype)
    x_pred = np.asarray(f(x, dt), dtype=P.dtype)
    if square_root:
        return x_pred, sqrt_predict(x, P, F, Q)[1], F
    return x_pred, _symmetrize(F @ P @ F.T + Q), F


def ekf_update(
    x: Array,
    P: Array,
    z: Array,
    R: Array,
    h: MeasurementFn,
    jac_h: MeasurementFn,
    residual: ResidualFn,
    square_root: bool = False,
    identity: Array | None = None,
) -> Correction:
    """Condition on ``z`` with ``h`` linearized at ``x`` (``P``, ``R`` factors if square-root)."""
    H = np.asarray(jac_h(x), dtype=P.dtype)
    y = np.asarray(residual(z, np.asarray(h(x), dtype=P.dtype)), dtype=P.dtype)
    if square_root:
        return _sqrt_correct(x, P, y, H, R)
    return _correct(x, P, y, H, R, identity)


# ---- Batch filtering -----------------------------------------------------------------------

# One filter step from (x, C), with C the covariance or its factor:
# returns (x_pred, C_pred, correction, transition_matrix).
_Step = Callable[[int, Array, Array], tuple[Array, Array, Correction, Array]]


class _Run(NamedTuple):
    result: FilterResult[Array]
    jacobians: Array


def _run_filter(x0: Array, C0: Array, zs: Array, step: _Step, square_root: bool) -> _Run:
    T, m = zs.shape
    n, dtype = x0.shape[0], C0.dtype
    means = np.empty((T, n), dtype=dtype)
    covs = np.empty((T, n, n), dtype=dtype)
    predicted_means = np.empty((T, n), dtype=dtype)
    predicted_covs = np.empty((T, n, n), dtype=dtype)
    jacobians = np.empty((T, n, n), dtype=dtype)
    innovations = np.empty((T, m), dtype=dtype)
    innovation_covs = np.empty((T, m, m), dtype=dtype)

    x, C = x0, C0
    for k in range(T):
        predicted_means[k], C_pred, (x, C, innovations[k], innovation_covs[k]), jacobians[k] = step(
            k, x, C
        )
        if square_root:
            predicted_covs[k], covs[k] = C_pred @ C_pred.T, C @ C.T
        else:
            predicted_covs[k], covs[k] = C_pred, C
        means[k] = x

    nis, log_likelihood = _innovation_stats(innovations, innovation_covs, square_root)
    result = FilterResult(means, covs, predicted_means, predicted_covs, nis, log_likelihood)
    return _Run(result, jacobians)


def kalman_filter(
    F: Array,
    H: Array,
    Q: Array,
    R: Array,
    x0: Array,
    P0: Array,
    zs: Array,
    square_root: bool = False,
) -> FilterResult[Array]:
    """Run predict + update for every row of ``zs``, starting from the prior ``(x0, P0)``."""
    if square_root:
        L_Q, L_R = psd_factor(Q, "Q"), psd_factor(R, "R")

        def sqrt_step(k: int, x: Array, S: Array) -> tuple[Array, Array, Correction, Array]:
            x_pred, S_pred = sqrt_predict(x, S, F, L_Q)
            return x_pred, S_pred, sqrt_update(x_pred, S_pred, zs[k], H, L_R), F

        return _run_filter(x0, psd_factor(P0, "P0"), zs, sqrt_step, True).result

    return _joseph_kalman_filter(F, H, Q, R, x0, P0, zs)


_STEADY_STATE_CHECK_EVERY = 16


def _joseph_kalman_filter(
    F: Array,
    H: Array,
    Q: Array,
    R: Array,
    x0: Array,
    P0: Array,
    zs: Array,
    detect_steady_state: bool = True,
) -> FilterResult[Array]:
    """Joseph-form batch filter with an exact steady-state shortcut.

    For a time-invariant model the covariance recursion ignores the measurements. Once the
    predicted covariance repeats bit for bit, every later covariance and gain would be the same
    bits again, so they are reused and each remaining step only updates the mean, with the same
    operations as the full step. The result is bitwise identical to the full computation.
    """
    T, m = zs.shape
    n, dtype = x0.shape[0], P0.dtype
    identity = np.eye(n, dtype=dtype)
    means = np.empty((T, n), dtype=dtype)
    covs = np.empty((T, n, n), dtype=dtype)
    predicted_means = np.empty((T, n), dtype=dtype)
    predicted_covs = np.empty((T, n, n), dtype=dtype)
    innovations = np.empty((T, m), dtype=dtype)
    innovation_covs = np.empty((T, m, m), dtype=dtype)

    x, P = x0, P0
    k = 0
    while k < T:
        x_pred, P_pred = predict(x, P, F, Q)
        K, S = _joseph_gain(P_pred, H, R)
        y = zs[k] - H @ x_pred
        x, P = x_pred + K @ y, _joseph_covariance(P_pred, K, H, R, identity)
        predicted_means[k], predicted_covs[k], means[k], covs[k] = x_pred, P_pred, x, P
        innovations[k], innovation_covs[k] = y, S
        k += 1
        # A repeat at step k persists at every later step, so checking only every
        # _STEADY_STATE_CHECK_EVERY steps is just as exact and keeps the check's cost negligible.
        if (
            detect_steady_state
            and k % _STEADY_STATE_CHECK_EVERY == 0
            and np.array_equal(P_pred, predicted_covs[k - 2])
        ):
            break

    # Steady state (if reached): same mean arithmetic as above, covariances repeat.
    predicted_covs[k:], covs[k:], innovation_covs[k:] = P_pred, P, S
    for j in range(k, T):
        x_pred = F @ x
        y = zs[j] - H @ x_pred
        x = x_pred + K @ y
        predicted_means[j], means[j], innovations[j] = x_pred, x, y

    nis, log_likelihood = _innovation_stats(innovations, innovation_covs, False)
    return FilterResult(means, covs, predicted_means, predicted_covs, nis, log_likelihood)


def ekf_filter(
    Q: Array,
    R: Array,
    x0: Array,
    P0: Array,
    zs: Array,
    dts: Array,
    f: TransitionFn,
    h: MeasurementFn,
    jac_f: TransitionFn,
    jac_h: MeasurementFn,
    residual: ResidualFn,
    square_root: bool = False,
) -> ExtendedFilterResult[Array]:
    """Extended Kalman filter over every row of ``zs``; ``dts[k]`` is the step before ``zs[k]``."""
    Q_, R_, C0 = (
        (psd_factor(Q, "Q"), psd_factor(R, "R"), psd_factor(P0, "P0"))
        if square_root
        else (Q, R, P0)
    )
    identity = np.eye(x0.shape[0], dtype=P0.dtype)

    def step(k: int, x: Array, C: Array) -> tuple[Array, Array, Correction, Array]:
        x_pred, C_pred, F = ekf_predict(x, C, Q_, float(dts[k]), f, jac_f, square_root)
        updated = ekf_update(x_pred, C_pred, zs[k], R_, h, jac_h, residual, square_root, identity)
        return x_pred, C_pred, updated, F

    run = _run_filter(x0, C0, zs, step, square_root)
    r = run.result
    return ExtendedFilterResult(
        r.means, r.covs, r.predicted_means, r.predicted_covs, r.nis, r.log_likelihood, run.jacobians
    )


def rts_smoother(F: Array, result: FilterResult[Array]) -> SmootherResult[Array]:
    """Rauch-Tung-Striebel backward pass over the output of :func:`kalman_filter`.

    ``F`` is the transition matrix, or a ``(T, n, n)`` stack where ``F[k]`` is the (linearized)
    transition into step ``k``, as the EKF stores it.
    """
    means = result.means.copy()
    covs = result.covs.copy()
    for k in range(means.shape[0] - 2, -1, -1):
        P = result.covs[k]
        P_pred = result.predicted_covs[k + 1]
        F_k = F if F.ndim == 2 else F[k + 1]
        # G = P F' P_pred^-1 = (P_pred^-1 F P)' since P and P_pred are symmetric.
        G = np.linalg.solve(P_pred, F_k @ P).T
        means[k] = result.means[k] + G @ (means[k + 1] - result.predicted_means[k + 1])
        covs[k] = _symmetrize(P + G @ (covs[k + 1] - P_pred) @ G.T)
    return SmootherResult(means, covs)
