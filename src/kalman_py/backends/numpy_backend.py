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


class UpdateResult(NamedTuple):
    x: Array
    P: Array  # the covariance, or its factor in square-root form
    nis: float
    log_likelihood: float


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


def update(x: Array, P: Array, z: Array, H: Array, R: Array) -> UpdateResult:
    """Condition the state on measurement ``z`` using the Joseph-form covariance update."""
    return _correct(x, P, z - H @ x, H, R)


def _correct(x: Array, P: Array, y: Array, H: Array, R: Array) -> UpdateResult:
    """Measurement update given the innovation ``y`` and (linearized) measurement matrix ``H``."""
    HP = H @ P
    S = _symmetrize(HP @ H.T + R)
    # P and S are symmetric, so K = P H' S^-1 = (S^-1 H P)'. One solve yields K' and S^-1 y,
    # avoiding an explicit inverse of S.
    sol = np.linalg.solve(S, np.column_stack((HP, y)))
    K = sol[:, :-1].T
    nis = float(y @ sol[:, -1])

    # Joseph form is far more robust than P = (I - KH) P, but can still turn indefinite under
    # rounding when cond(P) is extreme; the square-root form below cannot.
    I_KH = np.eye(x.shape[0], dtype=P.dtype) - K @ H
    P_post = _symmetrize(I_KH @ P @ I_KH.T + K @ R @ K.T)

    _, logdet = np.linalg.slogdet(S)
    log_likelihood = -0.5 * (nis + float(logdet) + y.shape[0] * _LOG_2PI)
    return UpdateResult(x + K @ y, P_post, nis, log_likelihood)


# ---- Square-root form ----------------------------------------------------------------------


def sqrt_predict(x: Array, S: Array, F: Array, L_Q: Array) -> tuple[Array, Array]:
    """Predict with covariance factors: ``S`` and ``L_Q`` are factors of ``P`` and ``Q``."""
    # P_pred = [F S, L_Q] [F S, L_Q]'. If [F S, L_Q]' = Q R then P_pred = R' R.
    r = np.linalg.qr(np.vstack(((F @ S).T, L_Q.T)), mode="r")
    return F @ x, r.T


def sqrt_update(x: Array, S: Array, z: Array, H: Array, L_R: Array) -> UpdateResult:
    """Square-root counterpart of :func:`update`; ``S`` and ``L_R`` factor ``P`` and ``R``."""
    return _sqrt_correct(x, S, z - H @ x, H, L_R)


def _sqrt_correct(x: Array, S: Array, y: Array, H: Array, L_R: Array) -> UpdateResult:
    m, n = L_R.shape[0], S.shape[0]
    # Triangularizing the pre-array  [L_R  H S]   gives   [L_y  0     ]
    #                                [0    S  ]           [Kb   S_post]
    # with L_y L_y' = H P H' + R (innovation covariance), Kb = K L_y and S_post S_post' = P_post.
    pre = np.block([[L_R, H @ S], [np.zeros((n, m), dtype=S.dtype), S]])
    post = np.linalg.qr(pre.T, mode="r").T
    L_y, Kb, S_post = post[:m, :m], post[m:, :m], post[m:, m:]

    w = np.linalg.solve(L_y, y)  # whitened innovation: L_y^-1 y
    nis = float(w @ w)
    logdet = 2 * float(np.log(np.abs(np.diagonal(L_y))).sum())
    log_likelihood = -0.5 * (nis + logdet + m * _LOG_2PI)
    return UpdateResult(x + Kb @ w, S_post, nis, log_likelihood)


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
) -> UpdateResult:
    """Condition on ``z`` with ``h`` linearized at ``x`` (``P``, ``R`` factors if square-root)."""
    H = np.asarray(jac_h(x), dtype=P.dtype)
    y = np.asarray(residual(z, np.asarray(h(x), dtype=P.dtype)), dtype=P.dtype)
    return (_sqrt_correct if square_root else _correct)(x, P, y, H, R)


# ---- Batch filtering -----------------------------------------------------------------------

# One filter step from (x, C) with C the covariance or its factor:
# returns (x_pred, C_pred, x_post, C_post, nis, log_likelihood, transition_matrix).
_Step = Callable[[int, Array, Array], tuple[Array, Array, Array, Array, float, float, Array]]


class _Run(NamedTuple):
    result: FilterResult[Array]
    jacobians: Array


def _run_filter(x0: Array, C0: Array, T: int, step: _Step, square_root: bool) -> _Run:
    n, dtype = x0.shape[0], C0.dtype
    means = np.empty((T, n), dtype=dtype)
    covs = np.empty((T, n, n), dtype=dtype)
    predicted_means = np.empty((T, n), dtype=dtype)
    predicted_covs = np.empty((T, n, n), dtype=dtype)
    jacobians = np.empty((T, n, n), dtype=dtype)
    nis = np.empty(T, dtype=dtype)
    log_likelihood = 0.0

    def cov(C: Array) -> Array:
        return C @ C.T if square_root else C

    x, C = x0, C0
    for k in range(T):
        x_pred, C_pred, x, C, nis[k], ll, jacobians[k] = step(k, x, C)
        predicted_means[k], predicted_covs[k] = x_pred, cov(C_pred)
        means[k], covs[k] = x, cov(C)
        log_likelihood += ll

    result = FilterResult(
        means, covs, predicted_means, predicted_covs, nis, np.asarray(log_likelihood, dtype=dtype)
    )
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

        def step(
            k: int, x: Array, S: Array
        ) -> tuple[Array, Array, Array, Array, float, float, Array]:
            x_pred, S_pred = sqrt_predict(x, S, F, L_Q)
            return (x_pred, S_pred, *sqrt_update(x_pred, S_pred, zs[k], H, L_R), F)

        return _run_filter(x0, psd_factor(P0, "P0"), zs.shape[0], step, True).result

    def joseph_step(
        k: int, x: Array, P: Array
    ) -> tuple[Array, Array, Array, Array, float, float, Array]:
        x_pred, P_pred = predict(x, P, F, Q)
        return (x_pred, P_pred, *update(x_pred, P_pred, zs[k], H, R), F)

    return _run_filter(x0, P0, zs.shape[0], joseph_step, False).result


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

    def step(k: int, x: Array, C: Array) -> tuple[Array, Array, Array, Array, float, float, Array]:
        x_pred, C_pred, F = ekf_predict(x, C, Q_, float(dts[k]), f, jac_f, square_root)
        updated = ekf_update(x_pred, C_pred, zs[k], R_, h, jac_h, residual, square_root)
        return (x_pred, C_pred, *updated, F)

    run = _run_filter(x0, C0, zs.shape[0], step, square_root)
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
