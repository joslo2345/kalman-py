"""NumPy backend.

All functions preserve the dtype of their inputs, so a float32 problem stays float32.
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
    P: Array
    nis: float
    log_likelihood: float


def _symmetrize(P: Array) -> Array:
    return 0.5 * (P + P.T)


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

    # Joseph form stays symmetric positive-definite under rounding, unlike P = (I - KH) P.
    I_KH = np.eye(x.shape[0], dtype=P.dtype) - K @ H
    P_post = _symmetrize(I_KH @ P @ I_KH.T + K @ R @ K.T)

    _, logdet = np.linalg.slogdet(S)
    log_likelihood = -0.5 * (nis + float(logdet) + y.shape[0] * _LOG_2PI)
    return UpdateResult(x + K @ y, P_post, nis, log_likelihood)


def ekf_predict(
    x: Array, P: Array, Q: Array, dt: float, f: TransitionFn, jac_f: TransitionFn
) -> tuple[Array, Array, Array]:
    """Propagate through ``f``, linearized at ``x``; also returns the Jacobian used."""
    F = np.asarray(jac_f(x, dt), dtype=P.dtype)
    x_pred = np.asarray(f(x, dt), dtype=P.dtype)
    return x_pred, _symmetrize(F @ P @ F.T + Q), F


def ekf_update(
    x: Array,
    P: Array,
    z: Array,
    R: Array,
    h: MeasurementFn,
    jac_h: MeasurementFn,
    residual: ResidualFn,
) -> UpdateResult:
    """Condition on ``z`` with ``h`` linearized at ``x``."""
    H = np.asarray(jac_h(x), dtype=P.dtype)
    y = np.asarray(residual(z, np.asarray(h(x), dtype=P.dtype)), dtype=P.dtype)
    return _correct(x, P, y, H, R)


def kalman_filter(
    F: Array, H: Array, Q: Array, R: Array, x0: Array, P0: Array, zs: Array
) -> FilterResult[Array]:
    """Run predict + update for every row of ``zs``, starting from the prior ``(x0, P0)``."""
    T, n = zs.shape[0], x0.shape[0]
    dtype = P0.dtype
    means = np.empty((T, n), dtype=dtype)
    covs = np.empty((T, n, n), dtype=dtype)
    predicted_means = np.empty((T, n), dtype=dtype)
    predicted_covs = np.empty((T, n, n), dtype=dtype)
    nis = np.empty(T, dtype=dtype)
    log_likelihood = 0.0

    x, P = x0, P0
    for k in range(T):
        x, P = predict(x, P, F, Q)
        predicted_means[k], predicted_covs[k] = x, P
        x, P, nis[k], ll = update(x, P, zs[k], H, R)
        means[k], covs[k] = x, P
        log_likelihood += ll

    return FilterResult(
        means, covs, predicted_means, predicted_covs, nis, np.asarray(log_likelihood, dtype=dtype)
    )


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
) -> ExtendedFilterResult[Array]:
    """Extended Kalman filter over every row of ``zs``; ``dts[k]`` is the step before ``zs[k]``."""
    T, n = zs.shape[0], x0.shape[0]
    dtype = P0.dtype
    means = np.empty((T, n), dtype=dtype)
    covs = np.empty((T, n, n), dtype=dtype)
    predicted_means = np.empty((T, n), dtype=dtype)
    predicted_covs = np.empty((T, n, n), dtype=dtype)
    jacobians = np.empty((T, n, n), dtype=dtype)
    nis = np.empty(T, dtype=dtype)
    log_likelihood = 0.0

    x, P = x0, P0
    for k in range(T):
        x, P, jacobians[k] = ekf_predict(x, P, Q, float(dts[k]), f, jac_f)
        predicted_means[k], predicted_covs[k] = x, P
        x, P, nis[k], ll = ekf_update(x, P, zs[k], R, h, jac_h, residual)
        means[k], covs[k] = x, P
        log_likelihood += ll

    return ExtendedFilterResult(
        means,
        covs,
        predicted_means,
        predicted_covs,
        nis,
        np.asarray(log_likelihood, dtype=dtype),
        jacobians,
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
