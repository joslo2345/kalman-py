"""JAX backend (optional; requires the ``jax`` extra).

Same algorithms as the NumPy backend, with the time loop written as ``jax.lax.scan`` so it
JIT-compiles. The jitted functions live at module level and take the model matrices as
arguments, so compiled code is reused by every filter with the same shapes and dtypes instead
of being recompiled per instance.

JAX computes in float32 unless ``jax_enable_x64`` is on; float64 inputs then get truncated, and
JAX warns about it.

Kalman filters typically work on tiny matrices, where each XLA dot or LAPACK call costs far more
in fixed overhead than in arithmetic. So for small shapes (known at trace time) products are
written as broadcast-multiply-sum and the innovation solve as an unrolled Cholesky, which XLA
fuses into straight-line code: about 3x faster per step for a 4-state / 2-measurement model.
Larger shapes fall back to ``@`` and LAPACK.
"""

from __future__ import annotations

import dataclasses
import math
from collections.abc import Callable
from functools import partial
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from numpy.typing import ArrayLike

from kalman_py.backends.numpy_backend import psd_factor
from kalman_py.result import ExtendedFilterResult, FilterResult, SmootherResult

_LOG_2PI = math.log(2 * math.pi)


def _register_pytree(cls: type) -> None:
    # Results cross jit boundaries, so JAX must know how to flatten them into arrays.
    names = [f.name for f in dataclasses.fields(cls)]
    jax.tree_util.register_pytree_node(
        cls,
        lambda r: ([getattr(r, name) for name in names], None),
        lambda _, leaves: cls(*leaves),
    )


_register_pytree(FilterResult)
_register_pytree(ExtendedFilterResult)
_register_pytree(SmootherResult)


def asarray(a: ArrayLike | jax.Array, dtype: np.dtype[Any]) -> jax.Array:
    return jnp.asarray(a, dtype=dtype)


# Above these sizes XLA's dot and LAPACK calls beat the fused code (measured on CPU: equal at
# n = 15, 4x slower at n = 50 for products).
_FUSED_MATMUL_MAX_DIM = 16
_UNROLLED_CHOLESKY_MAX_DIM = 8


def _mm(A: jax.Array, B: jax.Array) -> jax.Array:
    """Matrix-matrix or matrix-vector product."""
    if max(A.shape + B.shape) > _FUSED_MATMUL_MAX_DIM:
        return A @ B
    if B.ndim == 1:
        return (A * B).sum(-1)
    return (A[:, :, None] * B[None, :, :]).sum(1)


def _solve_spd(S: jax.Array, B: jax.Array) -> tuple[jax.Array, jax.Array]:
    """Return ``S^-1 B`` and ``log det S`` for a symmetric positive-definite ``S``."""
    m = S.shape[0]
    if m > _UNROLLED_CHOLESKY_MAX_DIM:
        return jnp.linalg.solve(S, B), jnp.linalg.slogdet(S)[1]

    # Cholesky S = L L', then forward and back substitution, as scalar expressions.
    L: list[list[Any]] = [[None] * m for _ in range(m)]
    for j in range(m):
        L[j][j] = jnp.sqrt(S[j, j] - sum((L[j][k] ** 2 for k in range(j)), start=0.0))
        for i in range(j + 1, m):
            L[i][j] = (S[i, j] - sum((L[i][k] * L[j][k] for k in range(j)), start=0.0)) / L[j][j]
    Y: list[Any] = [None] * m
    for i in range(m):
        Y[i] = (B[i] - sum((L[i][k] * Y[k] for k in range(i)), start=0.0)) / L[i][i]
    X: list[Any] = [None] * m
    for i in reversed(range(m)):
        X[i] = (Y[i] - sum((L[k][i] * X[k] for k in range(i + 1, m)), start=0.0)) / L[i][i]
    logdet = 2 * jnp.log(jnp.stack([L[i][i] for i in range(m)])).sum()
    return jnp.stack(X), logdet


def _symmetrize(P: jax.Array) -> jax.Array:
    return 0.5 * (P + P.T)


def _correct(
    x: jax.Array, P: jax.Array, y: jax.Array, H: jax.Array, R: jax.Array
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    # Mirrors numpy_backend._correct; see the comments there.
    HP = _mm(H, P)
    S = _symmetrize(_mm(HP, H.T) + R)
    sol, logdet = _solve_spd(S, jnp.column_stack((HP, y)))
    K = sol[:, :-1].T
    nis = (y * sol[:, -1]).sum()

    I_KH = jnp.eye(x.shape[0], dtype=P.dtype) - _mm(K, H)
    P_post = _symmetrize(_mm(_mm(I_KH, P), I_KH.T) + _mm(_mm(K, R), K.T))

    log_likelihood = -0.5 * (nis + logdet + y.shape[0] * _LOG_2PI)
    return x + _mm(K, y), P_post, nis, log_likelihood


def _sqrt_predict_cov(S: jax.Array, F: jax.Array, L_Q: jax.Array) -> jax.Array:
    # Mirrors numpy_backend.sqrt_predict; see the comments there.
    r = jnp.linalg.qr(jnp.vstack((_mm(F, S).T, L_Q.T)), mode="r")
    return r.T


def _sqrt_correct(
    x: jax.Array, S: jax.Array, y: jax.Array, H: jax.Array, L_R: jax.Array
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    # Mirrors numpy_backend._sqrt_correct; see the comments there.
    m, n = L_R.shape[0], S.shape[0]
    pre = jnp.block([[L_R, _mm(H, S)], [jnp.zeros((n, m), dtype=S.dtype), S]])
    post = jnp.linalg.qr(pre.T, mode="r").T
    L_y, Kb, S_post = post[:m, :m], post[m:, :m], post[m:, m:]

    w = jax.scipy.linalg.solve_triangular(L_y, y, lower=True)
    nis = (w * w).sum()
    logdet = 2 * jnp.log(jnp.abs(jnp.diagonal(L_y))).sum()
    log_likelihood = -0.5 * (nis + logdet + m * _LOG_2PI)
    return x + _mm(Kb, w), S_post, nis, log_likelihood


def _predict_cov(C: jax.Array, F: jax.Array, Qc: jax.Array, square_root: bool) -> jax.Array:
    """Predicted covariance, or its factor in square-root form (then ``Qc`` factors Q)."""
    if square_root:
        return _sqrt_predict_cov(C, F, Qc)
    return _symmetrize(_mm(_mm(F, C), F.T) + Qc)


def _correct_any(
    x: jax.Array, C: jax.Array, y: jax.Array, H: jax.Array, Rc: jax.Array, square_root: bool
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    return (_sqrt_correct if square_root else _correct)(x, C, y, H, Rc)


def _cov(C: jax.Array, square_root: bool) -> jax.Array:
    return _mm(C, C.T) if square_root else C


def jacobian(fn: Callable[..., Any]) -> Callable[..., jax.Array]:
    """Jitted forward-mode Jacobian of ``fn`` with respect to its first argument."""
    return jax.jit(jax.jacfwd(fn))


def _subtract(a: jax.Array, b: jax.Array) -> jax.Array:
    return a - b


@partial(jax.jit, static_argnames=("square_root",))
def _kalman_filter(
    F: jax.Array,
    H: jax.Array,
    Qc: jax.Array,
    Rc: jax.Array,
    x0: jax.Array,
    C0: jax.Array,
    zs: jax.Array,
    square_root: bool,
) -> FilterResult[jax.Array]:
    # Qc, Rc, C0 are Q, R, P0, or their factors in square-root form.
    def step(
        carry: tuple[jax.Array, jax.Array], z: jax.Array
    ) -> tuple[tuple[jax.Array, jax.Array], tuple[jax.Array, ...]]:
        x, C = carry
        x_pred, C_pred = _mm(F, x), _predict_cov(C, F, Qc, square_root)
        x, C, nis, ll = _correct_any(x_pred, C_pred, z - _mm(H, x_pred), H, Rc, square_root)
        cov, pred_cov = _cov(C, square_root), _cov(C_pred, square_root)
        return (x, C), (x, cov, x_pred, pred_cov, nis, ll)

    _, (means, covs, pred_means, pred_covs, nis, lls) = jax.lax.scan(step, (x0, C0), zs)
    return FilterResult(means, covs, pred_means, pred_covs, nis, lls.sum())


@partial(jax.jit, static_argnames=("f", "h", "jac_f", "jac_h", "residual", "square_root"))
def _ekf_filter(
    Qc: jax.Array,
    Rc: jax.Array,
    x0: jax.Array,
    C0: jax.Array,
    zs: jax.Array,
    dts: jax.Array,
    *,
    f: Callable[[jax.Array, jax.Array], jax.Array],
    h: Callable[[jax.Array], jax.Array],
    jac_f: Callable[[jax.Array, jax.Array], jax.Array] | None,
    jac_h: Callable[[jax.Array], jax.Array] | None,
    residual: Callable[[jax.Array, jax.Array], jax.Array] | None,
    square_root: bool,
) -> ExtendedFilterResult[jax.Array]:
    # The model functions are static: jit caches compiled code per function object, so passing
    # the same module-level functions to many filters compiles once.
    jac_f_ = jac_f or jax.jacfwd(f)
    jac_h_ = jac_h or jax.jacfwd(h)
    residual_ = residual or _subtract

    def step(
        carry: tuple[jax.Array, jax.Array], inputs: tuple[jax.Array, jax.Array]
    ) -> tuple[tuple[jax.Array, jax.Array], tuple[jax.Array, ...]]:
        x, C = carry
        z, dt = inputs
        F = jnp.asarray(jac_f_(x, dt), dtype=C.dtype)
        x_pred = jnp.asarray(f(x, dt), dtype=C.dtype)
        C_pred = _predict_cov(C, F, Qc, square_root)
        H = jnp.asarray(jac_h_(x_pred), dtype=C.dtype)
        y = jnp.asarray(residual_(z, jnp.asarray(h(x_pred), dtype=C.dtype)), dtype=C.dtype)
        x, C, nis, ll = _correct_any(x_pred, C_pred, y, H, Rc, square_root)
        cov, pred_cov = _cov(C, square_root), _cov(C_pred, square_root)
        return (x, C), (x, cov, x_pred, pred_cov, nis, ll, F)

    _, (means, covs, pred_means, pred_covs, nis, lls, jacobians) = jax.lax.scan(
        step, (x0, C0), (zs, dts)
    )
    return ExtendedFilterResult(means, covs, pred_means, pred_covs, nis, lls.sum(), jacobians)


@jax.jit
def _rts_smoother(F: jax.Array, result: FilterResult[jax.Array]) -> SmootherResult[jax.Array]:
    # F is either the transition matrix or a (T, n, n) stack of per-step Jacobians (EKF).
    per_step = F.ndim == 3

    def step(
        carry: tuple[jax.Array, jax.Array], filtered: tuple[jax.Array, ...]
    ) -> tuple[tuple[jax.Array, jax.Array], tuple[jax.Array, jax.Array]]:
        x_next, P_next = carry
        x, P, x_pred, P_pred, *rest = filtered
        F_k = rest[0] if per_step else F
        # LU rather than _solve_spd: P_pred can be near-singular when Q is.
        G = jnp.linalg.solve(P_pred, _mm(F_k, P)).T
        x_s = x + _mm(G, x_next - x_pred)
        P_s = _symmetrize(P + _mm(_mm(G, P_next - P_pred), G.T))
        return (x_s, P_s), (x_s, P_s)

    last = (result.means[-1], result.covs[-1])
    filtered = (
        result.means[:-1],
        result.covs[:-1],
        result.predicted_means[1:],
        result.predicted_covs[1:],
    ) + ((F[1:],) if per_step else ())
    _, (means, covs) = jax.lax.scan(step, last, filtered, reverse=True)
    return SmootherResult(
        jnp.concatenate([means, last[0][None]]), jnp.concatenate([covs, last[1][None]])
    )


def kalman_filter(
    F: ArrayLike,
    H: ArrayLike,
    Q: ArrayLike,
    R: ArrayLike,
    x0: ArrayLike,
    P0: ArrayLike,
    zs: jax.Array,
    square_root: bool = False,
) -> FilterResult[jax.Array]:
    """Run predict + update for every row of ``zs``, starting from the prior ``(x0, P0)``."""
    dtype = zs.dtype
    Q, R, P0 = _covariance_inputs(Q, R, P0, dtype, square_root)
    F_, H_, Q_, R_, x0_, P0_ = (asarray(a, dtype) for a in (F, H, Q, R, x0, P0))
    return _kalman_filter(  # type: ignore[no-any-return]
        F_, H_, Q_, R_, x0_, P0_, zs, square_root=square_root
    )


def _covariance_inputs(
    Q: ArrayLike, R: ArrayLike, P0: ArrayLike, dtype: np.dtype[Any], square_root: bool
) -> tuple[ArrayLike, ArrayLike, ArrayLike]:
    """In square-root form, replace Q, R and P0 by factors (computed once, outside jit)."""
    if not square_root:
        return Q, R, P0
    return tuple(  # type: ignore[return-value]
        psd_factor(np.asarray(a, dtype=dtype), name) for a, name in ((Q, "Q"), (R, "R"), (P0, "P0"))
    )


def rts_smoother(F: ArrayLike, result: FilterResult[jax.Array]) -> SmootherResult[jax.Array]:
    """Rauch-Tung-Striebel backward pass over the output of :func:`kalman_filter`."""
    return _rts_smoother(asarray(F, result.means.dtype), result)  # type: ignore[no-any-return]


def ekf_filter(
    Q: ArrayLike,
    R: ArrayLike,
    x0: ArrayLike,
    P0: ArrayLike,
    zs: jax.Array,
    dts: ArrayLike,
    f: Callable[[jax.Array, jax.Array], jax.Array],
    h: Callable[[jax.Array], jax.Array],
    jac_f: Callable[[jax.Array, jax.Array], jax.Array] | None,
    jac_h: Callable[[jax.Array], jax.Array] | None,
    residual: Callable[[jax.Array, jax.Array], jax.Array] | None,
    square_root: bool = False,
) -> ExtendedFilterResult[jax.Array]:
    """Extended Kalman filter; missing Jacobians are derived with ``jax.jacfwd``.

    ``f``, ``h``, the Jacobians and ``residual`` must be written with ``jax.numpy`` so JAX can
    trace them.
    """
    dtype = zs.dtype
    Q, R, P0 = _covariance_inputs(Q, R, P0, dtype, square_root)
    Q_, R_, x0_, P0_, dts_ = (asarray(a, dtype) for a in (Q, R, x0, P0, dts))
    return _ekf_filter(  # type: ignore[no-any-return]
        Q_,
        R_,
        x0_,
        P0_,
        zs,
        dts_,
        f=f,
        h=h,
        jac_f=jac_f,
        jac_h=jac_h,
        residual=residual,
        square_root=square_root,
    )
