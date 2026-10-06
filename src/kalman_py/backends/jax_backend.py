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
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from numpy.typing import ArrayLike

from kalman_py.result import FilterResult, SmootherResult

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


def _predict(x: jax.Array, P: jax.Array, F: jax.Array, Q: jax.Array) -> tuple[jax.Array, jax.Array]:
    return _mm(F, x), _symmetrize(_mm(_mm(F, P), F.T) + Q)


def _update(
    x: jax.Array, P: jax.Array, z: jax.Array, H: jax.Array, R: jax.Array
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    # Mirrors numpy_backend.update; see the comments there.
    y = z - _mm(H, x)
    HP = _mm(H, P)
    S = _symmetrize(_mm(HP, H.T) + R)
    sol, logdet = _solve_spd(S, jnp.column_stack((HP, y)))
    K = sol[:, :-1].T
    nis = (y * sol[:, -1]).sum()

    I_KH = jnp.eye(x.shape[0], dtype=P.dtype) - _mm(K, H)
    P_post = _symmetrize(_mm(_mm(I_KH, P), I_KH.T) + _mm(_mm(K, R), K.T))

    log_likelihood = -0.5 * (nis + logdet + y.shape[0] * _LOG_2PI)
    return x + _mm(K, y), P_post, nis, log_likelihood


@jax.jit
def _kalman_filter(
    F: jax.Array,
    H: jax.Array,
    Q: jax.Array,
    R: jax.Array,
    x0: jax.Array,
    P0: jax.Array,
    zs: jax.Array,
) -> FilterResult[jax.Array]:
    def step(
        carry: tuple[jax.Array, jax.Array], z: jax.Array
    ) -> tuple[tuple[jax.Array, jax.Array], tuple[jax.Array, ...]]:
        x_pred, P_pred = _predict(*carry, F, Q)
        x, P, nis, ll = _update(x_pred, P_pred, z, H, R)
        return (x, P), (x, P, x_pred, P_pred, nis, ll)

    _, (means, covs, pred_means, pred_covs, nis, lls) = jax.lax.scan(step, (x0, P0), zs)
    return FilterResult(means, covs, pred_means, pred_covs, nis, lls.sum())


@jax.jit
def _rts_smoother(F: jax.Array, result: FilterResult[jax.Array]) -> SmootherResult[jax.Array]:
    def step(
        carry: tuple[jax.Array, jax.Array], filtered: tuple[jax.Array, ...]
    ) -> tuple[tuple[jax.Array, jax.Array], tuple[jax.Array, jax.Array]]:
        x_next, P_next = carry
        x, P, x_pred, P_pred = filtered
        # LU rather than _solve_spd: P_pred can be near-singular when Q is.
        G = jnp.linalg.solve(P_pred, _mm(F, P)).T
        x_s = x + _mm(G, x_next - x_pred)
        P_s = _symmetrize(P + _mm(_mm(G, P_next - P_pred), G.T))
        return (x_s, P_s), (x_s, P_s)

    last = (result.means[-1], result.covs[-1])
    filtered = (
        result.means[:-1],
        result.covs[:-1],
        result.predicted_means[1:],
        result.predicted_covs[1:],
    )
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
) -> FilterResult[jax.Array]:
    """Run predict + update for every row of ``zs``, starting from the prior ``(x0, P0)``."""
    dtype = zs.dtype
    F_, H_, Q_, R_, x0_, P0_ = (asarray(a, dtype) for a in (F, H, Q, R, x0, P0))
    return _kalman_filter(F_, H_, Q_, R_, x0_, P0_, zs)  # type: ignore[no-any-return]


def rts_smoother(F: ArrayLike, result: FilterResult[jax.Array]) -> SmootherResult[jax.Array]:
    """Rauch-Tung-Striebel backward pass over the output of :func:`kalman_filter`."""
    return _rts_smoother(asarray(F, result.means.dtype), result)  # type: ignore[no-any-return]
