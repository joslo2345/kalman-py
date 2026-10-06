"""Unscented Kalman filter."""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any, Literal, overload

import numpy as np
from numpy.typing import ArrayLike

from kalman_py._common import (
    Backend,
    as_float_arrays,
    check_filter_result,
    check_measurements,
    check_shape,
    jax_backend,
)
from kalman_py._typing import Array
from kalman_py.backends import numpy_backend
from kalman_py.result import SmootherResult, UnscentedFilterResult

if TYPE_CHECKING:
    import jax

TransitionFn = Callable[[Any, Any], Any]
MeasurementFn = Callable[[Any], Any]
ResidualFn = Callable[[Any, Any], Any]


class UnscentedKalmanFilter:
    """Unscented Kalman filter for ``x_k = f(x_{k-1}, dt) + w``, ``z_k = h(x_k) + v``.

    ``w ~ N(0, Q)`` and ``v ~ N(0, R)``; ``(x0, P0)`` is the prior before the first prediction.
    No Jacobians are needed: ``f`` and ``h`` are evaluated at Van der Merwe scaled sigma points.

    The defaults ``alpha=1, beta=2, kappa=0`` keep every covariance weight nonnegative, so the
    predicted covariance cannot become indefinite. Small ``alpha`` (e.g. 1e-3, common in the
    literature) concentrates the points but makes the central weight strongly negative.

    ``residual_z(z, z_pred)`` replaces ``z - z_pred``, e.g. to wrap angles into [-pi, pi). It is
    also used to average the sigma-point measurements, so angle measurements near +-pi need no
    separate mean function. Sigma points are redrawn from the predicted distribution before
    each update, so the process noise is reflected in the measurement prediction.

    The JAX backend vmaps ``f``, ``h`` and ``residual_z`` over sigma points, so they must be
    ``jax.numpy`` code there; pass module-level functions so compiled code is reused.
    """

    def __init__(
        self,
        f: TransitionFn,
        h: MeasurementFn,
        Q: ArrayLike,
        R: ArrayLike,
        x0: ArrayLike,
        P0: ArrayLike,
        *,
        alpha: float = 1.0,
        beta: float = 2.0,
        kappa: float = 0.0,
        residual_z: ResidualFn | None = None,
    ) -> None:
        self.Q, self.R, self.x0, self.P0 = as_float_arrays(Q, R, x0, P0)
        if self.x0.ndim != 1:
            raise ValueError(f"x0 must be 1-D, got shape {self.x0.shape}")
        if self.R.ndim != 2:
            raise ValueError(f"R must be 2-D, got shape {self.R.shape}")
        n, m = self.x0.shape[0], self.R.shape[0]
        check_shape("Q", self.Q, (n, n))
        check_shape("R", self.R, (m, m))
        check_shape("P0", self.P0, (n, n))

        self.f, self.h, self.residual_z = f, h, residual_z
        self.alpha, self.beta, self.kappa = alpha, beta, kappa
        self._weights = numpy_backend.sigma_weights(n, alpha, beta, kappa, self.P0.dtype)
        self._np_residual: ResidualFn = residual_z if residual_z is not None else np.subtract

        self.x = self.x0.copy()
        self.P = self.P0.copy()

    @property
    def dim_x(self) -> int:
        return int(self.x0.shape[0])

    @property
    def dim_z(self) -> int:
        return int(self.R.shape[0])

    def predict(self, dt: float) -> None:
        """Advance the current estimate ``(x, P)`` by ``dt``."""
        self.x, self.P, _ = numpy_backend.ukf_predict(
            self.x, self.P, self.Q, dt, self.f, self._weights
        )

    def update(self, z: ArrayLike) -> None:
        """Condition the current estimate on one measurement."""
        z_arr = np.asarray(z, dtype=self.P.dtype).reshape(-1)
        check_shape("z", z_arr, (self.dim_z,))
        self.x, self.P, _, _ = numpy_backend.ukf_update(
            self.x, self.P, z_arr, self.R, self.h, self._np_residual, self._weights
        )

    def _time_steps(self, dt: ArrayLike, T: int) -> Array:
        dts = np.asarray(dt, dtype=self.P0.dtype)
        if dts.ndim == 0:
            return np.full(T, dts, dtype=self.P0.dtype)
        if dts.shape != (T,):
            raise ValueError(f"dt must be a scalar or have shape ({T},), got {dts.shape}")
        return dts

    @overload
    def filter(
        self, zs: ArrayLike, dt: ArrayLike, backend: Literal["numpy"] = ...
    ) -> UnscentedFilterResult[Array]: ...
    @overload
    def filter(
        self, zs: ArrayLike | jax.Array, dt: ArrayLike, backend: Literal["jax"]
    ) -> UnscentedFilterResult[jax.Array]: ...

    def filter(
        self, zs: ArrayLike | jax.Array, dt: ArrayLike, backend: Backend = "numpy"
    ) -> UnscentedFilterResult[Array] | UnscentedFilterResult[jax.Array]:
        """Filter a ``(T, dim_z)`` measurement sequence, starting from ``(x0, P0)``.

        ``dt`` is the time step before each measurement: a scalar, or one value per
        measurement. This does not change the step-by-step state.
        """
        if backend == "numpy":
            zs_np = check_measurements(np.asarray(zs, dtype=self.P0.dtype), self.dim_z)
            return numpy_backend.ukf_filter(
                self.Q,
                self.R,
                self.x0,
                self.P0,
                zs_np,
                self._time_steps(dt, zs_np.shape[0]),
                self.f,
                self.h,
                self._np_residual,
                self._weights,
            )
        if backend == "jax":
            jb = jax_backend()
            zs_jax = check_measurements(jb.asarray(zs, self.P0.dtype), self.dim_z)
            result: UnscentedFilterResult[jax.Array] = jb.ukf_filter(
                self.Q,
                self.R,
                self.x0,
                self.P0,
                zs_jax,
                self._time_steps(dt, zs_jax.shape[0]),
                self.f,
                self.h,
                self.residual_z,
                self.alpha,
                self.beta,
                self.kappa,
            )
            return result
        raise ValueError(f"unknown backend {backend!r}; expected 'numpy' or 'jax'")

    @overload
    def smooth(self, result: UnscentedFilterResult[Array]) -> SmootherResult[Array]: ...
    @overload
    def smooth(self, result: UnscentedFilterResult[jax.Array]) -> SmootherResult[jax.Array]: ...

    def smooth(
        self, result: UnscentedFilterResult[Array] | UnscentedFilterResult[jax.Array]
    ) -> SmootherResult[Array] | SmootherResult[jax.Array]:
        """Unscented RTS smoother over the output of :meth:`filter`, on the same backend."""
        check_filter_result(result.means.shape, self.dim_x)
        if isinstance(result.means, np.ndarray):
            return numpy_backend.rts_smoother_from_cross(result.cross_covariances[1:], result)
        smoothed: SmootherResult[jax.Array] = jax_backend().rts_smoother_from_cross(
            result.cross_covariances[1:], result
        )
        return smoothed
