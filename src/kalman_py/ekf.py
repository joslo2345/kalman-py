"""Extended Kalman filter."""

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
from kalman_py.result import ExtendedFilterResult, SmootherResult

if TYPE_CHECKING:
    import jax

TransitionFn = Callable[[Any, Any], Any]
MeasurementFn = Callable[[Any], Any]
ResidualFn = Callable[[Any, Any], Any]


class ExtendedKalmanFilter:
    """Extended Kalman filter for ``x_k = f(x_{k-1}, dt) + w``, ``z_k = h(x_k) + v``.

    ``w ~ N(0, Q)`` and ``v ~ N(0, R)``; ``(x0, P0)`` is the prior before the first prediction.

    Jacobians ``jac_f(x, dt)`` and ``jac_h(x)`` are optional: when omitted they are derived
    automatically with ``jax.jacfwd``, which needs JAX and requires ``f`` and ``h`` to be written
    with ``jax.numpy``. The JAX backend also traces ``f``, ``h`` and ``residual_z``, so they must
    be ``jax.numpy`` code there too. ``residual_z(z, z_pred)`` replaces ``z - z_pred``, e.g. to
    wrap angle differences into [-pi, pi).

    For the JAX backend, pass module-level functions rather than fresh lambdas: compiled code is
    cached per function object.

    ``square_root=True`` selects the square-root covariance form; see
    :class:`~kalman_py.KalmanFilter`.
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
        jac_f: TransitionFn | None = None,
        jac_h: MeasurementFn | None = None,
        residual_z: ResidualFn | None = None,
        square_root: bool = False,
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

        self.f, self.h = f, h
        self.jac_f, self.jac_h, self.residual_z = jac_f, jac_h, residual_z
        if jac_f is None or jac_h is None:
            try:
                jb = jax_backend()
            except ImportError as e:
                raise ImportError(
                    "pass jac_f and jac_h, or install JAX (`pip install kalman-py[jax]`) to "
                    "derive them automatically"
                ) from e
        self._np_jac_f: TransitionFn = jac_f if jac_f is not None else jb.jacobian(f)
        self._np_jac_h: MeasurementFn = jac_h if jac_h is not None else jb.jacobian(h)
        self._np_residual: ResidualFn = residual_z if residual_z is not None else np.subtract

        self.square_root = square_root
        self._identity = np.eye(n, dtype=self.P0.dtype)
        if square_root:
            self._L_Q = numpy_backend.psd_factor(self.Q, "Q")
            self._L_R = numpy_backend.psd_factor(self.R, "R")
            self._S = numpy_backend.psd_factor(self.P0, "P0")

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
        if self.square_root:
            self.x, self._S, _ = numpy_backend.ekf_predict(
                self.x, self._S, self._L_Q, dt, self.f, self._np_jac_f, square_root=True
            )
            self.P = self._S @ self._S.T
        else:
            self.x, self.P, _ = numpy_backend.ekf_predict(
                self.x, self.P, self.Q, dt, self.f, self._np_jac_f
            )

    def update(self, z: ArrayLike) -> None:
        """Condition the current estimate on one measurement."""
        z_arr = np.asarray(z, dtype=self.P.dtype).reshape(-1)
        check_shape("z", z_arr, (self.dim_z,))
        if self.square_root:
            self.x, self._S, _, _ = numpy_backend.ekf_update(
                self.x,
                self._S,
                z_arr,
                self._L_R,
                self.h,
                self._np_jac_h,
                self._np_residual,
                square_root=True,
            )
            self.P = self._S @ self._S.T
        else:
            self.x, self.P, _, _ = numpy_backend.ekf_update(
                self.x,
                self.P,
                z_arr,
                self.R,
                self.h,
                self._np_jac_h,
                self._np_residual,
                identity=self._identity,
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
    ) -> ExtendedFilterResult[Array]: ...
    @overload
    def filter(
        self, zs: ArrayLike | jax.Array, dt: ArrayLike, backend: Literal["jax"]
    ) -> ExtendedFilterResult[jax.Array]: ...

    def filter(
        self, zs: ArrayLike | jax.Array, dt: ArrayLike, backend: Backend = "numpy"
    ) -> ExtendedFilterResult[Array] | ExtendedFilterResult[jax.Array]:
        """Filter a ``(T, dim_z)`` measurement sequence, starting from ``(x0, P0)``.

        ``dt`` is the time step before each measurement: a scalar, or one value per
        measurement for irregular sampling. This does not change the step-by-step state.
        """
        if backend == "numpy":
            zs_np = check_measurements(np.asarray(zs, dtype=self.P0.dtype), self.dim_z)
            return numpy_backend.ekf_filter(
                self.Q,
                self.R,
                self.x0,
                self.P0,
                zs_np,
                self._time_steps(dt, zs_np.shape[0]),
                self.f,
                self.h,
                self._np_jac_f,
                self._np_jac_h,
                self._np_residual,
                self.square_root,
            )
        if backend == "jax":
            jb = jax_backend()
            zs_jax = check_measurements(jb.asarray(zs, self.P0.dtype), self.dim_z)
            result: ExtendedFilterResult[jax.Array] = jb.ekf_filter(
                self.Q,
                self.R,
                self.x0,
                self.P0,
                zs_jax,
                self._time_steps(dt, zs_jax.shape[0]),
                self.f,
                self.h,
                self.jac_f,
                self.jac_h,
                self.residual_z,
                self.square_root,
            )
            return result
        raise ValueError(f"unknown backend {backend!r}; expected 'numpy' or 'jax'")

    @overload
    def smooth(self, result: ExtendedFilterResult[Array]) -> SmootherResult[Array]: ...
    @overload
    def smooth(self, result: ExtendedFilterResult[jax.Array]) -> SmootherResult[jax.Array]: ...

    def smooth(
        self, result: ExtendedFilterResult[Array] | ExtendedFilterResult[jax.Array]
    ) -> SmootherResult[Array] | SmootherResult[jax.Array]:
        """Extended RTS smoother over the output of :meth:`filter`, on the same backend."""
        check_filter_result(result.means.shape, self.dim_x)
        if isinstance(result.means, np.ndarray):
            return numpy_backend.rts_smoother(result.transition_jacobians, result)
        smoothed: SmootherResult[jax.Array] = jax_backend().rts_smoother(
            result.transition_jacobians, result
        )
        return smoothed
