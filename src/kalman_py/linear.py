"""Linear Kalman filter."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal, overload

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
from kalman_py.result import FilterResult, SmootherResult

if TYPE_CHECKING:
    import jax


class KalmanFilter:
    """Linear Kalman filter for ``x_k = F x_{k-1} + w``, ``z_k = H x_k + v``.

    ``w ~ N(0, Q)`` and ``v ~ N(0, R)``. ``(x0, P0)`` is the prior *before* the first
    prediction, so every measurement is preceded by a predict step (the FilterPy
    convention).

    Use [`predict`][kalman_py.KalmanFilter.predict] and
    [`update`][kalman_py.KalmanFilter.update] in real-time loops, or
    [`filter`][kalman_py.KalmanFilter.filter] to process a whole measurement sequence at once,
    then [`smooth`][kalman_py.KalmanFilter.smooth] to refine it with future data.

    ``square_root=True`` propagates a factor ``S`` of the covariance (``P = S S'``) with QR
    decompositions instead of using the Joseph form. The covariance then stays positive
    semi-definite even for extremely precise sensors, at the cost of a QR per step. In this
    mode ``P`` is derived from the factor after every step, so assigning to it has no effect.
    """

    def __init__(
        self,
        F: ArrayLike,
        H: ArrayLike,
        Q: ArrayLike,
        R: ArrayLike,
        x0: ArrayLike,
        P0: ArrayLike,
        *,
        square_root: bool = False,
    ) -> None:
        self.F, self.H, self.Q, self.R, self.x0, self.P0 = as_float_arrays(F, H, Q, R, x0, P0)
        if self.H.ndim != 2:
            raise ValueError(f"H must be 2-D, got shape {self.H.shape}")
        m, n = self.H.shape
        check_shape("F", self.F, (n, n))
        check_shape("Q", self.Q, (n, n))
        check_shape("R", self.R, (m, m))
        check_shape("x0", self.x0, (n,))
        check_shape("P0", self.P0, (n, n))
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
        return int(self.H.shape[0])

    def _override(self, name: str, value: ArrayLike, shape: tuple[int, ...]) -> Array:
        """A per-call replacement for a model matrix, in the filter's dtype."""
        array = np.asarray(value, dtype=self.P0.dtype)
        check_shape(name, array, shape)
        return array

    def predict(self, F: ArrayLike | None = None, Q: ArrayLike | None = None) -> None:
        """Advance the current estimate ``(x, P)`` by one time step.

        ``F`` and ``Q`` replace the model's matrices for this step only, e.g. for a varying
        time step.
        """
        n = (self.dim_x, self.dim_x)
        F_ = self.F if F is None else self._override("F", F, n)
        if self.square_root:
            L_Q = (
                self._L_Q if Q is None else numpy_backend.psd_factor(self._override("Q", Q, n), "Q")
            )
            self.x, self._S = numpy_backend.sqrt_predict(self.x, self._S, F_, L_Q)
            self.P = self._S @ self._S.T
        else:
            Q_ = self.Q if Q is None else self._override("Q", Q, n)
            self.x, self.P = numpy_backend.predict(self.x, self.P, F_, Q_)

    def update(self, z: ArrayLike, H: ArrayLike | None = None, R: ArrayLike | None = None) -> None:
        """Condition the current estimate on one measurement.

        ``H`` and ``R`` replace the measurement model for this update only, e.g. to fuse sensors
        that measure different things: ``H`` may then have any number of rows ``k``, with ``z``
        of length ``k`` and ``R`` of shape ``(k, k)`` (``R`` is required when ``k`` differs from
        the model's measurement dimension).
        """
        H_ = self.H if H is None else np.asarray(H, dtype=self.P0.dtype)
        if H_.ndim != 2 or H_.shape[1] != self.dim_x:
            raise ValueError(f"H must have shape (k, {self.dim_x}), got {H_.shape}")
        k = H_.shape[0]
        if R is None and k != self.dim_z:
            raise ValueError(f"an H with {k} rows needs its own R of shape ({k}, {k})")
        z_arr = np.asarray(z, dtype=self.P.dtype).reshape(-1)
        check_shape("z", z_arr, (k,))
        if self.square_root:
            L_R = (
                self._L_R
                if R is None
                else numpy_backend.psd_factor(self._override("R", R, (k, k)), "R")
            )
            self.x, self._S, _, _ = numpy_backend.sqrt_update(self.x, self._S, z_arr, H_, L_R)
            self.P = self._S @ self._S.T
        else:
            R_ = self.R if R is None else self._override("R", R, (k, k))
            self.x, self.P, _, _ = numpy_backend.update(
                self.x, self.P, z_arr, H_, R_, self._identity
            )

    @overload
    def filter(self, zs: ArrayLike, backend: Literal["numpy"] = ...) -> FilterResult[Array]: ...
    @overload
    def filter(
        self, zs: ArrayLike | jax.Array, backend: Literal["jax"]
    ) -> FilterResult[jax.Array]: ...

    def filter(
        self, zs: ArrayLike | jax.Array, backend: Backend = "numpy"
    ) -> FilterResult[Array] | FilterResult[jax.Array]:
        """Filter a measurement sequence of shape ``(T, dim_z)``, starting from ``(x0, P0)``.

        This does not change the step-by-step state ``(x, P)``. A 1-D ``zs`` is accepted when
        ``dim_z == 1``. With ``backend="jax"`` the result holds JAX arrays, and JAX arrays
        passed as ``zs`` stay on their device.
        """
        if backend == "numpy":
            zs_np = check_measurements(np.asarray(zs, dtype=self.P0.dtype), self.dim_z)
            return numpy_backend.kalman_filter(
                self.F, self.H, self.Q, self.R, self.x0, self.P0, zs_np, self.square_root
            )
        if backend == "jax":
            jb = jax_backend()
            zs_jax = check_measurements(jb.asarray(zs, self.P0.dtype), self.dim_z)
            result: FilterResult[jax.Array] = jb.kalman_filter(
                self.F, self.H, self.Q, self.R, self.x0, self.P0, zs_jax, self.square_root
            )
            return result
        raise ValueError(f"unknown backend {backend!r}; expected 'numpy' or 'jax'")

    @overload
    def smooth(self, result: FilterResult[Array]) -> SmootherResult[Array]: ...
    @overload
    def smooth(self, result: FilterResult[jax.Array]) -> SmootherResult[jax.Array]: ...

    def smooth(
        self, result: FilterResult[Array] | FilterResult[jax.Array]
    ) -> SmootherResult[Array] | SmootherResult[jax.Array]:
        """Run the RTS smoother over the output of [`filter`][kalman_py.KalmanFilter.filter],
        on the same backend."""
        check_filter_result(result.means.shape, self.dim_x)
        if isinstance(result.means, np.ndarray):
            return numpy_backend.rts_smoother(self.F, result)
        smoothed: SmootherResult[jax.Array] = jax_backend().rts_smoother(self.F, result)
        return smoothed
