"""Input handling shared by the filter classes."""

from __future__ import annotations

from types import ModuleType
from typing import TYPE_CHECKING, Literal, TypeVar

import numpy as np
from numpy.typing import ArrayLike

from kalman_py._typing import Array

if TYPE_CHECKING:
    import jax

Backend = Literal["numpy", "jax"]
Measurements = TypeVar("Measurements", Array, "jax.Array")


def jax_backend() -> ModuleType:
    try:
        from kalman_py.backends import jax_backend
    except ImportError as e:
        raise ImportError(
            "the JAX backend needs JAX; install it with `pip install kalman-py[jax]`"
        ) from e
    return jax_backend


def as_float_arrays(*arrays: ArrayLike) -> list[Array]:
    """Convert to arrays of one common float dtype, keeping float32 if all inputs are float32."""
    converted = [np.asarray(a) for a in arrays]
    dtype = np.result_type(*converted)
    if not np.issubdtype(dtype, np.floating):
        dtype = np.dtype(np.float64)
    return [a.astype(dtype, copy=True) for a in converted]


def check_shape(name: str, a: Array, shape: tuple[int, ...]) -> None:
    if a.shape != shape:
        raise ValueError(f"{name} must have shape {shape}, got {a.shape}")


def check_measurements(zs: Measurements, dim_z: int) -> Measurements:
    """Validate a ``(T, dim_z)`` measurement array; a 1-D array is accepted when ``dim_z == 1``."""
    if zs.ndim == 1 and dim_z == 1:
        zs = zs.reshape(-1, 1)
    if zs.ndim != 2 or zs.shape[1] != dim_z:
        raise ValueError(f"zs must have shape (T, {dim_z}), got {zs.shape}")
    return zs


def check_filter_result(means_shape: tuple[int, ...], dim_x: int) -> None:
    if len(means_shape) != 2 or means_shape[1] != dim_x:
        raise ValueError(
            f"result does not match this filter: expected means of shape (T, {dim_x}), "
            f"got {means_shape}"
        )
