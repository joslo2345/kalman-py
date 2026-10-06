# Backends: NumPy and JAX

Every filter takes `backend="numpy"` (the default) or `backend="jax"` in `filter`, and `smooth`
uses the backend that produced its input.

| | NumPy | JAX |
|---|---|---|
| Install | always available | `pip install kalman-py[jax]` |
| Step-by-step API | yes | no (use NumPy) |
| Batch speed (2-D tracking) | ~2.8 µs/step | ~0.6 µs/step |
| Results | NumPy arrays | JAX arrays, left on the device |

```python
import jax
import numpy as np

from kalman_py import KalmanFilter

jax.config.update("jax_enable_x64", True)

dt = 1.0
F = np.array([[1.0, dt], [0.0, 1.0]])
kf = KalmanFilter(F, [[1.0, 0.0]], 0.01 * np.eye(2), [[1.0]], x0=[0.0, 1.0], P0=np.eye(2))
zs = np.arange(1, 1001, dtype=float)[:, None]

result = kf.filter(zs, backend="jax")
jax.block_until_ready(result.means)  # JAX runs asynchronously
print(type(result.means).__name__)
```

## Precision

JAX computes in float32 unless 64-bit mode is on. With float64 inputs and 64-bit mode off, JAX
truncates them and warns. Turn it on at startup for results that match the NumPy backend:

```python
jax.config.update("jax_enable_x64", True)
```

float32 problems stay float32 on both backends either way.

## Compilation

The first call for a given set of shapes compiles; later calls with the same shapes and dtypes
reuse the compiled code, even from different filter objects. When timing, warm up first and
wait for the result with `jax.block_until_ready`.

For the EKF and UKF, the model functions are part of what's compiled. Define them once at module
level and pass the same function objects to every filter: a new lambda per filter is a new
function and compiles again.

## Model functions

On the JAX backend, `f`, `h` and `residual_z` are traced by JAX, so they must use `jax.numpy`
(`jnp.array`, `jnp.hypot`, ...) rather than NumPy calls that convert to concrete arrays. Code
written with `jax.numpy` also works on the NumPy backend, so one version serves both.
