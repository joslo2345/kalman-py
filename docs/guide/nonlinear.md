# Nonlinear filters

For `x_k = f(x_{k-1}, dt) + w` and `z_k = h(x_k) + v`, kalman-py has an extended Kalman filter,
which linearizes `f` and `h` with their Jacobians, and an unscented Kalman filter, which
propagates sigma points through them. Both take `f(x, dt)` and `h(x)`, a `dt` per step or one
for all, and both smooth.

```python
import jax
import jax.numpy as jnp
import numpy as np

from kalman_py import ExtendedKalmanFilter, UnscentedKalmanFilter

jax.config.update("jax_enable_x64", True)


def f(x, dt):  # 2-D constant velocity: (px, py, vx, vy)
    return jnp.array([x[0] + dt * x[2], x[1] + dt * x[3], x[2], x[3]])


def h(x):  # range and bearing from a sensor at the origin
    return jnp.array([jnp.hypot(x[0], x[1]), jnp.arctan2(x[1], x[0])])


def wrap_bearing(z, z_pred):
    d = jnp.asarray(z) - jnp.asarray(z_pred)
    return d.at[1].set((d[1] + jnp.pi) % (2 * jnp.pi) - jnp.pi)


model = dict(Q=0.01 * np.eye(4), R=np.diag([0.25, 1e-4]), x0=[100.0, 5.0, -1.0, 0.0], P0=np.eye(4))
zs = np.asarray(jax.vmap(h)(jnp.array([[100.0 - k, 5.0, -1.0, 0.0] for k in range(1, 51)])))

ekf = ExtendedKalmanFilter(f, h, **model, residual_z=wrap_bearing)
ukf = UnscentedKalmanFilter(f, h, **model, residual_z=wrap_bearing)
ekf_result = ekf.filter(zs, dt=1.0, backend="jax")
ukf_result = ukf.filter(zs, dt=1.0, backend="jax")
```

## Jacobians (EKF)

Pass `jac_f(x, dt)` and `jac_h(x)` if you have them. Otherwise they are derived with
`jax.jacfwd`, on either backend, which needs JAX and `jax.numpy` model functions. With both
Jacobians supplied, the EKF runs on NumPy without JAX.

## Angles and other wrapped quantities

`residual_z(z, z_pred)` replaces `z - z_pred`. For angles, wrap the difference into [-π, π), as
`wrap_bearing` does above. The UKF also uses it to average its sigma-point measurements (relative
to the central point), so bearings near ±π need no separate mean function.

## Sigma points (UKF)

The UKF uses Van der Merwe's scaled sigma points with `alpha`, `beta`, `kappa`. The defaults
`alpha=1, beta=2, kappa=0` keep every covariance weight nonnegative, so the predicted covariance
can't become indefinite. Small `alpha` (e.g. 1e-3, common in the literature) concentrates the
points but makes the central weight strongly negative.

## Vectorized model functions (UKF)

On the NumPy backend the UKF calls `f` and `h` once per sigma point. With `vectorized=True` it
passes the whole `(2n + 1, n)` stack in one call instead, which is much faster (about 118 → 49 µs
per step on a range-bearing problem). Write such functions with `[..., i]` indexing so they work
on single points too. The JAX backend batches with `vmap` either way.

```python
def f_stack(x, dt):
    x = jnp.asarray(x)
    return jnp.stack(
        [x[..., 0] + dt * x[..., 2], x[..., 1] + dt * x[..., 3], x[..., 2], x[..., 3]], axis=-1
    )


def h_stack(x):
    x = jnp.asarray(x)
    return jnp.stack([jnp.hypot(x[..., 0], x[..., 1]), jnp.arctan2(x[..., 1], x[..., 0])], axis=-1)


def wrap_stack(z, z_pred):
    d = jnp.asarray(z) - jnp.asarray(z_pred)
    return d.at[..., 1].set((d[..., 1] + jnp.pi) % (2 * jnp.pi) - jnp.pi)


fast_ukf = UnscentedKalmanFilter(f_stack, h_stack, **model, residual_z=wrap_stack, vectorized=True)
print(np.abs(fast_ukf.filter(zs, dt=1.0).means - ukf.filter(zs, dt=1.0).means).max())
```
