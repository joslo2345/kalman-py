# Getting started

## A first filter

A target moves along a line with roughly constant velocity, and we measure its position.

```python
import numpy as np

from kalman_py import KalmanFilter

dt = 0.1
F = np.array([[1.0, dt], [0.0, 1.0]])  # state: (position, velocity)
H = np.array([[1.0, 0.0]])  # we measure position
Q = 0.5 * np.array([[dt**3 / 3, dt**2 / 2], [dt**2 / 2, dt]])  # process noise
R = np.array([[0.25]])  # measurement noise
kf = KalmanFilter(F, H, Q, R, x0=[0.0, 1.0], P0=np.eye(2))

rng = np.random.default_rng(0)
zs = (np.arange(1, 101) * dt + 0.5 * rng.standard_normal(100))[:, None]
```

Use it one measurement at a time, as in a real-time loop:

```python
for z in zs[:3]:
    kf.predict()
    kf.update(z)
print(kf.x, kf.P)
```

or on the whole sequence at once, then smooth:

```python
result = kf.filter(zs)
smoothed = kf.smooth(result)
print(result.means.shape, result.covs.shape)  # (100, 2) (100, 2, 2)
```

`filter` always starts from `(x0, P0)` and leaves the step-by-step state `(kf.x, kf.P)` alone.

## Conventions

- **The prior comes before the first prediction.** `(x0, P0)` describes the state before any
  measurement, and every measurement is preceded by a predict step, as in FilterPy. pykalman
  instead treats its initial state as the state at the first measurement; to reproduce its
  numbers, pass `F @ x0` and `F @ P0 @ F.T + Q`.
- **Shapes.** States are 1-D arrays of length `n`, measurements of length `m`; a sequence of
  measurements is `(T, m)` (a 1-D array is accepted when `m == 1`).
- **Precision.** Results have the dtype of the model: float32 inputs stay float32 end to end,
  integers become float64.
- **Varying models.** In the step API, `predict(F=..., Q=...)` and `update(z, H=..., R=...)`
  replace the model for one call, e.g. for a varying time step or a second sensor (see the
  [sensor fusion tutorial](tutorials/02_sensor_fusion.ipynb)).

## What a result contains

| Field | Shape | Meaning |
|---|---|---|
| `means`, `covs` | `(T, n)`, `(T, n, n)` | posterior after each measurement |
| `predicted_means`, `predicted_covs` | `(T, n)`, `(T, n, n)` | prior just before each measurement |
| `nis` | `(T,)` | normalized innovation squared (for [consistency checks](api/diagnostics.md)) |
| `log_likelihood` | 0-d array | log p(z₁, …, z_T) under the model |

`smooth` returns `means`, `covs` and the smoother `gains`.

```python
print(float(result.log_likelihood), result.nis.mean())
```

## Next

- Run it compiled: [Backends](guide/backends.md).
- Nonlinear models: [EKF and UKF](guide/nonlinear.md).
- Choosing between the Joseph and square-root forms: [Numerical robustness](guide/robustness.md).
