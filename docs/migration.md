# Migrating from FilterPy

kalman-py follows FilterPy's conventions where it can: the filter predicts before the first
update, and `update` accepts per-call `H` and `R`. On the same linear problem the two produce the
same estimates to rounding error, so switching shouldn't change your results.

## The main differences

| | FilterPy | kalman-py |
|---|---|---|
| Construction | `KalmanFilter(dim_x, dim_z)`, then assign `kf.F`, `kf.H`, ... | `KalmanFilter(F, H, Q, R, x0, P0)` |
| State shape | column vector `(n, 1)` | 1-D array `(n,)` |
| Step API | `predict(u, B, F, Q)`, `update(z, R, H)` | `predict(F, Q)`, `update(z, H, R)` |
| Control input `B u` | supported | not supported yet |
| Batch | `batch_filter(zs)` → means, covs, priors | `filter(zs)` → result object; `backend="jax"` to compile |
| Smoother | `rts_smoother(Xs, Ps)` | `smooth(result)` |
| Likelihood | `kf.log_likelihood` (last step) | `result.log_likelihood` (whole sequence) |
| Mahalanobis distance | `kf.mahalanobis` | `sqrt(result.nis)`, per step |
| EKF Jacobians | passed to every `update(z, HJacobian, Hx)` | `h`, optional `jac_h`, given once; derived by autodiff if omitted |
| EKF nonlinear `f` | override `predict_x` | pass `f(x, dt)`, optional `jac_f` |
| UKF sigma points | `points=MerweScaledSigmaPoints(n, alpha, beta, kappa)` | `alpha=`, `beta=`, `kappa=` |
| UKF angle mean | `z_mean_fn` plus `residual_z` | `residual_z` only |
| UKF state angles | `x_mean_fn`, `residual_x` | not supported yet |
| Covariance form | standard / Joseph | Joseph, or `square_root=True` |

## Side by side

A FilterPy filter and the equivalent kalman-py one:

```python
import warnings

import numpy as np

warnings.filterwarnings("ignore", category=SyntaxWarning)  # FilterPy's docstrings
from filterpy.kalman import KalmanFilter as FilterPyKalmanFilter

from kalman_py import KalmanFilter

dt = 0.1
F = np.array([[1.0, dt], [0.0, 1.0]])
H = np.array([[1.0, 0.0]])
Q = 0.1 * np.array([[dt**3 / 3, dt**2 / 2], [dt**2 / 2, dt]])
R = np.array([[0.5]])
zs = np.sin(np.arange(50) * dt)[:, None]

# FilterPy
fp = FilterPyKalmanFilter(dim_x=2, dim_z=1)
fp.F, fp.H, fp.Q, fp.R = F, H, Q, R
fp.x, fp.P = np.array([[0.0], [1.0]]), np.eye(2)
fp_means, fp_covs, _, _ = fp.batch_filter(zs)

# kalman-py
kf = KalmanFilter(F, H, Q, R, x0=[0.0, 1.0], P0=np.eye(2))
result = kf.filter(zs)

print(np.abs(result.means - fp_means[:, :, 0]).max())  # ~1e-16
```

Smoothing:

```python
fp_smoothed, fp_smoothed_covs, _, _ = fp.rts_smoother(fp_means, fp_covs)
smoothed = kf.smooth(result)
print(np.abs(smoothed.means - fp_smoothed[:, :, 0]).max())
```

## Process noise

FilterPy's `Q_discrete_white_noise(dim=2, dt=dt, var=q)` is the discrete white-noise
*acceleration* model. kalman-py has no helper; the continuous white-noise acceleration model
used throughout these docs is

```python
q = 0.1
Q = q * np.array([[dt**3 / 3, dt**2 / 2], [dt**2 / 2, dt]])
```

These are two different standard models, not two spellings of one: pick the one your problem
calls for.

## From pykalman

pykalman treats `initial_state_mean` as the state **at** the first measurement (it updates
before predicting). To reproduce its numbers, give kalman-py the prior one step earlier, or
equivalently give pykalman `F @ x0` and `F @ P0 @ F.T + Q`. pykalman's EM corresponds to
[`fit_noise`](api/learning.md), which also offers gradient-based maximum likelihood.
