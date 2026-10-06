# Numerical robustness

A covariance must stay symmetric and positive semi-definite. In exact arithmetic the Kalman
recursion guarantees that; in floating point, a very precise sensor (small `R`) makes `P`
ill-conditioned and rounding can break it.

## The default: Joseph form

Every filter updates the covariance with the Joseph form, `(I − KH) P (I − KH)ᵀ + K R Kᵀ`,
which tolerates small errors in the gain, and symmetrizes `P` after every step, so covariances
are exactly symmetric. On a 2,000-step stress test with a very precise sensor, FilterPy's
covariance drifted out of symmetry by up to 1e-6 and could not be Cholesky-factored on 56 steps;
kalman-py's stayed exactly symmetric and failed to factor on 1.

The Joseph form can still lose definiteness when `cond(P)` approaches 1/eps. That's rare in
float64 but happens readily in float32.

## Square-root form

`square_root=True` (on `KalmanFilter`, `ExtendedKalmanFilter` and `UnscentedKalmanFilter`, both
backends) propagates a factor `S` with `P = S Sᵀ` using QR decompositions. Covariances are then
positive semi-definite by construction. It costs about 1.5× (NumPy) to 2.6× (JAX) per step.

```python
import numpy as np

from kalman_py import KalmanFilter

f32 = np.float32
kf = KalmanFilter(
    np.eye(2, dtype=f32),  # F
    np.eye(2, dtype=f32),  # H
    1e-4 * np.eye(2, dtype=f32),  # Q
    1e-6 * np.eye(2, dtype=f32),  # R: a very precise sensor
    np.zeros(2, dtype=f32),
    np.eye(2, dtype=f32),
    square_root=True,
)
result = kf.filter(np.zeros((10, 2), dtype=f32))
print(result.covs.dtype, np.linalg.eigvalsh(result.covs.astype(float)).min() > 0)
```

The square-root UKF uses Cholesky downdates when the central sigma-point weight is negative
(small `alpha`), and a downdate can genuinely fail. It then raises `CovarianceDowndateError`
instead of continuing with an invalid covariance.

## In float32

On the benchmark's ill-conditioned float32 scenario (S4, 1,000,000 steps), the square-root form
and the JAX backend never failed, a textbook filter became unusable after 67,589 steps, and the
default NumPy Joseph form became indefinite after 2,753. **Use `square_root=True` for float32.**
See [Benchmarks](../benchmarks.md#float32-stability-s4).

## Errors instead of silent failures

- The UKF needs a factor of `P` for its sigma points. If `P` is singular or barely indefinite
  from rounding, both backends fall back to an eigendecomposition; if it is genuinely
  indefinite, they raise `ValueError` (the JAX backend after the run, since JAX can't raise
  inside compiled code).
- Q, R and P0 may be singular (e.g. process noise that drives only some states); the
  square-root forms factor them with an eigendecomposition when Cholesky fails.
