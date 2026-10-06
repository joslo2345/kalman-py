"""A short textbook Kalman filter with no numerical safeguards, as a stability baseline (S4).

Uses an explicit inverse and the simple covariance update P = (I - K H) P, with no
symmetrization, so rounding can make P indefinite. Runs in the dtype of its inputs.
"""

import numpy as np


def naive_filter(
    F: np.ndarray,
    H: np.ndarray,
    Q: np.ndarray,
    R: np.ndarray,
    x0: np.ndarray,
    P0: np.ndarray,
    zs: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return posterior means (T, n) and covariances (T, n, n)."""
    n = x0.shape[0]
    identity = np.eye(n, dtype=P0.dtype)
    means = np.empty((len(zs), n), dtype=P0.dtype)
    covs = np.empty((len(zs), n, n), dtype=P0.dtype)
    x, P = x0, P0
    for k, z in enumerate(zs):
        x = F @ x
        P = F @ P @ F.T + Q
        K = P @ H.T @ np.linalg.inv(H @ P @ H.T + R)
        x = x + K @ (z - H @ x)
        P = (identity - K @ H) @ P
        means[k], covs[k] = x, P
        if not np.all(np.isfinite(P)):
            means[k + 1 :], covs[k + 1 :] = np.nan, np.nan
            break
    return means, covs
