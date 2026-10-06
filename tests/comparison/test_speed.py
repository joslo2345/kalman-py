from timeit import timeit

import numpy as np
import pytest

from kalman_py import KalmanFilter
from tests.scenarios import LinearScenario

jax = pytest.importorskip("jax")
filterpy_kalman = pytest.importorskip("filterpy.kalman")


def run_filterpy(sc: LinearScenario) -> np.ndarray:
    fp = filterpy_kalman.KalmanFilter(dim_x=sc.n, dim_z=sc.m)
    fp.F, fp.H, fp.Q, fp.R = sc.F, sc.H, sc.Q, sc.R
    fp.x, fp.P = sc.x0.reshape(-1, 1).copy(), sc.P0.copy()
    out = np.empty((len(sc.zs), sc.n))
    for k, z in enumerate(sc.zs):
        fp.predict()
        fp.update(z)
        out[k] = fp.x.ravel()
    return out


@pytest.mark.slow
def test_jax_batch_is_10x_faster_than_filterpy(long_scenario: LinearScenario) -> None:
    sc = long_scenario  # 100,000 steps
    kf = KalmanFilter(**sc.params)

    def run_ours() -> None:
        jax.block_until_ready(kf.filter(sc.zs, backend="jax").means)

    run_ours()  # warm-up so JIT compilation isn't timed
    t_ours = min(timeit(run_ours, number=1) for _ in range(3))
    t_fp = timeit(lambda: run_filterpy(sc), number=1)

    print(f"[report] ours {t_ours:.3f}s vs FilterPy {t_fp:.3f}s ({t_fp / t_ours:.1f}x)")
    assert t_ours * 10 < t_fp
