import shutil
from pathlib import Path

import numpy as np
import pytest

from benchmarks.scenarios import VECTORS, load

ALL = ["S1", "S2", "S3", "S4", "S5"]
LINEAR_WITH_TRUTH = ["S1", "S2", "S5"]


@pytest.mark.parametrize("sid", ALL)
def test_scenarios_load_with_verified_checksums(sid: str) -> None:
    sc = load(sid)
    seeds, steps, m = sc.measurements.shape
    assert m == sc.m and seeds == len(sc.meta["seeds"])
    assert sc.measurements.dtype == sc.dtype
    if sc.truth is not None:
        assert sc.truth.shape == (seeds, steps, sc.n)
    assert sc.Q.shape == sc.P0.shape == (sc.n, sc.n)


def test_scenario_table_matches_the_guide() -> None:
    expected = {  # (n, m, steps, seeds, precision)
        "S1": (2, 1, 10_000, 1, "float64"),
        "S2": (4, 2, 10_000, 1, "float64"),
        "S3": (4, 2, 500, 200, "float64"),
        "S4": (4, 2, 1_000_000, 1, "float32"),
        "S5": (15, 6, 10_000, 1, "float64"),
    }
    for sid, (n, m, steps, seeds, precision) in expected.items():
        sc = load(sid)
        assert (sc.n, sc.m, sc.measurements.shape[1], sc.measurements.shape[0]) == (
            n,
            m,
            steps,
            seeds,
        )
        assert sc.meta["precision"] == precision
    assert load("S4").truth is None


def test_corrupted_file_is_rejected(tmp_path: Path) -> None:
    shutil.copytree(VECTORS / "S1", tmp_path / "S1")
    data = tmp_path / "S1" / "measurements.f64"
    raw = bytearray(data.read_bytes())
    raw[100] ^= 1
    data.write_bytes(bytes(raw))
    with pytest.raises(ValueError, match="does not match its sha256"):
        load("S1", root=tmp_path)
    load("S1", root=tmp_path, verify=False)  # explicit opt-out still reads


def assert_sample_covariance(samples: np.ndarray, expected: np.ndarray) -> None:
    """Sample covariance within 5 standard errors of ``expected`` for every entry. For Gaussian
    data, Var[S_ij] = (M_ii M_jj + M_ij^2) / N."""
    N = samples.shape[0]
    S = np.atleast_2d(np.cov(samples.T))
    d = np.diagonal(expected)
    standard_error = np.sqrt((np.outer(d, d) + expected**2) / N)
    tol = 5 * standard_error + 1e-12 * np.abs(expected).max()  # exact zeros (singular Q)
    assert np.all(np.abs(S - expected) <= tol), np.max(np.abs(S - expected) / tol)


@pytest.mark.parametrize("sid", LINEAR_WITH_TRUTH)
def test_stored_data_follows_the_declared_model(sid: str) -> None:
    # Measurement residuals z - H x have covariance R; process residuals x_k - F x_{k-1}
    # have covariance Q (the format's conventions: truth[k] is where z[k] was taken).
    sc = load(sid)
    truth, zs = sc.truth_single, sc.zs
    assert sc.H is not None
    assert_sample_covariance(zs - truth @ sc.H.T, sc.R)
    assert_sample_covariance(truth[1:] - truth[:-1] @ sc.F.T, sc.Q)
