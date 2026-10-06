import numpy as np
import pytest

from kalman_py.diagnostics import chi2_bounds, chi2_cdf, chi2_ppf, consistency_check, nees

stats = pytest.importorskip("scipy.stats")


@pytest.mark.parametrize("dof", [0.5, 1, 2, 4, 30, 2000, 10_000])
@pytest.mark.parametrize("p", [1e-6, 0.025, 0.5, 0.975, 1 - 1e-6])
def test_chi2_quantile_and_cdf_match_scipy(dof: float, p: float) -> None:
    np.testing.assert_allclose(chi2_ppf(p, dof), stats.chi2.ppf(p, dof), rtol=1e-10)
    np.testing.assert_allclose(chi2_cdf(stats.chi2.ppf(p, dof), dof), p, rtol=1e-10, atol=1e-15)


def test_chi2_bounds_are_for_the_average_over_runs() -> None:
    lo, hi = chi2_bounds(dof=4, n_runs=50, confidence=0.95)
    np.testing.assert_allclose([lo, hi], stats.chi2.ppf([0.025, 0.975], 200) / 50, rtol=1e-10)


@pytest.mark.parametrize("args", [(0, 1, 0.95), (2, 0, 0.95), (2, 1, 1.0)])
def test_chi2_bounds_reject_bad_arguments(args: tuple[float, int, float]) -> None:
    with pytest.raises(ValueError):
        chi2_bounds(*args)


def test_nees_single_run_and_batched() -> None:
    rng = np.random.default_rng(0)
    A = rng.normal(size=(3, 6, 2, 2))
    covs = A @ np.swapaxes(A, -1, -2) + np.eye(2)
    truth, means = rng.normal(size=(3, 6, 2)), rng.normal(size=(3, 6, 2))

    batched = nees(truth, means, covs)
    assert batched.shape == (3, 6)
    e = truth[1, 4] - means[1, 4]
    np.testing.assert_allclose(batched[1, 4], e @ np.linalg.inv(covs[1, 4]) @ e, rtol=1e-12)
    np.testing.assert_array_equal(nees(truth[1], means[1], covs[1]), batched[1])

    with pytest.raises(ValueError, match="covs must end in shape"):
        nees(truth, means, covs[..., :1])


def test_consistency_check_on_exact_chi2_samples() -> None:
    rng = np.random.default_rng(1)
    values = rng.chisquare(df=3, size=(200, 2000))
    check = consistency_check(values, dof=3)
    assert check.n_runs == 200
    assert check.average.shape == (2000,)
    # Each step's average is inside its 95% interval with probability 0.95.
    assert abs(check.fraction_inside - 0.95) < 0.02


def test_consistency_check_flags_over_and_under_confidence() -> None:
    rng = np.random.default_rng(2)
    values = rng.chisquare(df=3, size=(200, 500))
    overconfident = consistency_check(2 * values, dof=3)  # errors larger than P claims
    overcautious = consistency_check(0.5 * values, dof=3)
    assert (
        overconfident.fraction_inside < 0.05 and overconfident.average.mean() > overconfident.upper
    )
    assert overcautious.fraction_inside < 0.05 and overcautious.average.mean() < overcautious.lower


def test_consistency_check_accepts_a_single_run() -> None:
    check = consistency_check(np.full(10, 2.0), dof=2)
    assert check.n_runs == 1 and check.fraction_inside == 1.0
