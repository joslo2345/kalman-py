# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project uses
[Semantic Versioning](https://semver.org/spec/v2.0.0.html). Until 1.0.0, minor versions may
include breaking changes.

## [Unreleased]

## [0.1.1] - 2026-10-06

No changes to the library.

### Changed

- The README, shown on the PyPI project page, now gives the PyPI install command
  (`pip install kalman-py`) and a version badge instead of saying the package isn't on PyPI.

## [0.1.0] - 2026-10-06

First release.

### Added

- `KalmanFilter`, `ExtendedKalmanFilter` and `UnscentedKalmanFilter`, each with a step-by-step
  API (`predict`, `update`) and a batch API (`filter`, `smooth`), on a NumPy backend and an
  optional JAX backend (`backend="jax"`, compiled with `jax.lax.scan`).
- RTS smoothers for all three filters (extended and unscented variants), returning smoother
  gains for lag-one covariances.
- Joseph-form covariance updates with exact symmetrization, and a `square_root=True` form for
  every filter that keeps covariances positive semi-definite by construction; the square-root
  UKF raises `CovarianceDowndateError` instead of continuing with an invalid covariance.
- EKF Jacobians derived automatically with `jax.jacfwd` when not supplied; UKF with Van der
  Merwe scaled sigma points, `residual_z` for wrapped quantities such as angles, and
  `vectorized=True` model functions.
- Per-call model overrides in the step API: `predict(F=, Q=)`, `update(z, H=, R=)`, e.g. for
  multi-rate sensor fusion.
- `learning.fit_noise`: maximum-likelihood estimation of `Q` and/or `R` by EM (NumPy) or by BFGS
  on the exact likelihood, differentiated through the JAX filter.
- `diagnostics`: NEES, NIS consistency checks against chi-squared bounds (no SciPy needed);
  `plotting`: estimates with uncertainty bands, consistency plots (`plot` extra).
- Results generic over NumPy and JAX arrays, registered as JAX pytrees; float32 inputs stay
  float32 end to end; fully type-hinted (`py.typed`).
- Benchmarks against FilterPy and pykalman on five frozen shared scenarios (S1–S5), a
  documentation site with three tutorial notebooks, and a migration guide from FilterPy.

[Unreleased]: https://github.com/joslo2345/kalman-py/compare/v0.1.1...HEAD
[0.1.1]: https://github.com/joslo2345/kalman-py/compare/v0.1.0...v0.1.1
[0.1.0]: https://github.com/joslo2345/kalman-py/releases/tag/v0.1.0
