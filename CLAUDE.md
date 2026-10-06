# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Current state

The project is scaffolded directly in this directory. Steps 5.1–5.4 are done: the linear KF, the RTS smoother and the EKF (with extended RTS smoothing), each on both the NumPy and JAX backends. The other modules under `src/kalman_py/` are still empty stubs. `kalman-python-repo-guide.md` is the build plan for `kalman-py`: a fast, typed Python Kalman filter library meant to fill gaps left by FilterPy and pykalman (JAX backend, automatic EKF Jacobians, Q/R learning, NIS/NEES diagnostics). Treat that guide as the spec and read the relevant section before implementing anything. Ruff excludes the guide because newer ruff formats code blocks inside Markdown.

## Tooling

Set up the environment with `uv sync --all-extras`.

- uv for environments and dependencies, `src/` layout, hatchling build, Python >=3.10.
- NumPy is the only required dependency. `jax` and `matplotlib` are optional extras (`[jax]`, `[plot]`). Code must work when JAX isn't installed.
- Lint/type: `uv run ruff check`, `uv run ruff format --check`, `uv run mypy src tests` (strict, line length 100).
- Tests: `uv run pytest` skips tests marked `slow` by default (`addopts = "-m 'not slow'"`). Run everything with `uv run pytest -m ""`. Run a single test with `uv run pytest tests/comparison/test_equivalence.py::test_filter_matches_filterpy`.
- FilterPy, pykalman and scipy are dev-only dependencies used for comparison. Use `pytest.importorskip` for them. pykalman may not install on the newest NumPy.

## Architecture

- `backends/numpy_backend.py` holds pure functions (`predict`, `update`, `kalman_filter`, `rts_smoother`). Each backend implements its own algorithms, so there's no separate `smoother.py` module, unlike the guide's layout. The classes (`linear.KalmanFilter`) validate inputs, keep the step-by-step state `(x, P)` and dispatch `filter(zs, backend=...)` to a backend. `filter()` always starts from `(x0, P0)` and leaves `(x, P)` untouched.
- `backends/jax_backend.py` mirrors the NumPy algorithms using `jax.lax.scan`. `linear.py` imports it lazily, so NumPy-only installs work. The jitted functions are module-level and take the matrices as arguments, so compiled code is shared across filter instances (a test checks `_kalman_filter._cache_size()`). Results stay JAX arrays on the device, and `smooth()` picks the backend from the result's array type.
- JAX performance: tiny matrices make XLA dot and LAPACK calls overhead-bound. For shapes ≤ `_FUSED_MATMUL_MAX_DIM` (16), `_mm` writes products as broadcast-multiply-sum. For measurement size ≤ `_UNROLLED_CHOLESKY_MAX_DIM` (8), `_solve_spd` unrolls the Cholesky factorization. This took a 4/2 model from 1.9 µs to about 0.6 µs per step. The thresholds come from CPU measurements. `test_matches_numpy_backend_on_both_kernel_paths` covers both sides of them. Use `_mm` rather than `@` for products inside the scan.
- `ekf.py`: `ExtendedKalmanFilter(f, h, Q, R, x0, P0, *, jac_f, jac_h, residual_z)` with `f(x, dt)` and `h(x)`. Missing Jacobians come from `jax.jacfwd`. In step mode and on the NumPy backend they come from `jax_backend.jacobian` (a jitted `jacfwd`), so JAX is only optional when both Jacobians are supplied. Model functions must be `jax.numpy` code to work under the JAX backend or autodiff. The JAX EKF takes them as static jit arguments, so compiled code is cached per function object and lambdas made per instance recompile. `filter(zs, dt)` accepts a scalar `dt` or one per step. `ExtendedFilterResult` adds `transition_jacobians`, which the smoother uses (the backends' `rts_smoother` accepts a 2-D F or a `(T, n, n)` stack).
- Both backends share the measurement step as `_correct(x, P, y, H, R)`. The linear update and the EKF update differ only in how they form the innovation `y` and `H`.
- `square_root=True` (on `KalmanFilter` and `ExtendedKalmanFilter`, both backends) carries a factor S with P = S Sᵀ. Predict uses QR of `[F S, L_Q]ᵀ`. The update triangularizes the pre-array `[[L_R, H S], [0, S]]` and reads off the innovation factor, the scaled gain and the posterior factor. `psd_factor` turns Q, R and P0 into factors (Cholesky, or `eigh` when they're singular) once, outside jit. Results and the smoother still use full covariances (S Sᵀ). In step mode the class keeps the factor in `_S` and recomputes `P` after each step. It costs about 2.6× per step on JAX (QR goes through LAPACK, not the fused path) and about 1.5× on NumPy, so Joseph stays the default.
- NumPy performance: on small matrices, cost per step is set by per-call overhead (about 0.6 µs per matmul, 2–3 µs per LAPACK call), not by arithmetic. So the per-step update (`_correct` = `_joseph_gain` + `_joseph_covariance`) skips NIS and likelihood. Batch loops store innovations and their covariances, and `_innovation_stats` computes NIS and log-likelihood for all steps with one batched call. Pass a cached `identity` to `update` in loops.
- `_joseph_kalman_filter` (linear, Joseph form, batch) has an exact steady-state shortcut. Every 16 steps it checks whether the predicted covariance repeats bit for bit. Once it does, the remaining steps reuse the gain and covariances and only update the mean, with the same arithmetic, so results stay bitwise identical. That takes the 2-D constant-velocity model from about 15 to about 3 µs per step. Many models never hit an exact repeat and get no speedup. Keep the steady-state arithmetic identical to the full-step code (`test_steady_state_shortcut_is_bitwise_exact` checks this).
- Measured NumPy per step on a 4/2 model: FilterPy 11.2 µs, our step API 14.8 µs (11.2 without symmetrization: the same number of calls hits the same floor), our batch 3.2 µs at a fixed point and 15.2 µs without one, pykalman batch 39.9 µs.
- `_common.py` holds input validation and the lazy `jax_backend()` loader used by both filter classes.
- `FilterResult` (`result.py`) stores posteriors plus the one-step-ahead priors (`predicted_means`, `predicted_covs`) that the RTS smoother needs, along with per-step NIS and the total log-likelihood (a 0-d array). `FilterResult` and `SmootherResult` are generic over the array type (`FilterResult[np.ndarray]` or `FilterResult[jax.Array]`) and are registered as JAX pytrees in `jax_backend.py`. `KalmanFilter.filter` and `smooth` use overloads, so mypy knows which array type each backend returns.
- Dtype is preserved end to end, so float32 inputs stay float32 (scenario S4 depends on this). Any constant created inside the backends, such as an identity matrix, must use the input dtype.
- Covariances are symmetrized after every predict and update, and gains come from `np.linalg.solve` rather than an explicit inverse.
- `tests/unit/test_smoother.py` checks the smoother against the exact posterior from conditioning the joint Gaussian of all states and measurements. This is an independent reference that doesn't share any of the recursion's code.
- The slow stability test runs on both backends. The slow speed test (`tests/comparison/test_speed.py`) asserts the JAX backend is ≥10× faster than FilterPy on 100k steps; it measured 19.6× on Apple Silicon CPU.
- The stability property test runs on all four combinations of backend and form. The square-root cases must pass. The Joseph-form cases are non-strict `xfail`: Hypothesis found `seed=3606, log_r=-9` (R ≈ 1e-9), where the NumPy Joseph form becomes truly indefinite (≈ -70 eps relative). `test_square_root.py::test_stays_positive_semidefinite_where_joseph_form_fails` pins that case deterministically, and it also asserts that the Joseph form still fails there, so it notices if the scenario changes.
- `tests/nonlinear_scenarios.py` (needs JAX) has the range-bearing EKF scenario, whose target crosses the ±π bearing boundary, plus model functions that accept NumPy or JAX input.
- `tests/scenarios.py` holds the shared synthetic problems, and `tests/conftest.py` exposes them as fixtures.

## Plan for the rest

- `src/kalman_py/`: `linear.py`, `ekf.py`, `ukf.py`, `smoother.py` (RTS), `learning.py` (EM / likelihood fitting of Q, R), `diagnostics.py` (NIS/NEES), `backends/{numpy_backend,jax_backend}.py`.
- API supports both step-wise use (`kf.predict(dt=...)`, `kf.update(z)`) and batch use (`kf.filter(zs, backend="numpy"|"jax")` returns a result with `.means`, `.covs` and `.nis`; `kf.smooth(result)`).
- Implementation order: linear KF (NumPy, Joseph-form update), then the RTS smoother, the JAX backend (`jax.lax.scan` time loop), the EKF (`jax.jacfwd` Jacobians on JAX, user-supplied Jacobians on NumPy), the UKF, Q/R learning, and finally diagnostics and plotting.
- The JAX jitted function should be module-level and take the matrices as arguments, so it's cached across filter instances. Otherwise each new filter recompiles and benchmarks end up measuring compilation.

## Correctness and benchmarking rules

- The stability test checks for a covariance that is positive semi-definite *to working precision* (min eigenvalue ≥ −10·n·eps·max eigenvalue), not strictly min eigenvalue > 0. When R is tiny, cond(P) reaches about 1/eps, and `eigvalsh` can't resolve the sign of the smallest eigenvalue, whichever filter produced P. Only a square-root (Cholesky-factor) filter would get past this.
- On linear-Gaussian problems the goal is to **match** FilterPy/pykalman (rtol 1e-9). Improvements come from speed, stability and features, not accuracy.
- Our filter and FilterPy predict before the first update. pykalman updates first. Align the prior (`x0 = F x0`, `P0 = F P0 Fᵀ + Q`) before comparing, because this mismatch is the most common cause of false test failures.
- Enable `jax.config.update("jax_enable_x64", True)` in `tests/conftest.py` and in benchmarks. JAX defaults to float32.
- Wrap JAX timings in `jax.block_until_ready` and do a warm-up run first.
- UKF comparisons use the same sigma-point parameters as FilterPy (`alpha=0.1, beta=2.0, kappa=-1.0`). Bearing residuals must wrap to [-π, π).
- Benchmark scenarios S1–S5 are shared across the C/C++/Python/Rust sibling repos through a test-vectors Git submodule. Don't modify scenario files after results have been collected.
- Benchmark output goes to `results/results.csv` (schema: `library,library_version,scenario,filter,precision,metric,value,unit,commit,cpu,os,toolchain,date`). `scripts/make_table.py` turns it into the README table between `<!-- BENCH:START -->` and `<!-- BENCH:END -->`. Library names starting with `kalman-py` count as our own variants.
- S4 runs in float32. If a library silently upcasts to float64, report it as "n/a" instead of giving it a result. `tracemalloc` peak memory is "n/a" for the JAX backend. A feature a baseline lacks is "n/a", never a failure.
