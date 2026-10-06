# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Current state

The project is scaffolded (guide Steps 1–3 plus the test layout) directly in this directory, but the modules under `src/kalman_py/` are empty stubs. `kalman-python-repo-guide.md` is the build plan for `kalman-py`: a fast, typed Python Kalman filter library meant to fill gaps left by FilterPy and pykalman (JAX backend, automatic EKF Jacobians, Q/R learning, NIS/NEES diagnostics). Treat that guide as the spec and read the relevant section before implementing anything. Ruff excludes the guide because newer ruff formats code blocks inside Markdown.

## Tooling

Set up the environment with `uv sync --all-extras`.

- uv for environments and dependencies, `src/` layout, hatchling build, Python >=3.10.
- NumPy is the only required dependency. `jax` and `matplotlib` are optional extras (`[jax]`, `[plot]`). Code must work when JAX isn't installed.
- Lint/type: `uv run ruff check`, `uv run ruff format --check`, `uv run mypy` (strict, line length 100).
- Tests: `uv run pytest` skips tests marked `slow` by default (`addopts = "-m 'not slow'"`). Run everything with `uv run pytest -m ""`. Run a single test with `uv run pytest tests/comparison/test_equivalence.py::test_filter_matches_filterpy`.
- FilterPy, pykalman and scipy are dev-only dependencies used for comparison. Use `pytest.importorskip` for them. pykalman may not install on the newest NumPy.

## Planned architecture

- `src/kalman_py/`: `linear.py`, `ekf.py`, `ukf.py`, `smoother.py` (RTS), `learning.py` (EM / likelihood fitting of Q, R), `diagnostics.py` (NIS/NEES), `backends/{numpy_backend,jax_backend}.py`.
- API supports both step-wise use (`kf.predict(dt=...)`, `kf.update(z)`) and batch use (`kf.filter(zs, backend="numpy"|"jax")` returns a result with `.means`, `.covs` and `.nis`; `kf.smooth(result)`).
- Implementation order: linear KF (NumPy, Joseph-form update), then the RTS smoother, the JAX backend (`jax.lax.scan` time loop), the EKF (`jax.jacfwd` Jacobians on JAX, user-supplied Jacobians on NumPy), the UKF, Q/R learning, and finally diagnostics and plotting.
- The JAX jitted function should be module-level and take the matrices as arguments, so it's cached across filter instances. Otherwise each new filter recompiles and benchmarks end up measuring compilation.

## Correctness and benchmarking rules

- On linear-Gaussian problems the goal is to **match** FilterPy/pykalman (rtol 1e-9). Improvements come from speed, stability and features, not accuracy.
- Our filter and FilterPy predict before the first update. pykalman updates first. Align the prior (`x0 = F x0`, `P0 = F P0 Fᵀ + Q`) before comparing, because this mismatch is the most common cause of false test failures.
- Enable `jax.config.update("jax_enable_x64", True)` in `tests/conftest.py` and in benchmarks. JAX defaults to float32.
- Wrap JAX timings in `jax.block_until_ready` and do a warm-up run first.
- UKF comparisons use the same sigma-point parameters as FilterPy (`alpha=0.1, beta=2.0, kappa=-1.0`). Bearing residuals must wrap to [-π, π).
- Benchmark scenarios S1–S5 are shared across the C/C++/Python/Rust sibling repos through a test-vectors Git submodule. Don't modify scenario files after results have been collected.
- Benchmark output goes to `results/results.csv` (schema: `library,library_version,scenario,filter,precision,metric,value,unit,commit,cpu,os,toolchain,date`). `scripts/make_table.py` turns it into the README table between `<!-- BENCH:START -->` and `<!-- BENCH:END -->`. Library names starting with `kalman-py` count as our own variants.
- S4 runs in float32. If a library silently upcasts to float64, report it as "n/a" instead of giving it a result. `tracemalloc` peak memory is "n/a" for the JAX backend. A feature a baseline lacks is "n/a", never a failure.
