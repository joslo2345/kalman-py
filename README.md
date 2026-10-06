# kalman-py

Fast, modern Kalman filters for Python: vectorized NumPy with an optional JAX backend
(JIT, batching, GPU), automatic EKF Jacobians, Q/R parameter learning, and built-in
NIS/NEES diagnostics.

Status: early development. See `kalman-python-repo-guide.md` for the roadmap.

## Development

```bash
uv sync --all-extras
uv run pytest            # fast tests
uv run pytest -m ""      # full suite, including slow comparison tests
uv run ruff check && uv run ruff format --check && uv run mypy src
```

## Benchmarks

Compared with [FilterPy](https://github.com/rlabbe/filterpy) (per-step mode: one predict + update
call per measurement) and [pykalman](https://github.com/pykalman/pykalman) (batch mode: the
whole sequence in one call) on the shared scenarios S1–S5 in `tests/vectors/` (format and
conventions in its README). Above 1.00x in the last column means ours is better than the best
other library. Raw data: `results/results.csv`.

**Preliminary:** these numbers come from a development laptop, not a dedicated machine with a
fixed CPU frequency, so timings can vary by 10–20% between runs.

Reproduce:

```bash
ENV="$(git rev-parse --short HEAD),<cpu>,<os>,python-$(python -c 'import platform; print(platform.python_version())'),$(date +%F)"
uv run python benchmarks/run.py "$ENV"
uv run python benchmarks/accuracy.py "$ENV"
uv run python scripts/make_table.py results/results.csv kalman-py --readme README.md
```

<!-- BENCH:START -->
Measured on Apple M3 Pro (arm64), macOS 27.0.1, python-3.12.13, commit 9a2edc7, 2026-10-05.

| Scenario | Filter | Precision | Metric | kalman-py | filterpy | kalman-py-jax | kalman-py-numpy | kalman-py-sqrt | naive | pykalman | Ours vs best other |
|---|---|---|---|---|---|---|---|---|---|---|---|
| S1 | KF batch | float64 | max_abs_diff (state) | n/a | n/a | n/a | n/a | n/a | n/a | 5.99e-12 | n/a |
| S1 | KF batch | float64 | peak_memory (bytes) | n/a | n/a | n/a | 1,524,384 | n/a | n/a | 1,136,154 | n/a |
| S1 | KF batch | float64 | rmse (state) | 0.4233 | n/a | n/a | 0.4233 | n/a | n/a | 0.4233 | 1.00x |
| S1 | KF batch | float64 | time_per_step (ns) | 406.8 | n/a | n/a | 2,606 | n/a | n/a | 38,047 | 93.53x |
| S1 | KF per-step | float64 | max_abs_diff (state) | n/a | 1.78e-15 | n/a | n/a | n/a | n/a | n/a | n/a |
| S1 | KF per-step | float64 | peak_memory (bytes) | 164,260 | 163,912 | n/a | n/a | n/a | n/a | n/a | 1.00x |
| S1 | KF per-step | float64 | rmse (state) | 0.4233 | 0.4233 | n/a | n/a | n/a | n/a | n/a | 1.00x |
| S1 | KF per-step | float64 | time_per_step (ns) | 13,252 | 10,320 | n/a | n/a | n/a | n/a | n/a | 0.78x |
| S2 | KF batch | float64 | max_abs_diff (state) | n/a | n/a | n/a | n/a | n/a | n/a | 3.32e-12 | n/a |
| S2 | KF batch | float64 | peak_memory (bytes) | n/a | n/a | n/a | 4,165,368 | n/a | n/a | 3,866,229 | n/a |
| S2 | KF batch | float64 | rmse (state) | 0.5073 | n/a | n/a | 0.5073 | n/a | n/a | 0.5073 | 1.00x |
| S2 | KF batch | float64 | time_per_step (ns) | 577.5 | n/a | n/a | 2,786 | n/a | n/a | 38,978 | 67.49x |
| S2 | KF per-step | float64 | max_abs_diff (state) | n/a | 3.55e-15 | n/a | n/a | n/a | n/a | n/a | n/a |
| S2 | KF per-step | float64 | peak_memory (bytes) | 325,324 | 324,632 | n/a | n/a | n/a | n/a | n/a | 1.00x |
| S2 | KF per-step | float64 | rmse (state) | 0.5073 | 0.5073 | n/a | n/a | n/a | n/a | n/a | 1.00x |
| S2 | KF per-step | float64 | time_per_step (ns) | 13,442 | 10,556 | n/a | n/a | n/a | n/a | n/a | 0.79x |
| S3 | EKF | float64 | nees (-) | 4.002 | 4.002 | n/a | n/a | n/a | n/a | n/a | – |
| S3 | EKF | float64 | rmse (state) | 0.8896 | 0.8896 | n/a | n/a | n/a | n/a | n/a | 1.00x |
| S3 | UKF | float64 | nees (-) | 4.001 | 3.979 | n/a | n/a | n/a | n/a | n/a | – |
| S3 | UKF | float64 | rmse (state) | 0.8896 | 0.8896 | n/a | n/a | n/a | n/a | n/a | 1.00x |
| S4 | KF | float32 | steps_to_failure (steps) | 2,753 | n/a | 1,000,000 | n/a | 1,000,000 | 67,589 | n/a | 0.04x |
| S4 | KF | float32 | steps_to_indefinite (steps) | 2,753 | n/a | 1,000,000 | n/a | 1,000,000 | 1,000,000 | n/a | 0.0028x |
| S5 | KF batch | float64 | peak_memory (bytes) | n/a | n/a | n/a | 42,821,472 | n/a | n/a | 45,672,193 | n/a |
| S5 | KF batch | float64 | rmse (state) | 0.1865 | n/a | n/a | 0.1865 | n/a | n/a | 0.1865 | 1.00x |
| S5 | KF batch | float64 | time_per_step (ns) | 5,535 | n/a | n/a | 23,730 | n/a | n/a | 52,290 | 9.45x |
| S5 | KF per-step | float64 | peak_memory (bytes) | 1,222,492 | 1,220,296 | n/a | n/a | n/a | n/a | n/a | 1.00x |
| S5 | KF per-step | float64 | rmse (state) | 0.1865 | 0.1865 | n/a | n/a | n/a | n/a | n/a | 1.00x |
| S5 | KF per-step | float64 | time_per_step (ns) | 22,639 | 18,923 | n/a | n/a | n/a | n/a | n/a | 0.84x |

- KF per-step rows: `kalman-py` is the step-by-step API (NumPy). KF batch rows: `kalman-py` is the JAX backend on CPU and `kalman-py-numpy` the NumPy backend; JAX has no peak_memory because tracemalloc can't see its buffers. max_abs_diff is the largest difference between `kalman-py` and that library's estimates.
- S3 UKF: kalman-py redraws sigma points from the predicted distribution before each update, while FilterPy reuses the propagated ones, so their estimates differ slightly; the EKFs agree to ~1e-13. A consistent filter has average NEES = 4.
- S4 (float32): FilterPy is n/a because it converts float32 inputs to float64 internally (its identity matrix is float64), so it can't run S4 as specified.
- S4: `kalman-py` is the default configuration (NumPy backend, Joseph form); `kalman-py-sqrt` uses `square_root=True`; `kalman-py-jax` is the JAX backend. 1,000,000 means the filter never failed.
<!-- BENCH:END -->
