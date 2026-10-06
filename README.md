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

<!-- BENCH:START -->
<!-- BENCH:END -->
