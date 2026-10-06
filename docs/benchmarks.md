# Benchmarks

kalman-py is compared with [FilterPy](https://github.com/rlabbe/filterpy) in per-step mode (one
predict + update call per measurement) and with [pykalman](https://github.com/pykalman/pykalman)
in batch mode (the whole sequence in one call), on five frozen scenarios shared with the C, C++
and Rust implementations. Every library gets the same inputs, seeds and precision.

| ID | Scenario | n / m | Filters | Length |
|---|---|---|---|---|
| S1 | 1-D constant velocity | 2 / 1 | KF | 10,000 steps |
| S2 | 2-D constant velocity | 4 / 2 | KF | 10,000 steps |
| S3 | Range-bearing tracking | 4 / 2 | EKF, UKF | 500 steps × 200 seeds |
| S4 | Ill-conditioned problem, float32 | 4 / 2 | KF | 1,000,000 steps |
| S5 | 15-state INS error-state model | 15 / 6 | KF | 10,000 steps |

The scenario files and their format are described in `tests/vectors/README.md` in the
repository.

!!! note "Preliminary"
    These numbers come from a development laptop (Apple M3 Pro), not a dedicated machine with a
    fixed CPU frequency, so timings can vary by 10–20% between runs.

## Speed

![Time per filter step](assets/benchmark-speed-light.svg#only-light)
![Time per filter step](assets/benchmark-speed-dark.svg#only-dark)

In batch mode the JAX backend is 93× (S1), 67× (S2) and 9.5× (S5) faster than pykalman. In
per-step mode kalman-py is 0.78–0.84× FilterPy's speed: both make the same number of small NumPy
calls, and kalman-py also symmetrizes every covariance. Estimates match in every linear case.

## Float32 stability (S4)

![Steps before the covariance stops being usable](assets/benchmark-stability-light.svg#only-light)
![Steps before the covariance stops being usable](assets/benchmark-stability-dark.svg#only-dark)

!!! warning
    On this problem the default NumPy configuration (Joseph form) is the first to fail. Use
    `square_root=True` when filtering in float32.

## Full results

--8<-- "README.md:bench"

## Reproduce

```bash
ENV="$(git rev-parse --short HEAD),<cpu>,<os>,python-<version>,$(date +%F)"
uv run python benchmarks/run.py "$ENV"
uv run python benchmarks/accuracy.py "$ENV"
uv run python scripts/make_table.py results/results.csv kalman-py --readme README.md
uv run python scripts/plot_benchmarks.py
```
