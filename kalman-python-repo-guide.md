# Building a Kalman Filter Library in Python

A step-by-step guide to setting up the repository for a fast, modern Python Kalman filter library.

## Project goals

The library should fill the gaps left by FilterPy and pykalman:

- **Performance.** It uses vectorized NumPy and an optional JAX backend for JIT compilation, batching, and GPU support.
- **Automatic Jacobians.** JAX's autodiff derives EKF Jacobians from user model functions.
- **Parameter learning.** It estimates Q and R from data through EM or gradient-based likelihood maximization.
- **Diagnostics.** It has built-in NIS/NEES consistency checks and plotting helpers.
- **Modern Python.** It is fully type-hinted, actively maintained, and well documented.

Before starting, review existing JAX-based projects such as dynamax to avoid duplicating work and to find your niche.

## Step 1: Create the repository

Use [uv](https://docs.astral.sh/uv/) for fast environment and dependency management.

```bash
uv init --lib kalman-py
cd kalman-py
git init
```

Add a permissive `LICENSE` (MIT or BSD-3-Clause, matching the scientific Python ecosystem).

## Step 2: Set up the directory layout

Use the `src/` layout so tests always run against the installed package.

```
kalman-py/
├── src/kalman_py/
│   ├── __init__.py
│   ├── py.typed            # marks the package as type-hinted
│   ├── linear.py
│   ├── ekf.py
│   ├── ukf.py
│   ├── smoother.py         # RTS smoother
│   ├── learning.py         # EM / likelihood fitting for Q and R
│   ├── diagnostics.py      # NIS / NEES
│   └── backends/
│       ├── numpy_backend.py
│       └── jax_backend.py
├── tests/
│   └── vectors/            # shared reference test data
├── examples/               # Jupyter notebooks
├── benchmarks/
├── docs/
├── pyproject.toml
├── README.md
└── LICENSE
```

## Step 3: Configure `pyproject.toml`

Keep NumPy as the only required dependency and make JAX optional.

```toml
[project]
name = "kalman-py"
version = "0.1.0"
requires-python = ">=3.10"
dependencies = ["numpy>=1.24"]

[project.optional-dependencies]
jax = ["jax>=0.4"]
plot = ["matplotlib>=3.7"]

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.ruff]
line-length = 100

[tool.mypy]
strict = true
```

Add development dependencies:

```bash
uv add --dev pytest hypothesis mypy ruff pytest-benchmark mkdocs-material "mkdocstrings[python]"
```

## Step 4: Design the API

Support both step-by-step use (for real-time loops) and batch use (for offline data).

```python
import numpy as np
from kalman_py import ExtendedKalmanFilter

def f(x, dt):           # process model
    return np.array([x[0] + x[1] * dt, x[1]])

def h(x):               # measurement model
    return x[:1]

ekf = ExtendedKalmanFilter(f=f, h=h, Q=Q, R=R, x0=x0, P0=P0)

# Step-by-step
ekf.predict(dt=0.1)
ekf.update(z)

# Batch, with JAX backend and automatic Jacobians
result = ekf.filter(measurements, dt=0.1, backend="jax")
smoothed = ekf.smooth(result)
print(result.nis.mean())
```

## Step 5: Implement in this order

1. Implement the linear KF with the NumPy backend, using the Joseph-form update.
2. Add the RTS smoother.
3. Add the JAX backend, using `jax.lax.scan` for the time loop so it JIT-compiles.
4. Add the EKF with `jax.jacfwd` for automatic Jacobians (and user-supplied Jacobians for the NumPy backend).
5. Add the UKF.
6. Add parameter learning for Q and R.
7. Add diagnostics and plotting helpers.

## Step 6: Write unit tests that prove the library is better

Use `pytest`, plus `hypothesis` for property-based tests. Add the libraries you compare against as development dependencies only:

```bash
uv add --dev filterpy pykalman scipy
```

pykalman has seen little maintenance, so it may not install cleanly on the newest NumPy. If it doesn't, pin an older NumPy in a separate comparison environment or use a maintained fork.

### Define "better" before writing tests

For a linear system with Gaussian noise, the Kalman filter is the optimal estimator. Two correct implementations should produce the **same** answer, so on accuracy the goal is to *match* FilterPy and pykalman, not beat them. Measurable improvements come from speed, stability, and features:

| Criterion | What the test does | Pass condition |
|---|---|---|
| Equivalence | Runs the same problem through our filter, FilterPy and pykalman | Means and covariances match within 1e-9 |
| Smoother equivalence | Compares our RTS smoother with pykalman's | Match within 1e-9 |
| Numerical stability | Generates random ill-conditioned problems with `hypothesis` | Our covariance always stays symmetric and positive-definite |
| Autodiff correctness | Compares JAX Jacobians to analytic ones | Match within 1e-10 |
| Parameter learning | Fits Q and R with our learner and with pykalman's EM | Our log-likelihood is equal or higher, in less time |
| Consistency | Runs Monte Carlo trials and computes NIS/NEES | Falls inside the 95% chi-squared bounds |
| Speed | Filters a 100,000-step sequence | At least 10x faster than FilterPy with the JAX backend |

### Organize the tests

```
tests/
├── conftest.py           # shared scenarios as fixtures, JAX setup
├── unit/                 # our code in isolation
├── comparison/           # our code vs FilterPy and pykalman
│   ├── test_equivalence.py
│   ├── test_stability.py
│   ├── test_learning.py
│   └── test_speed.py
└── vectors/              # shared reference test data
```

Mark slow tests so everyday runs stay fast:

```toml
[tool.pytest.ini_options]
markers = ["slow: long-running comparison tests"]
addopts = "-m 'not slow'"
```

Run the full suite with `uv run pytest -m ""`.

### Enable 64-bit precision for JAX

JAX defaults to 32-bit floats, which makes equivalence tests with float64 libraries fail for the wrong reason. Turn on 64-bit mode in `conftest.py`:

```python
import jax
jax.config.update("jax_enable_x64", True)
```

### Example: equivalence with FilterPy

```python
import numpy as np
import pytest
from kalman_py import KalmanFilter

filterpy_kalman = pytest.importorskip("filterpy.kalman")


def test_filter_matches_filterpy(cv_scenario):
    sc = cv_scenario
    ours = KalmanFilter(F=sc.F, H=sc.H, Q=sc.Q, R=sc.R, x0=sc.x0, P0=sc.P0).filter(sc.zs)

    fp = filterpy_kalman.KalmanFilter(dim_x=4, dim_z=2)
    fp.F, fp.H, fp.Q, fp.R = sc.F, sc.H, sc.Q, sc.R
    fp.x, fp.P = sc.x0.reshape(-1, 1).copy(), sc.P0.copy()

    for k, z in enumerate(sc.zs):
        fp.predict()
        fp.update(z)
        np.testing.assert_allclose(ours.means[k], fp.x.ravel(), rtol=1e-9, atol=1e-12)
        np.testing.assert_allclose(ours.covs[k], fp.P, rtol=1e-9, atol=1e-12)
```

### Example: smoother equivalence with pykalman

```python
pykalman = pytest.importorskip("pykalman")


def test_smoother_matches_pykalman(cv_scenario):
    sc = cv_scenario
    kf = KalmanFilter(F=sc.F, H=sc.H, Q=sc.Q, R=sc.R, x0=sc.x0, P0=sc.P0)
    ours = kf.smooth(kf.filter(sc.zs))

    pk = pykalman.KalmanFilter(
        transition_matrices=sc.F, observation_matrices=sc.H,
        transition_covariance=sc.Q, observation_covariance=sc.R,
        initial_state_mean=sc.x0, initial_state_covariance=sc.P0,
    )
    means, covs = pk.smooth(sc.zs)

    np.testing.assert_allclose(ours.means, means, rtol=1e-9, atol=1e-12)
    np.testing.assert_allclose(ours.covs, covs, rtol=1e-9, atol=1e-12)
```

Before comparing, check that both libraries define the initial state the same way (prior before the first measurement versus after), since a mismatch here is the most common cause of false failures.

### Example: stability across random ill-conditioned problems

```python
from hypothesis import given, settings, strategies as st
from tests.scenarios import make_random_linear


@given(seed=st.integers(0, 2**32 - 1), log_r=st.floats(-12, 2))
@settings(max_examples=200, deadline=None)
def test_covariance_stays_symmetric_positive_definite(seed, log_r):
    sc = make_random_linear(seed=seed, r_scale=10.0**log_r, steps=2000)
    res = KalmanFilter(**sc.params).filter(sc.zs)

    P = np.asarray(res.covs)
    np.testing.assert_allclose(P, np.swapaxes(P, -1, -2), atol=1e-10)
    assert np.linalg.eigvalsh(P).min() > 0
```

Run the same generated problems through FilterPy in a separate reporting script and record how often its covariance loses positive-definiteness. Keep that out of the pass/fail test, because the baseline isn't guaranteed to fail.

### Example: automatic Jacobians are correct

```python
import jax.numpy as jnp
from kalman_py.backends.jax_backend import jacobian
from tests.scenarios import range_bearing_h, range_bearing_h_jacobian


@given(x=st.lists(st.floats(-100, 100), min_size=4, max_size=4))
def test_autodiff_jacobian_matches_analytic(x):
    x = jnp.array(x)
    if jnp.hypot(x[0], x[1]) < 1e-3:   # range-bearing is undefined at the origin
        return
    np.testing.assert_allclose(jacobian(range_bearing_h)(x),
                               range_bearing_h_jacobian(x), rtol=1e-10, atol=1e-10)
```

### Example: parameter learning beats pykalman's EM

```python
import time
from kalman_py.learning import fit_noise


@pytest.mark.slow
def test_noise_learning_beats_pykalman_em(cv_scenario_unknown_noise):
    sc = cv_scenario_unknown_noise

    t0 = time.perf_counter()
    ours = fit_noise(F=sc.F, H=sc.H, zs=sc.zs, x0=sc.x0, P0=sc.P0)
    t_ours = time.perf_counter() - t0

    pk = pykalman.KalmanFilter(transition_matrices=sc.F, observation_matrices=sc.H,
                               initial_state_mean=sc.x0, initial_state_covariance=sc.P0)
    t0 = time.perf_counter()
    pk = pk.em(sc.zs, n_iter=50, em_vars=["transition_covariance", "observation_covariance"])
    t_pk = time.perf_counter() - t0

    assert ours.loglikelihood >= pk.loglikelihood(sc.zs) - 1e-6
    assert t_ours < t_pk
    np.testing.assert_allclose(ours.R, sc.true_R, rtol=0.1)  # recovers the truth
```

### Consistency tests

Run 500 Monte Carlo trials where the true state is known. Use `scipy.stats.chi2` to compute the 95% bounds and check that average NIS and NEES fall inside them.

### Example: speed against FilterPy

```python
import jax
from timeit import timeit


@pytest.mark.slow
def test_jax_batch_is_10x_faster_than_filterpy(long_scenario):
    sc = long_scenario  # 100,000 steps
    kf = KalmanFilter(**sc.params)

    def run_ours():
        jax.block_until_ready(kf.filter(sc.zs, backend="jax").means)

    run_ours()  # warm-up so JIT compilation isn't timed
    t_ours = min(timeit(run_ours, number=1) for _ in range(3))
    t_fp = timeit(lambda: run_filterpy(sc), number=1)

    print(f"[report] ours {t_ours:.3f}s vs FilterPy {t_fp:.3f}s ({t_fp / t_ours:.1f}x)")
    assert t_ours * 10 < t_fp
```

`block_until_ready` matters because JAX runs asynchronously. Without it, the timer stops before the work finishes.

### Publish a comparison report

Have CI run the full suite with `-m ""`, collect the `[report]` lines and `pytest-benchmark` results into a `comparison.md` table, and upload it as a build artifact. Fail the build if any pass condition breaks or speed regresses by more than 10%. This report is the evidence behind any claim of being better than FilterPy or pykalman.

## Step 7: Produce comparison numbers against existing libraries

The unit tests in the previous step prove the library is *correct*. This step produces the *numbers* for the README: a table that shows, scenario by scenario, how this library compares to the existing ones. Everything runs from one command, writes raw data to a CSV file, and regenerates the table automatically, so anyone can reproduce the results.

### Benchmark scenarios

Use the same five scenarios in all four repos (C, C++, Python, Rust). Store them as data files (true states, measurements, F, H, Q, R, x0, P0) in the shared test-vectors repo and pull it in as a Git submodule. Because every library reads exactly the same inputs, the numbers are comparable against the external libraries *and* across our four implementations.

| ID | Scenario | State / measurement size | Filters | Length | What it shows |
|---|---|---|---|---|---|
| S1 | 1D constant velocity | 2 / 1 | KF | 10,000 steps | Overhead on tiny problems |
| S2 | 2D constant velocity | 4 / 2 | KF | 10,000 steps | A typical tracking workload |
| S3 | Range-bearing tracking | 4 / 2 | EKF, UKF | 500 steps × 200 seeds | Accuracy on a nonlinear problem |
| S4 | Ill-conditioned problem | 4 / 2 | KF | 1,000,000 steps, 32-bit floats | Numerical stability |
| S5 | INS error-state | 15 / 6 | KF / EKF | 10,000 steps | Speed on a larger state |

Freeze the scenario files before collecting results. Changing a scenario after seeing which library wins makes the numbers meaningless.

### Metrics

| Metric (CSV name) | Unit | How it's measured | Better is |
|---|---|---|---|
| `time_per_step` | ns | Median of 30 timed runs of predict+update, after a warm-up run | Lower |
| `rmse` | state units | Root-mean-square error against the true states in the scenario file | Lower |
| `max_abs_diff` | state units | Largest difference from each baseline's estimate (S1, S2) | Close to 0 |
| `nees` | – | Average normalized estimation error squared, compared with 95% chi-squared bounds | Inside bounds |
| `steps_to_failure` | steps | First step where the covariance is no longer symmetric positive-definite (S4) | Higher |
| `peak_memory` | bytes | Peak Python heap during one run, measured with `tracemalloc` | Lower |

### What to compare against

| Library | Scenarios | Why | How to include it |
|---|---|---|---|
| [FilterPy](https://github.com/rlabbe/filterpy) | S1–S5 (KF, EKF, UKF) | The most widely used Python Kalman filter library | `uv add --dev filterpy`; version recorded with `importlib.metadata` |
| [pykalman](https://github.com/pykalman/pykalman) | S1, S2, S5 (batch KF) | The standard choice for batch filtering, smoothing and EM | `uv add --dev pykalman`; version recorded the same way |
| Naive textbook filter | S4 | Shows what an unstabilized hand-written filter does | A short textbook implementation in `benchmarks/baselines/naive.py` |

FilterPy and pykalman work in different ways, so the comparison is split into two modes. Comparing a per-step library against a batch library would be misleading.

- **Per-step mode** (one predict+update call per measurement, as in a real-time loop): our library vs FilterPy.
- **Batch mode** (the whole measurement sequence in one call, as in offline analysis): our NumPy and JAX backends vs pykalman.

### The harness

`benchmarks/run.py` uses the real FilterPy and pykalman APIs and writes rows in the shared CSV format:

```python
"""Usage: python benchmarks/run.py "commit,cpu,os,toolchain,date" """
import csv
import statistics
import sys
import time
from importlib.metadata import version

import jax
import numpy as np
from filterpy.kalman import KalmanFilter as FPKalmanFilter
from pykalman import KalmanFilter as PKKalmanFilter

from kalman_py import KalmanFilter
from scenarios import load  # reads the shared scenario files

jax.config.update("jax_enable_x64", True)
REPEATS = 30
ENV = sys.argv[1].split(",")


def median_ns_per_step(fn, steps):
    fn()  # warm-up; for JAX this also keeps compilation out of the timing
    times = []
    for _ in range(REPEATS):
        t0 = time.perf_counter()
        fn()
        times.append(time.perf_counter() - t0)
    return statistics.median(times) / steps * 1e9


# ---- Per-step mode ----------------------------------------------------------

def filterpy_step(sc):
    kf = FPKalmanFilter(dim_x=sc.n, dim_z=sc.m)
    kf.F, kf.H, kf.Q, kf.R = sc.F, sc.H, sc.Q, sc.R
    kf.x, kf.P = sc.x0.reshape(-1, 1).copy(), sc.P0.copy()
    out = np.empty((len(sc.zs), sc.n))
    for k, z in enumerate(sc.zs):
        kf.predict()
        kf.update(z)
        out[k] = kf.x.ravel()
    return out


def ours_step(sc):
    kf = KalmanFilter(F=sc.F, H=sc.H, Q=sc.Q, R=sc.R, x0=sc.x0, P0=sc.P0)
    out = np.empty((len(sc.zs), sc.n))
    for k, z in enumerate(sc.zs):
        kf.predict()
        kf.update(z)
        out[k] = kf.x
    return out


# ---- Batch mode -------------------------------------------------------------

def pykalman_batch(sc):
    # pykalman updates with the first measurement before its first prediction, while
    # FilterPy and our library predict first. Moving the prior forward one step
    # aligns the two conventions so the estimates are directly comparable.
    x0 = sc.F @ sc.x0
    P0 = sc.F @ sc.P0 @ sc.F.T + sc.Q
    kf = PKKalmanFilter(transition_matrices=sc.F, observation_matrices=sc.H,
                        transition_covariance=sc.Q, observation_covariance=sc.R,
                        initial_state_mean=x0, initial_state_covariance=P0)
    means, _ = kf.filter(sc.zs)
    return means


def ours_batch_numpy(sc):
    kf = KalmanFilter(F=sc.F, H=sc.H, Q=sc.Q, R=sc.R, x0=sc.x0, P0=sc.P0)
    return kf.filter(sc.zs).means


def ours_batch_jax(sc):
    kf = KalmanFilter(F=sc.F, H=sc.H, Q=sc.Q, R=sc.R, x0=sc.x0, P0=sc.P0)
    # block_until_ready stops the timer only after JAX has actually finished
    return np.asarray(jax.block_until_ready(kf.filter(sc.zs, backend="jax").means))


OURS = version("kalman-py")
RUNS = [
    # library             version              filter label    function
    ("kalman-py",       OURS,                "KF per-step", ours_step),
    ("filterpy",        version("filterpy"), "KF per-step", filterpy_step),
    ("kalman-py",       OURS,                "KF batch",    ours_batch_jax),
    ("kalman-py-numpy", OURS,                "KF batch",    ours_batch_numpy),
    ("pykalman",        version("pykalman"), "KF batch",    pykalman_batch),
]

with open("results/results.csv", "a", newline="") as f:
    out = csv.writer(f)
    for sid in ["S1", "S2", "S5"]:
        sc = load(sid)
        for lib, ver, filt, fn in RUNS:
            ns = median_ns_per_step(lambda: fn(sc), len(sc.zs))
            est = fn(sc)
            rmse = float(np.sqrt(np.mean((est - sc.truth) ** 2)))
            out.writerow([lib, ver, sid, filt, "float64", "time_per_step", f"{ns:.1f}", "ns", *ENV])
            out.writerow([lib, ver, sid, filt, "float64", "rmse", f"{rmse:.6g}", "state", *ENV])
```

Design the JAX backend so its compiled function is cached across filter instances (for example, a module-level `jax.jit` function that takes the matrices as arguments). Otherwise every new `KalmanFilter` would recompile, and the batch numbers would mostly measure compilation.

### Nonlinear accuracy (S3) with FilterPy's real API

```python
from filterpy.kalman import MerweScaledSigmaPoints, UnscentedKalmanFilter


def filterpy_ukf(sc):
    points = MerweScaledSigmaPoints(n=4, alpha=0.1, beta=2.0, kappa=-1.0)
    ukf = UnscentedKalmanFilter(dim_x=4, dim_z=2, dt=sc.dt, fx=sc.fx, hx=sc.hx,
                                points=points, residual_z=sc.residual_z)
    ukf.x, ukf.P, ukf.Q, ukf.R = sc.x0.copy(), sc.P0.copy(), sc.Q, sc.R
    out = np.empty((len(sc.zs), 4))
    for k, z in enumerate(sc.zs):
        ukf.predict()
        ukf.update(z)
        out[k] = ukf.x
    return out
```

Use the same sigma-point parameters (`alpha`, `beta`, `kappa`) in our UKF. The bearing measurement is an angle, so pass a residual function that wraps differences into [-π, π) (`residual_z` above) to both libraries; without it, both filters produce large errors whenever the target crosses ±π. For the EKF, FilterPy's `ExtendedKalmanFilter.update(z, HJacobian, Hx, residual=...)` takes the Jacobian and measurement functions directly. Average `rmse` and `nees` over the 200 seeds.

### Memory

```python
import tracemalloc


def peak_bytes(fn):
    tracemalloc.start()
    fn()
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return peak
```

`tracemalloc` sees only memory allocated through Python. JAX device buffers aren't included, so report `peak_memory` as "n/a" for the JAX backend rather than an artificially low number.

### Stability (S4)

S4 uses 32-bit floats. Check the dtype of each library's covariance after a few steps. NumPy silently converts to 64-bit whenever a 32-bit array meets a 64-bit one (an identity matrix created inside the library, for example), and a library that does this isn't running S4 as specified. Mark it "n/a" and explain why in a footnote under the table.

### One command to produce everything

```bash
ENV="$(git rev-parse --short HEAD),$(uname -m),$(uname -s),python-$(python -c 'import platform; print(platform.python_version())'),$(date -I)"
uv run python benchmarks/run.py "$ENV"
uv run python benchmarks/accuracy.py "$ENV"     # S3, S4: rmse, nees, steps_to_failure
uv run python scripts/make_table.py results/results.csv kalman-py
```

Run with JAX on CPU for the main table so every library uses the same hardware. If you also publish GPU numbers, put them in a separate, clearly labeled table.

### Rules for a fair comparison

- Every library gets the same scenario file, random seed, initial state, and noise matrices.
- Every library uses the same floating-point precision. If a baseline doesn't support the precision a scenario needs, its cell is marked "n/a".
- Only predict+update is timed. Setup, file loading, data conversion, and printing are excluded.
- Every result records the library version (or commit hash), CPU, OS, and toolchain.
- A feature a baseline doesn't have (for example a UKF) is reported as "n/a", never as a failure.
- Publish the raw CSV and the scripts, so the authors of the other libraries can check the setup and reproduce the numbers.

### Output format

Every benchmark run appends rows to `results/results.csv` with this schema:

```
library,library_version,scenario,filter,precision,metric,value,unit,commit,cpu,os,toolchain,date
```

The script below turns the CSV into the README table and calculates how our library compares with the best other library for each row (above 1.00x means ours is better). Save it as `scripts/make_table.py`. It's Python in every repo, since it's only tooling.

```python
"""Usage: python scripts/make_table.py results/results.csv <our-library-name>"""
import csv
import sys
from collections import defaultdict

LOWER_IS_BETTER = {"time_per_step", "cycles_per_step", "rmse", "heap_allocations",
                   "peak_memory", "flash_bytes", "ram_bytes"}
HIGHER_IS_BETTER = {"steps_to_failure"}

rows = list(csv.DictReader(open(sys.argv[1], newline="")))
ours = sys.argv[2]
libs = [ours] + sorted({r["library"] for r in rows} - {ours})

cells = defaultdict(dict)
for r in rows:
    key = (r["scenario"], r["filter"], r["precision"], r["metric"], r["unit"])
    cells[key][r["library"]] = float(r["value"])


def ratio(metric, vals):
    # Libraries named "<ours>-something" are our own variants (e.g. another backend)
    others = [v for lib, v in vals.items() if not lib.startswith(ours)]
    if ours not in vals or not others:
        return "n/a"
    if metric in LOWER_IS_BETTER:
        best = min(others)
        return "–" if vals[ours] == 0 else f"{best / vals[ours]:.2f}x"
    if metric in HIGHER_IS_BETTER:
        best = max(others)
        return "–" if best == 0 else f"{vals[ours] / best:.2f}x"
    return "–"  # nees, max_abs_diff: read the values directly


print("| Scenario | Filter | Precision | Metric | " + " | ".join(libs) + " | Ours vs best other |")
print("|" + "---|" * (len(libs) + 5))
for (scen, filt, prec, metric, unit), vals in sorted(cells.items()):
    values = [f"{vals[lib]:.4g}" if lib in vals else "n/a" for lib in libs]
    print(f"| {scen} | {filt} | {prec} | {metric} ({unit}) | " + " | ".join(values)
          + f" | {ratio(metric, vals)} |")
```

Paste the output into the README between `<!-- BENCH:START -->` and `<!-- BENCH:END -->` markers, or have CI do it.

### Where to run the numbers

GitHub-hosted CI runners share hardware, so their timings can vary by 10–20% between runs. Use them to catch regressions, but produce the published numbers on one dedicated machine, with nothing else running and a fixed CPU frequency (on Linux, set the `performance` governor). State that machine's specs above the table.

### What the README table will look like

The values below are placeholders until the benchmarks run.

| Scenario | Filter | Precision | Metric | kalman-py | filterpy | kalman-py-numpy | pykalman | Ours vs best other |
|---|---|---|---|---|---|---|---|---|
| S2 | KF per-step | float64 | time_per_step (ns) | – | – | n/a | n/a | – |
| S2 | KF batch | float64 | time_per_step (ns) | – | n/a | – | – | – |
| S2 | KF batch | float64 | peak_memory (bytes) | n/a | n/a | – | – | – |
| S3 | UKF | float64 | rmse (state) | – | – | n/a | n/a | – |
| S4 | KF | float32 | steps_to_failure (steps) | – | – | n/a | n/a | – |
| S5 | KF batch | float64 | time_per_step (ns) | – | n/a | – | – | – |

## Step 8: Add continuous integration

Create a GitHub Actions workflow that:

- Tests on Python 3.10 through the latest release, on Linux, macOS, and Windows.
- Runs tests with and without JAX installed.
- Runs `ruff check`, `ruff format --check`, and `mypy`.
- Reports coverage with `pytest-cov`.

## Step 9: Write documentation

- Build docs with MkDocs Material and mkdocstrings for API reference.
- Add notebook tutorials for a tracking example, sensor fusion, and learning noise parameters.
- Write a migration guide from FilterPy, since many users will be switching from it.

## Step 10: Release

- Publish to PyPI using GitHub Actions with trusted publishing (no API tokens stored).
- Use semantic versioning and keep a `CHANGELOG.md`.
- Submit a conda-forge recipe once the API is stable.
