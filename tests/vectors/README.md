# Benchmark scenarios S1–S5

Shared, frozen inputs for the C, C++, Python and Rust Kalman filter libraries. Every library
reads exactly these files, so results are comparable across libraries and implementations. Do
not regenerate or edit them after results have been collected (`scripts/generate_vectors.py`
refuses to overwrite without `--force`; regeneration is not bit-reproducible across NumPy/BLAS
versions anyway).

| ID | Scenario | n / m | Filters | Steps × seeds | Precision |
|---|---|---|---|---|---|
| S1 | 1-D constant velocity | 2 / 1 | KF | 10,000 × 1 | float64 |
| S2 | 2-D constant velocity | 4 / 2 | KF | 10,000 × 1 | float64 |
| S3 | Range-bearing tracking (bearing wraps at ±π) | 4 / 2 | EKF, UKF | 500 × 200 | float64 |
| S4 | Ill-conditioned problem | 4 / 2 | KF | 1,000,000 × 1 | float32 |
| S5 | 15-state INS error-state model, GPS position + velocity | 15 / 6 | KF | 10,000 × 1 | float64 |

## Format

Each scenario directory holds a `meta.json` and raw binary arrays.

- **Binary arrays** (`*.f64`, `*.f32`): little-endian IEEE-754, C order (last index fastest), no
  header. Shapes are in `meta.json`:
  - `measurements`: `[seeds, steps, m]`
  - `truth`: `[seeds, steps, n]` (absent for S4, whose stability metric needs none)
- **`meta.json`**:
  - `model`: `F`, `H`, `Q`, `R`, `x0`, `P0` as nested lists (row-major), exact float64 values.
    S3 has no `H`: its measurement model `h` and residual are described in text.
  - `seeds`, `dt` (`null` when unused), `precision`, `measurement_model` (`linear` or
    `range_bearing`), `filters`.
  - `files`: for each array its `path`, `dtype`, `shape` and `sha256` of the file contents.
    Readers should verify the hash.

## Conventions

- `(x0, P0)` is the prior **before** the first prediction: each measurement `z_k` is preceded by
  a predict step (the FilterPy convention). Libraries that update first (e.g. pykalman) must
  be given `F x0` and `F P0 Fᵀ + Q` instead.
- `truth[s, k]` is the state at which `measurements[s, k]` was taken.
- S4 is filtered entirely in float32 from these float64 model values cast to float32. Its
  selection rule, fixed before results were collected, is recorded in its `meta.json`.
