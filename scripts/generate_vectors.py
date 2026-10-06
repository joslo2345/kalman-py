"""Generate the shared benchmark scenarios S1-S5 into tests/vectors/.

Usage: python scripts/generate_vectors.py [--out tests/vectors] [--force]

The files are the frozen inputs for every implementation (C, C++, Python, Rust), so this refuses
to overwrite existing ones without --force: changing a scenario after seeing results makes the
numbers meaningless. Regenerating is not bit-reproducible across NumPy/BLAS versions, so the
generated files, not this script, are the reference; meta.json records their SHA-256.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tests.nonlinear_scenarios import SENSOR_R, make_range_bearing
from tests.scenarios import cv_model_2d, make_random_linear, simulate

STEPS = 10_000


def cv_model_1d(dt: float, q: float) -> tuple[np.ndarray, np.ndarray]:
    F = np.array([[1.0, dt], [0.0, 1.0]])
    Q = q * np.array([[dt**3 / 3, dt**2 / 2], [dt**2 / 2, dt]])
    return F, Q


def ins_error_state_model(dt: float) -> dict[str, np.ndarray]:
    """15-state INS error model at rest: position, velocity, attitude errors, accelerometer and
    gyro biases; GPS measures position and velocity. First-order discretization F = I + A dt."""
    f = np.array([0.0, 0.0, -9.81])  # specific force (level, at rest)
    skew_f = np.array([[0, -f[2], f[1]], [f[2], 0, -f[0]], [-f[1], f[0], 0]])
    I3, Z3 = np.eye(3), np.zeros((3, 3))
    A = np.block(
        [
            [Z3, I3, Z3, Z3, Z3],  # d(dp) = dv
            [Z3, Z3, -skew_f, -I3, Z3],  # d(dv) = -[f x] dtheta - b_a
            [Z3, Z3, Z3, Z3, -I3],  # d(dtheta) = -b_g
            [Z3, Z3, Z3, Z3, Z3],  # biases: random walks
            [Z3, Z3, Z3, Z3, Z3],
        ]
    )
    F = np.eye(15) + A * dt
    # Accelerometer 0.05 m/s/sqrt(s), gyro 1e-3 rad/sqrt(s), bias walks 1e-4 and 1e-5; no
    # direct position noise, so Q is singular.
    q = np.concatenate(
        [
            np.zeros(3),
            np.full(3, 0.05**2),
            np.full(3, 1e-3**2),
            np.full(3, 1e-4**2),
            np.full(3, 1e-5**2),
        ]
    )
    H = np.hstack([np.eye(6), np.zeros((6, 9))])
    R = np.diag(np.concatenate([np.full(3, 1.0**2), np.full(3, 0.1**2)]))
    P0 = np.diag(
        np.concatenate(
            [
                np.full(3, 10.0**2),
                np.full(3, 1.0),
                np.full(3, 0.01**2),
                np.full(3, 0.1**2),
                np.full(3, 0.01**2),
            ]
        )
    )
    return {"F": F, "H": H, "Q": np.diag(q * dt), "R": R, "x0": np.zeros(15), "P0": P0}


def write_scenario(
    root: Path,
    sid: str,
    description: str,
    filters: list[str],
    model: dict[str, object],
    arrays: dict[str, np.ndarray],
    extra: dict[str, object],
    force: bool,
) -> None:
    directory = root / sid
    if (directory / "meta.json").exists() and not force:
        print(f"{sid}: exists, left unchanged (use --force to overwrite)")
        return
    directory.mkdir(parents=True, exist_ok=True)
    files = {}
    for name, array in arrays.items():
        data = np.ascontiguousarray(array, dtype=array.dtype.newbyteorder("<"))
        suffix = {np.dtype("float64"): "f64", np.dtype("float32"): "f32"}[np.dtype(array.dtype)]
        path = directory / f"{name}.{suffix}"
        path.write_bytes(data.tobytes())
        files[name] = {
            "path": path.name,
            "dtype": f"{np.dtype(array.dtype).name}-le",
            "shape": list(data.shape),
            "sha256": hashlib.sha256(data.tobytes()).hexdigest(),
        }
    meta = {
        "id": sid,
        "description": description,
        "filters": filters,
        **extra,
        "model": {
            k: (np.asarray(v).tolist() if not isinstance(v, str) else v) for k, v in model.items()
        },
        "files": files,
    }
    (directory / "meta.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    print(f"{sid}: wrote {', '.join(f['path'] for f in files.values())}")


def linear_extra(
    seeds: list[int], dt: float | None, precision: str = "float64"
) -> dict[str, object]:
    return {"seeds": seeds, "dt": dt, "precision": precision, "measurement_model": "linear"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, default=Path("tests/vectors"))
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    # S1: 1-D constant velocity, position measured.
    F, Q = cv_model_1d(dt=0.1, q=0.5)
    H, R = np.array([[1.0, 0.0]]), np.array([[0.25]])
    x0, P0 = np.array([0.0, 1.0]), np.eye(2)
    sc = simulate(F, H, Q, R, x0, P0, STEPS, seed=1)
    write_scenario(
        args.out,
        "S1",
        "1-D constant velocity; position measured",
        ["KF"],
        {"F": F, "H": H, "Q": Q, "R": R, "x0": x0, "P0": P0},
        {"truth": sc.truth[None], "measurements": sc.zs[None]},
        linear_extra([1], 0.1),
        args.force,
    )

    # S2: 2-D constant velocity, position measured (the typical tracking workload).
    F, Q = cv_model_2d(dt=0.1, q=0.5)
    H = np.array([[1.0, 0, 0, 0], [0, 1.0, 0, 0]])
    R, x0, P0 = np.diag([0.5, 0.8]), np.array([0.0, 0.0, 1.0, -1.0]), np.diag([1.0, 1.0, 0.5, 0.5])
    sc = simulate(F, H, Q, R, x0, P0, STEPS, seed=2)
    write_scenario(
        args.out,
        "S2",
        "2-D constant velocity; position measured",
        ["KF"],
        {"F": F, "H": H, "Q": Q, "R": R, "x0": x0, "P0": P0},
        {"truth": sc.truth[None], "measurements": sc.zs[None]},
        linear_extra([2], 0.1),
        args.force,
    )

    # S3: range-bearing tracking of a target passing behind the sensor (bearing wraps at +-pi).
    seeds = list(range(200))
    runs = [make_range_bearing(seed=s, steps=500) for s in seeds]
    first = runs[0]
    write_scenario(
        args.out,
        "S3",
        "2-D constant velocity, range-bearing measurements from a sensor at the origin; the "
        "target passes behind the sensor, so bearings wrap from +pi to -pi",
        ["EKF", "UKF"],
        {
            "F": first.F,
            "Q": first.Q,
            "R": SENSOR_R,
            "x0": first.x0,
            "P0": first.P0,
            "h": "z = [hypot(px, py), atan2(py, px)] for state (px, py, vx, vy)",
            "residual": "bearing residual wrapped to [-pi, pi)",
            "ukf_sigma_points": "Van der Merwe scaled, alpha=0.1, beta=2, kappa=-1 (all libraries)",
        },
        {
            "truth": np.stack([r.truth for r in runs]),
            "measurements": np.stack([r.zs for r in runs]),
        },
        {"seeds": seeds, "dt": 1.0, "precision": "float64", "measurement_model": "range_bearing"},
        args.force,
    )

    # S4: float32 stability. Selection rule, fixed before results were collected: among
    # make_random_linear(seed, r_scale) for seed = 0, 1, ... and r_scale = 1e-3, 1e-4, 1e-5,
    # the first problem where a naive textbook filter (P = (I - KH) P) in float32 survives at
    # least 1,000 steps but fails within 1,000,000. That is seed=1, r_scale=1e-4. The filter
    # prior is P0 = I. Only measurements are stored: the metric needs no truth.
    steps = 1_000_000
    sc = make_random_linear(seed=1, r_scale=1e-4, steps=steps)
    write_scenario(
        args.out,
        "S4",
        "ill-conditioned 4-state / 2-measurement problem in float32 (precise sensors, random "
        "measurement geometry): numerical stability",
        ["KF"],
        {"F": sc.F, "H": sc.H, "Q": sc.Q, "R": sc.R, "x0": sc.x0, "P0": np.eye(4)},
        {"measurements": sc.zs[None].astype(np.float32)},
        {
            **linear_extra([1], None, "float32"),
            "selection_rule": "first (seed, r_scale) in seed-major order over r_scale in "
            "(1e-3, 1e-4, 1e-5) where a naive float32 filter survives >= 1,000 steps and fails "
            "within 1,000,000",
        },
        args.force,
    )

    # S5: 15-state INS error-state model with GPS position and velocity (larger state).
    model = ins_error_state_model(dt=0.01)
    sc = simulate(
        model["F"], model["H"], model["Q"], model["R"], model["x0"], model["P0"], STEPS, seed=5
    )
    write_scenario(
        args.out,
        "S5",
        "15-state INS error-state model at rest (position, velocity, attitude, accelerometer "
        "and gyro bias errors) with GPS position and velocity at 100 Hz",
        ["KF"],
        model,
        {"truth": sc.truth[None], "measurements": sc.zs[None]},
        linear_extra([5], 0.01),
        args.force,
    )


if __name__ == "__main__":
    main()
