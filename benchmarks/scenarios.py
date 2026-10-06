"""Load the shared benchmark scenarios from tests/vectors/ (format: tests/vectors/README.md)."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

VECTORS = Path(__file__).resolve().parents[1] / "tests" / "vectors"
_DTYPES = {"float64-le": np.dtype("<f8"), "float32-le": np.dtype("<f4")}


@dataclass(frozen=True)
class Scenario:
    id: str
    meta: dict[str, Any]
    F: np.ndarray
    Q: np.ndarray
    R: np.ndarray
    x0: np.ndarray
    P0: np.ndarray
    H: np.ndarray | None  # None for nonlinear measurement models (S3)
    measurements: np.ndarray  # (seeds, steps, m)
    truth: np.ndarray | None  # (seeds, steps, n)

    @property
    def n(self) -> int:
        return int(self.F.shape[0])

    @property
    def m(self) -> int:
        return int(self.R.shape[0])

    @property
    def dt(self) -> float | None:
        dt = self.meta["dt"]
        return None if dt is None else float(dt)

    @property
    def dtype(self) -> np.dtype[Any]:
        return np.dtype(self.meta["precision"])

    # Single-seed conveniences, matching the names in the repo guide's harness.
    @property
    def zs(self) -> np.ndarray:
        return self._only_seed(self.measurements)

    @property
    def truth_single(self) -> np.ndarray:
        assert self.truth is not None, f"{self.id} stores no truth"
        return self._only_seed(self.truth)

    @property
    def params(self) -> dict[str, np.ndarray]:
        assert self.H is not None, f"{self.id} has a nonlinear measurement model"
        return {k: getattr(self, k) for k in ("F", "H", "Q", "R", "x0", "P0")}

    def _only_seed(self, a: np.ndarray) -> np.ndarray:
        if a.shape[0] != 1:
            raise ValueError(f"{self.id} has {a.shape[0]} seeds; index measurements directly")
        return a[0]


def load(sid: str, root: Path = VECTORS, verify: bool = True) -> Scenario:
    """Load scenario ``sid`` (e.g. "S2"), checking every file's SHA-256 against meta.json."""
    directory = root / sid
    meta = json.loads((directory / "meta.json").read_text())
    arrays = {}
    for name, info in meta["files"].items():
        raw = (directory / info["path"]).read_bytes()
        if verify and hashlib.sha256(raw).hexdigest() != info["sha256"]:
            raise ValueError(f"{sid}/{info['path']} does not match its sha256 in meta.json")
        arrays[name] = np.frombuffer(raw, dtype=_DTYPES[info["dtype"]]).reshape(info["shape"])
    model = meta["model"]
    dtype = np.dtype(meta["precision"])

    def matrix(key: str) -> np.ndarray:
        return np.asarray(model[key], dtype=np.float64).astype(dtype)

    return Scenario(
        id=sid,
        meta=meta,
        F=matrix("F"),
        Q=matrix("Q"),
        R=matrix("R"),
        x0=matrix("x0"),
        P0=matrix("P0"),
        H=matrix("H") if "H" in model else None,
        measurements=arrays["measurements"].astype(dtype),
        truth=arrays.get("truth"),
    )
