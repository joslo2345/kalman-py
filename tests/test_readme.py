"""Run the README's Python examples, in order, in one namespace: they must stay correct."""

import re
from pathlib import Path

import pytest

README = Path(__file__).resolve().parents[1] / "README.md"


def python_blocks() -> list[str]:
    return re.findall(r"```python\n(.*?)```", README.read_text(encoding="utf-8"), flags=re.DOTALL)


def test_readme_has_examples() -> None:
    assert len(python_blocks()) >= 4


def test_readme_examples_run() -> None:
    pytest.importorskip("jax")
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    namespace: dict[str, object] = {}
    try:
        for block in python_blocks():
            exec(compile(block, str(README), "exec"), namespace)  # noqa: S102 (our own README)
    finally:
        plt.close("all")

    # Spot-check that the examples did what the prose says.
    result = namespace["result"]
    assert result.means.shape == (200, 2)  # type: ignore[attr-defined]
    ukf_means = namespace["ukf_result"].means  # type: ignore[attr-defined]
    truth = namespace["truth"]
    rmse = float(((ukf_means[:, :2] - truth[:, :2]) ** 2).mean() ** 0.5)  # type: ignore[index]
    assert rmse < 5.0, "the UKF should track the target through the bearing wrap"
    assert namespace["fit"].R.shape == (1, 1)  # type: ignore[attr-defined]
