"""Run the documentation's code: every ```python block of each page (in order, one namespace
per page), and, as a slow test, every tutorial notebook."""

import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
PAGES = sorted(p for p in (ROOT / "docs").rglob("*.md") if "```python" in p.read_text())
NOTEBOOKS = sorted((ROOT / "examples").glob("*.ipynb"))


def test_documentation_has_runnable_pages_and_notebooks() -> None:
    names = {p.relative_to(ROOT / "docs").as_posix() for p in PAGES}
    assert {"getting-started.md", "migration.md", "guide/nonlinear.md"} <= names
    assert len(NOTEBOOKS) == 3


@pytest.mark.parametrize("page", PAGES, ids=lambda p: p.relative_to(ROOT / "docs").as_posix())
def test_page_examples_run(page: Path) -> None:
    pytest.importorskip("jax")
    pytest.importorskip("filterpy")
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    namespace: dict[str, object] = {}
    for block in re.findall(r"```python\n(.*?)```", page.read_text(), flags=re.DOTALL):
        exec(compile(block, str(page), "exec"), namespace)  # noqa: S102 (our own docs)


@pytest.mark.slow
@pytest.mark.parametrize("notebook", NOTEBOOKS, ids=lambda p: p.stem)
def test_notebook_runs(notebook: Path) -> None:
    nbformat = pytest.importorskip("nbformat")
    nbclient = pytest.importorskip("nbclient")
    pytest.importorskip("ipykernel")
    pytest.importorskip("jax")
    nb = nbformat.read(notebook, as_version=4)
    nbclient.NotebookClient(nb, timeout=300, kernel_name="python3").execute(
        cwd=str(notebook.parent)
    )
