"""MkDocs hook: publish the notebooks in examples/ as the Tutorials section.

The notebooks stay in examples/ (where they're run and linked from the README); this adds each
one to the site as tutorials/<name>.ipynb, wrapped like mkdocs-jupyter's own notebook files.
Hooks run after plugins, so mkdocs-jupyter's on_files never sees these files itself.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from mkdocs.structure.files import File, Files
from mkdocs_jupyter.plugin import NotebookFile


def on_files(files: Files, config: Any) -> Files:
    examples = Path(config.config_file_path).parent / "examples"
    for notebook in sorted(examples.glob("*.ipynb")):
        file = File.generated(config, f"tutorials/{notebook.name}", abs_src_path=str(notebook))
        wrapped = NotebookFile(file, **config)
        # NotebookFile skips File.__init__, so File's cached abs_src_path would be derived from a
        # src_dir of None (generated files have none); set it to the notebook itself.
        wrapped.abs_src_path = str(notebook)
        files.append(wrapped)
    return files
