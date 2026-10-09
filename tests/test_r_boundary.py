"""Fix the line between the default package and the optional R bridge."""

from __future__ import annotations

import ast
import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest

_SRC = Path(__file__).resolve().parents[1] / "src" / "statract"
_ALLOWED = {
    Path("r/__init__.py"),
    Path("r/_env.py"),
}


def _imports_rpy2(tree: ast.AST) -> bool:
    for node in ast.walk(tree):
        names: list[str] = []
        if isinstance(node, ast.Import):
            names = [alias.name for alias in node.names]
        elif isinstance(node, ast.ImportFrom) and node.module:
            names = [node.module]
        for name in names:
            root = name.split(".", 1)[0]
            if root in {"rpy2", "rpy2_arrow"}:
                return True
    return False


def test_import_statract_does_not_load_rpy2() -> None:
    """``import statract`` and ``import statract.r`` stay short of loading rpy2."""
    code = """
import sys
import statract
import statract.r as rmod

def _loaded(name: str) -> bool:
    return name in sys.modules or any(key.startswith(name + ".") for key in sys.modules)

assert not _loaded("rpy2")
assert not _loaded("rpy2_arrow")
assert rmod._rpy2_loaded is False
assert statract is not None
"""
    completed = subprocess.run(
        [sys.executable, "-c", code],
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr


def test_run_without_the_extra_names_endolab_r() -> None:
    """A bridge call with no rpy2 raises ``RNotAvailableError`` naming ``statract[r]``."""
    if importlib.util.find_spec("rpy2") is not None:
        pytest.skip("rpy2 is installed; tests/test_r_bridge.py covers the live bridge")
    from statract import r as rmod

    with pytest.raises(rmod.RNotAvailableError, match=r"statract\[r\]"):
        rmod.run("1+1")


def test_rpy2_import_statements_stay_in_the_bridge() -> None:
    offenders: list[str] = []
    for path in sorted(_SRC.rglob("*.py")):
        rel = path.relative_to(_SRC)
        if rel in _ALLOWED:
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        if _imports_rpy2(tree):
            offenders.append(rel.as_posix())
    assert offenders == []
