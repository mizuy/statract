"""Compatibility shim. Implementation lives in ``studyloop.reporting``."""

from __future__ import annotations

import importlib as _importlib

from statract._deprecate import warn_moved

warn_moved("statract.reporting", "studyloop.reporting")

_impl = _importlib.import_module("studyloop.reporting")


def __getattr__(name: str):
    return getattr(_impl, name)


def __dir__() -> list[str]:
    return dir(_impl)


def _export_names() -> list[str]:
    names = getattr(_impl, "__all__", None)
    if names is None:
        names = [name for name in dir(_impl) if not name.startswith("_")]
    return list(names)


__all__ = _export_names()
for _name in __all__:
    globals()[_name] = getattr(_impl, _name)
del _name
