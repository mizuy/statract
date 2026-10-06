"""Deprecation warnings for shims that point at studyloop."""

from __future__ import annotations

import warnings


def warn_moved(old: str, new: str) -> None:
    warnings.warn(
        f"{old} is deprecated; import {new}",
        DeprecationWarning,
        stacklevel=2,
    )
