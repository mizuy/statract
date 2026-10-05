"""Deprecated shims for removed propensity-score / GLM helpers.

The former ``psmatch`` and ``GLMHelper`` implementations were incomplete or
incorrect. Instantiating either class raises ``NotImplementedError``.

Use ``statract.match_sample`` for matching, and ``statract.fit_glm``
with ``Fit.tidy`` for regression tables.
"""

from __future__ import annotations

import warnings
from typing import Any

_PSMATCH_MSG = (
    "psmatch was removed; use statract.match_sample "
    "(see docs/api/stat/matching.md)"
)
_GLMHELPER_MSG = (
    "GLMHelper was removed; use statract.fit_glm and Fit.tidy "
    "(see docs/api/stat/fit.md)"
)


class psmatch:
    """Deprecated shim. Raises immediately; use ``match_sample`` instead."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        warnings.warn(
            "psmatch is deprecated; use statract.match_sample",
            DeprecationWarning,
            stacklevel=2,
        )
        raise NotImplementedError(_PSMATCH_MSG)


class GLMHelper:
    """Deprecated shim. Raises immediately; use ``fit_glm`` and ``Fit.tidy``."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        warnings.warn(
            "GLMHelper is deprecated; use statract.fit_glm and Fit.tidy",
            DeprecationWarning,
            stacklevel=2,
        )
        raise NotImplementedError(_GLMHELPER_MSG)
