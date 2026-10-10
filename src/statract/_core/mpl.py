"""Matplotlib Japanese font setup (japanize-matplotlib).

Import this module before drawing text that includes Japanese characters.
Safe to import multiple times. This is a private helper, not a public API.

Notes:
- ``japanize-matplotlib`` may need ``setuptools`` on Python 3.12+ (distutils).
- Avoid ``fontweight='bold'`` on Japanese text; matplotlib may fall back to a
  Latin-only bold face and render tofu (□) for some glyphs.
"""

from __future__ import annotations

try:
    import japanize_matplotlib  # noqa: F401
except ModuleNotFoundError as exc:
    if "distutils" not in str(exc):
        raise
    import setuptools  # noqa: F401  # provides distutils on Py3.12+

    import japanize_matplotlib  # noqa: F401

import matplotlib as _mpl

# Keep minus sign usable with CJK fonts; keep family after japanize.
_mpl.rcParams["axes.unicode_minus"] = False
if not _mpl.rcParams.get("font.family"):
    _mpl.rcParams["font.family"] = "IPAexGothic"
