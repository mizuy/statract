"""Configure rpy2 before its first import."""

from __future__ import annotations

import os

os.environ.setdefault("RPY2_CFFI_MODE", "ABI")
