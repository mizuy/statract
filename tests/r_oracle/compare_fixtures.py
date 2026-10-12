"""Compare regenerated R oracle fixtures with saved copies.

Usage: python tests/r_oracle/compare_fixtures.py SAVED_DIR NEW_DIR NAME...

For each NAME, reads SAVED_DIR/NAME.json and NEW_DIR/NAME.json and checks every
number with |new - saved| <= atol + rtol * |saved|. Strings that differ (package
versions) are listed but do not count as failures. Exits 1 if any number is
outside the tolerance or the structure differs.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

RTOL = 1e-4
ATOL = 1e-6


def _walk(saved, new, path, out):
    if isinstance(saved, dict) and isinstance(new, dict):
        for key in sorted(set(saved) | set(new)):
            if key not in saved or key not in new:
                out["structure"].append(f"{path}/{key}")
            else:
                _walk(saved[key], new[key], f"{path}/{key}", out)
    elif isinstance(saved, list) and isinstance(new, list):
        if len(saved) != len(new):
            out["structure"].append(f"{path} (length {len(saved)} -> {len(new)})")
        for i, (a, b) in enumerate(zip(saved, new)):
            _walk(a, b, f"{path}[{i}]", out)
    elif isinstance(saved, bool) or isinstance(new, bool) or saved is None or new is None:
        if saved != new:
            out["structure"].append(f"{path} ({saved!r} -> {new!r})")
    elif isinstance(saved, (int, float)) and isinstance(new, (int, float)):
        diff = abs(new - saved)
        rel = diff / abs(saved) if saved else diff
        if rel > out["worst"][0]:
            out["worst"] = (rel, path)
        if diff > ATOL + RTOL * abs(saved):
            out["numbers"].append(f"{path}: {saved!r} -> {new!r}")
    elif saved != new:
        out["strings"].append(f"{path}: {saved!r} -> {new!r}")


def main() -> int:
    saved_dir, new_dir, *names = sys.argv[1:]
    status = 0
    for name in names:
        saved = json.loads((Path(saved_dir) / f"{name}.json").read_text())
        new = json.loads((Path(new_dir) / f"{name}.json").read_text())
        out = {"numbers": [], "structure": [], "strings": [], "worst": (0.0, "")}
        _walk(saved, new, "", out)
        ok = not out["numbers"] and not out["structure"]
        rel, where = out["worst"]
        print(f"{'ok  ' if ok else 'FAIL'} {name}: largest relative change {rel:.1e} {where}")
        for line in out["strings"][:5]:
            print(f"     text changed (ignored) {line}")
        for line in (out["structure"] + out["numbers"])[:10]:
            print(f"     {line}")
        status |= not ok
    return status


if __name__ == "__main__":
    sys.exit(main())
