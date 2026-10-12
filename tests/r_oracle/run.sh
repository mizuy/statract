#!/usr/bin/env bash
# Regenerate R oracle fixtures, show what changed, and run the matching tests.
# Run from the repository root: bash tests/r_oracle/run.sh [name ...]
# With no names, every tests/r_oracle/scripts/*.R is run. A name is the script
# file name without .R (e.g. ordinal gamm4). Set SKIP_INSTALL=1 to skip
# installing missing R packages from CRAN.
set -uo pipefail

scripts=tests/r_oracle/scripts
if [ "$#" -eq 0 ]; then
  set -- $(ls "$scripts"/*.R | xargs -n1 basename | sed 's/\.R$//')
fi

files=()
for name in "$@"; do
  f="$scripts/$name.R"
  [ -f "$f" ] || { echo "no such script: $f" >&2; exit 2; }
  files+=("$f")
done

if [ "${SKIP_INSTALL:-0}" != 1 ]; then
  pkgs=$(grep -hoE 'library\([A-Za-z0-9.]+\)|[A-Za-z][A-Za-z0-9.]*::' "${files[@]}" \
    | sed -E 's/library\(|\)|:://g' | sort -u | tr '\n' ' ')
  Rscript -e "pkgs <- strsplit('$pkgs', ' ')[[1]]
    miss <- setdiff(pkgs, rownames(installed.packages()))
    if (length(miss)) install.packages(miss, repos = 'https://cloud.r-project.org')" || exit 1
fi

failed=()
for i in "${!files[@]}"; do
  echo "== Rscript ${files[$i]}"
  Rscript "${files[$i]}" || failed+=("$1")
  shift
done

echo "== fixture changes"
git diff --stat -- tests/r_oracle/fixtures/

tests=()
for f in "${files[@]}"; do
  t="tests/r_oracle/test_$(basename "$f" .R).py"
  [ -f "$t" ] && tests+=("$t")
done
status=0
if [ "${#tests[@]}" -gt 0 ]; then
  echo "== pytest"
  uv run pytest -q -p no:cacheprovider "${tests[@]}" || status=1
fi

if [ "${#failed[@]}" -gt 0 ]; then
  echo "R scripts that failed: ${failed[*]}" >&2
  status=1
fi
exit $status
