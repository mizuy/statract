#!/usr/bin/env bash
# Regenerate R oracle fixtures, show what changed, and run the matching tests.
# Run from the repository root: bash tests/r_oracle/run.sh [--check] [name ...]
# With no names, every tests/r_oracle/scripts/*.R is run. A name is the script
# file name without .R (e.g. ordinal gamm4).
# --check compares the new output with the saved fixtures (rtol 1e-4) and then
# puts the saved fixtures back, so package versions on this machine do not
# matter. Without it, the fixtures are overwritten.
# Missing R packages are installed from CRAN into the user library
# (R_LIBS_USER). Set SKIP_INSTALL=1 to skip that.
set -uo pipefail

check=0
if [ "${1:-}" = "--check" ]; then
  check=1
  shift
fi

scripts=tests/r_oracle/scripts
fixtures=tests/r_oracle/fixtures
if [ "$#" -eq 0 ]; then
  set -- $(ls "$scripts"/*.R | xargs -n1 basename | sed 's/\.R$//')
fi
names=("$@")

for name in "${names[@]}"; do
  [ -f "$scripts/$name.R" ] || { echo "no such script: $scripts/$name.R" >&2; exit 2; }
done

if [ "${SKIP_INSTALL:-0}" != 1 ]; then
  pkgs=$(for name in "${names[@]}"; do cat "$scripts/$name.R"; done \
    | grep -oE 'library\([A-Za-z0-9.]+\)|[A-Za-z][A-Za-z0-9.]*::' \
    | sed -E 's/library\(|\)|:://g' | sort -u | tr '\n' ' ')
  PKGS="$pkgs" Rscript -e '
    pkgs <- strsplit(trimws(Sys.getenv("PKGS")), " +")[[1]]
    miss <- setdiff(pkgs, rownames(installed.packages()))
    if (length(miss)) {
      lib <- Sys.getenv("R_LIBS_USER")
      dir.create(lib, recursive = TRUE, showWarnings = FALSE)
      install.packages(miss, lib = lib, repos = "https://cloud.r-project.org")
      left <- setdiff(miss, rownames(installed.packages(lib.loc = c(lib, .libPaths()))))
      if (length(left)) stop("could not install: ", paste(left, collapse = " "))
    }' || exit 1
fi

saved=""
if [ "$check" = 1 ]; then
  saved=$(mktemp -d)
  cp "$fixtures"/*.json "$saved"/
fi

failed=()
for name in "${names[@]}"; do
  echo "== Rscript $scripts/$name.R"
  Rscript "$scripts/$name.R" || failed+=("$name")
done

status=0
if [ "$check" = 1 ]; then
  echo "== compare with saved fixtures"
  checked=()
  for name in "${names[@]}"; do
    [ -f "$saved/$name.json" ] && checked+=("$name")
  done
  if [ "${#checked[@]}" -gt 0 ]; then
    uv run python tests/r_oracle/compare_fixtures.py "$saved" "$fixtures" "${checked[@]}" || status=1
  fi
  cp "$saved"/*.json "$fixtures"/
  rm -rf "$saved"
else
  echo "== fixture changes"
  git diff --stat -- "$fixtures"/

  tests=()
  for name in "${names[@]}"; do
    t="tests/r_oracle/test_$name.py"
    [ -f "$t" ] && tests+=("$t")
  done
  if [ "${#tests[@]}" -gt 0 ]; then
    echo "== pytest"
    uv run pytest -q -p no:cacheprovider "${tests[@]}" || status=1
  fi
fi

if [ "${#failed[@]}" -gt 0 ]; then
  echo "R scripts that failed: ${failed[*]}" >&2
  status=1
fi
exit $status
