#!/usr/bin/env bash
# Install R 4.3.3 and every R package used by the R oracle scripts
# (tests/r_oracle/scripts) and the speed benchmark (bench/run_r.R).
# For Ubuntu 24.04 cloud sessions, where CRAN is blocked but apt and
# github.com work. Run as root from the repository root:
#   bash scripts/cloud-install-r.sh      (or: task r:setup)
# Safe to run again; finished steps are skipped.
#
# apt gives R 4.3.3 and the package versions the fixtures were made with.
# Packages apt does not have are built from source from the CRAN git
# mirrors at github.com/cran (scripts/install_cran_git.R).
set -euo pipefail

sudo=""
[ "$(id -u)" = 0 ] || sudo="sudo"

APT_PKGS=(
  r-base-core r-base-dev
  r-cran-jsonlite r-cran-survival r-cran-sandwich r-cran-lmtest
  r-cran-mass r-cran-nnet r-cran-mgcv r-cran-lme4 r-cran-gamm4
  r-cran-glmmtmb r-cran-rms r-cran-hmisc r-cran-matchit r-cran-proc
  r-cran-ordinal r-cran-mice r-cran-marginaleffects r-cran-glmnet
  r-cran-epir r-cran-cmprsk r-cran-partykit
)

# Not in apt. survRM2 is pinned to the version in its fixture.
GIT_PKGS=(
  coxme survRM2@1.0-4 stdReg2 medicaldata causaldata cobalt WeightIt
)

missing=()
for p in "${APT_PKGS[@]}"; do
  dpkg -s "$p" >/dev/null 2>&1 || missing+=("$p")
done
if [ "${#missing[@]}" -gt 0 ]; then
  $sudo apt-get update -qq
  DEBIAN_FRONTEND=noninteractive $sudo apt-get install -y -qq \
    --no-install-recommends "${missing[@]}"
fi

$sudo Rscript "$(dirname "$0")/install_cran_git.R" "${GIT_PKGS[@]}"

R --version | head -1
