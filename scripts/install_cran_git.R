# Install CRAN packages from the read-only git mirrors at github.com/cran.
# Used when CRAN itself cannot be reached (e.g. cloud sessions).
#
#   Rscript scripts/install_cran_git.R pkg [pkg@version ...]
#
# A bare name installs the newest release (the mirror's default branch);
# pkg@version installs that tag. Depends, Imports and LinkingTo are
# resolved recursively: a dependency that is missing, or older than the
# version the DESCRIPTION asks for, is also installed from its mirror.
# Packages go to the first entry of .libPaths() (set R_LIBS_USER to change).

args <- commandArgs(trailingOnly = TRUE)
if (!length(args)) stop("usage: Rscript scripts/install_cran_git.R pkg[@version] ...")

lib <- .libPaths()[1]
work <- tempfile("cran-git-")
dir.create(work)
base_pkgs <- rownames(installed.packages(priority = "base"))

installed_version <- function(pkg) {
  ip <- installed.packages(lib.loc = .libPaths())
  if (!pkg %in% rownames(ip)) return(NULL)
  package_version(ip[pkg, "Version"])
}

parse_deps <- function(desc) {
  fields <- intersect(c("Depends", "Imports", "LinkingTo"), colnames(desc))
  txt <- paste(na.omit(desc[1, fields]), collapse = ",")
  parts <- trimws(strsplit(gsub("\\s+", " ", txt), ",")[[1]])
  parts <- parts[nzchar(parts)]
  out <- character()
  for (p in parts) {
    name <- trimws(sub("\\(.*$", "", p))
    if (name == "R" || name %in% base_pkgs) next
    req <- NA_character_
    m <- regmatches(p, regexec("\\(\\s*>=?\\s*([0-9.-]+)\\s*\\)", p))[[1]]
    if (length(m) == 2) req <- m[2]
    out[name] <- req
  }
  out
}

done <- character()

install_one <- function(pkg, version = NULL, need = NULL) {
  if (pkg %in% done) return(invisible())
  have <- installed_version(pkg)
  if (is.null(version) && !is.null(have) &&
      (is.null(need) || is.na(need) || have >= package_version(need))) {
    return(invisible())
  }
  if (!is.null(version) && !is.null(have) && have == package_version(version)) {
    done <<- c(done, pkg)
    return(invisible())
  }
  dir <- file.path(work, pkg)
  ref <- if (is.null(version)) character() else c("--branch", version)
  url <- sprintf("https://github.com/cran/%s", pkg)
  message("== clone ", pkg, if (!is.null(version)) paste0("@", version))
  rc <- system2("git", c("clone", "-q", "--depth", "1", ref, url, dir))
  if (rc != 0) stop("git clone failed: ", url, " ", paste(ref, collapse = " "))
  desc <- read.dcf(file.path(dir, "DESCRIPTION"))
  deps <- parse_deps(desc)
  for (d in names(deps)) install_one(d, need = deps[[d]])
  message("== install ", pkg, " ", desc[1, "Version"])
  rc <- system2(file.path(R.home("bin"), "R"),
                c("CMD", "INSTALL", "--no-docs", "--no-multiarch",
                  "-l", shQuote(lib), shQuote(dir)))
  if (rc != 0) stop("R CMD INSTALL failed: ", pkg)
  done <<- c(done, pkg)
  invisible()
}

for (a in args) {
  bits <- strsplit(a, "@", fixed = TRUE)[[1]]
  install_one(bits[1], if (length(bits) > 1) bits[2] else NULL)
}
unlink(work, recursive = TRUE)
