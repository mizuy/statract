# IPTW oracle for tests/r_oracle/test_iptw.py (statract.propensity_weights).
# Run from the repository root: Rscript tests/r_oracle/scripts/iptw.R
#
# Hand-coded from WeightIt 1.x / cobalt 4.x formulas; regenerate with the
# packages when available (tests/r_oracle/scripts/iptw_weightit.R writes the
# same JSON with weightit(), trim(), and bal.tab()).
#
# WeightIt and cobalt are not on this machine (CRAN is blocked), so this script
# uses glm() for the propensity score and base R for the rest:
#   - weights for ATE / ATT / ATC / ATO from the fitted score (get_w_from_ps)
#   - stabilize = TRUE (ATE): times the s.weights share of the unit's own arm
#   - trim(at): quantile (type 3, as WeightIt) or count, from the top, lower = TRUE also
#     from the bottom; the focal arm of ATT / ATC is left alone
#   - bal.tab: factor levels split as cobalt does (all levels for 3+ levels,
#     the second level for 2 levels), binary covariates as raw differences,
#     continuous as SMD with the unweighted s.d. ("pooled" for ATE,
#     "treated" for ATT, "control" for ATC) or, for ATO, the s.d. of the
#     whole sample under the balancing weights (cobalt's "weighted"), weighted variances with cobalt's
#     reliability-weight formula, variance ratios treated / control, and
#     Kish effective sample sizes.
suppressPackageStartupMessages(library(jsonlite))

data_dir <- "tests/r_oracle/data"
out_path <- "tests/r_oracle/fixtures/iptw.json"

# --- data -------------------------------------------------------------------
set.seed(20261010)
n <- 400
age <- round(rnorm(n, 55, 10), 2)
bmi <- round(rlnorm(n, log(25), 0.15), 2)
smoke <- rbinom(n, 1, 0.35)
race <- sample(c("asian", "black", "white"), n, replace = TRUE, prob = c(0.2, 0.3, 0.5))
lp <- -0.4 + 0.04 * (age - 55) + 0.08 * (bmi - 25) + 0.7 * smoke +
  0.6 * (race == "black") - 0.5 * (race == "asian")
treat <- rbinom(n, 1, plogis(lp))
write.csv(data.frame(treat, age, bmi, smoke, race), file.path(data_dir, "iptw_a.csv"), row.names = FALSE)

set.seed(20261011)
n <- 250
age <- round(rnorm(n, 60, 8), 2)
sex <- sample(c("F", "M"), n, replace = TRUE)
stage <- sample(c("I", "II", "III"), n, replace = TRUE, prob = c(0.5, 0.3, 0.2))
sw <- round(runif(n, 0.5, 2), 4)
lp <- -0.2 + 0.05 * (age - 60) + 0.5 * (sex == "M") + 0.8 * (stage == "III")
treat <- rbinom(n, 1, plogis(lp))
age[c(7, 50, 123)] <- NA
write.csv(data.frame(treat, age, sex, stage, sw), file.path(data_dir, "iptw_b.csv"),
          row.names = FALSE, na = "")

read_sample <- function(name) read.csv(file.path(data_dir, paste0(name, ".csv")))

# --- weights ------------------------------------------------------------------
w_from_ps <- function(ps, t, estimand) {
  switch(estimand,
    ATE = ifelse(t == 1, 1 / ps, 1 / (1 - ps)),
    ATT = ifelse(t == 1, 1, ps / (1 - ps)),
    ATC = ifelse(t == 1, (1 - ps) / ps, 1),
    ATO = ifelse(t == 1, 1 - ps, ps)
  )
}

trim_w <- function(w, t, estimand, at, lower) {
  idx <- switch(estimand, ATT = t == 0, ATC = t == 1, rep(TRUE, length(w)))
  v <- w[idx]
  if (at < 1) {
    if (at < 0.5) at <- 1 - at
    top <- unname(quantile(v, at, type = 3))
    bottom <- if (lower) unname(quantile(v, 1 - at, type = 3)) else -Inf
  } else {
    top <- sort(v, decreasing = TRUE)[at + 1]
    bottom <- if (lower) sort(v)[at + 1] else -Inf
  }
  v[v > top] <- top
  v[v < bottom] <- bottom
  w[idx] <- v
  w
}

# --- balance ------------------------------------------------------------------
split_covs <- function(d, vars) {
  out <- list()
  for (v in vars) {
    x <- d[[v]]
    if (is.character(x) || is.factor(x)) {
      lev <- sort(unique(as.character(x)))
      if (length(lev) == 2) lev <- lev[2]
      for (l in lev) out[[paste0(v, "_", l)]] <- as.numeric(as.character(x) == l)
    } else {
      out[[v]] <- as.numeric(x)
    }
  }
  out
}

w_var <- function(x, w, binary) {
  w <- w / sum(w)
  m <- sum(w * x)
  if (binary) return(m * (1 - m))
  sum(w * (x - m)^2) / (1 - sum(w^2))
}

binarize <- function(x) {
  u <- unique(x)
  if (0 %in% u) as.numeric(x != 0) else as.numeric(x == max(u))
}

bal_row <- function(x, t, w, s, type, sd_denom, binary_std) {
  is_bin <- type == "Binary"
  std <- type != "Binary" || binary_std
  denom <- 1
  if (std) {
    v1 <- w_var(x[t == 1], s[t == 1], is_bin)
    v0 <- w_var(x[t == 0], s[t == 0], is_bin)
    denom <- sqrt(switch(sd_denom, pooled = (v1 + v0) / 2, treated = v1, control = v0,
                         weighted = w_var(x, w * s, is_bin)))
  }
  stat <- function(ww) {
    m1 <- sum(ww[t == 1] * x[t == 1]) / sum(ww[t == 1])
    m0 <- sum(ww[t == 0] * x[t == 0]) / sum(ww[t == 0])
    vr <- if (is_bin) NA else w_var(x[t == 1], ww[t == 1], FALSE) / w_var(x[t == 0], ww[t == 0], FALSE)
    c(m1, m0, (m1 - m0) / denom, vr)
  }
  c(stat(s), stat(w * s))
}

ess <- function(w) sum(w)^2 / sum(w^2)

# --- cases --------------------------------------------------------------------
cases <- list()
add_case <- function(id, sample, covs, estimand, stabilize = FALSE, trim = NULL,
                     trim_lower = FALSE, s_weights = NULL, binary = "raw") {
  d0 <- read_sample(sample)
  used <- c("treat", covs, s_weights)
  rows <- which(complete.cases(d0[, used]))
  d <- d0[rows, ]
  f <- reformulate(covs, response = "treat")
  s <- if (is.null(s_weights)) rep(1, nrow(d)) else d[[s_weights]]
  fam <- if (is.null(s_weights)) binomial() else quasibinomial()
  fit <- glm(f, data = d, family = fam, weights = s)
  ps <- unname(fitted(fit))
  t <- d$treat
  w <- w_from_ps(ps, t, estimand)
  if (stabilize) {
    p1 <- sum(s * t) / sum(s)
    w <- w * ifelse(t == 1, p1, 1 - p1)
  }
  if (!is.null(trim)) w <- trim_w(w, t, estimand, trim, trim_lower)

  sd_denom <- switch(estimand, ATT = "treated", ATC = "control", ATO = "weighted", "pooled")
  covlist <- c(list(prop.score = ps), split_covs(d, covs))
  types <- character(0)
  tab <- NULL
  for (nm in names(covlist)) {
    x <- covlist[[nm]]
    type <- if (nm == "prop.score") "Distance" else if (length(unique(x)) == 2) "Binary" else "Contin."
    if (type == "Binary") x <- binarize(x)
    types <- c(types, type)
    tab <- rbind(tab, bal_row(x, t, w, s, type, sd_denom, binary == "std"))
  }
  cases[[id]] <<- list(
    data = sample,
    formula = paste("treat ~", paste(covs, collapse = " + ")),
    estimand = estimand,
    stabilize = stabilize,
    trim = trim,
    trim_lower = trim_lower,
    s_weights = s_weights,
    binary = binary,
    rows = rows - 1L,
    ps = ps,
    weights = unname(w),
    balance = list(
      term = names(covlist),
      type = types,
      mean_treated_unadjusted = tab[, 1],
      mean_control_unadjusted = tab[, 2],
      diff_unadjusted = tab[, 3],
      variance_ratio_unadjusted = tab[, 4],
      mean_treated_adjusted = tab[, 5],
      mean_control_adjusted = tab[, 6],
      diff_adjusted = tab[, 7],
      variance_ratio_adjusted = tab[, 8]
    ),
    ess = list(
      control = c(ess(s[t == 0]), ess((w * s)[t == 0])),
      treated = c(ess(s[t == 1]), ess((w * s)[t == 1]))
    )
  )
}

covs_a <- c("age", "bmi", "smoke", "race")
add_case("a_ate", "iptw_a", covs_a, "ATE")
add_case("a_att", "iptw_a", covs_a, "ATT")
add_case("a_atc", "iptw_a", covs_a, "ATC")
add_case("a_ato", "iptw_a", covs_a, "ATO")
add_case("a_ate_stab", "iptw_a", covs_a, "ATE", stabilize = TRUE)
add_case("a_ate_trim95", "iptw_a", covs_a, "ATE", trim = 0.95)
add_case("a_att_trim90", "iptw_a", covs_a, "ATT", trim = 0.9)
add_case("a_atc_trim90_lower", "iptw_a", covs_a, "ATC", trim = 0.9, trim_lower = TRUE)
add_case("a_ate_trim5_lower", "iptw_a", covs_a, "ATE", trim = 5, trim_lower = TRUE)
add_case("a_ate_stab_trim99", "iptw_a", covs_a, "ATE", stabilize = TRUE, trim = 0.99)
add_case("a_ate_binary_std", "iptw_a", covs_a, "ATE", binary = "std")
covs_b <- c("age", "sex", "stage")
add_case("b_ate", "iptw_b", covs_b, "ATE")
add_case("b_ate_sw_stab", "iptw_b", covs_b, "ATE", stabilize = TRUE, s_weights = "sw")
add_case("b_att_sw", "iptw_b", covs_b, "ATT", s_weights = "sw")
add_case("b_ato_trim3", "iptw_b", covs_b, "ATO", trim = 3)

writeLines(toJSON(list(
  r_version = R.version.string,
  source = "hand-coded from WeightIt 1.x / cobalt 4.x formulas",
  cases = cases
), auto_unbox = TRUE, digits = NA, na = "string", null = "null"), out_path)
