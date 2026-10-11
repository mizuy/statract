# IPTW outcome-model oracle for tests/r_oracle/test_iptw_outcome.py
# (statract.weighted_glm and PropensityWeights.effect).
# Run from the repository root: Rscript tests/r_oracle/scripts/iptw_outcome.R
#
# WeightIt (>= 1.0) is not installable on this machine, so this script codes
# glm_weightit()'s M-estimation by hand with glm() and base R:
#   - the propensity score is glm(treat ~ covs, binomial) with s.weights as
#     prior weights; the weights are WeightIt's get_w_from_ps
#   - the outcome is glm(y ~ ..., weights = w * s)
#   - vcov = "asympt" stacks the outcome score (w * s * x * (y - mu) mu.eta / V)
#     on the propensity score (s * z * (a - e)) and, for stabilize = TRUE, on
#     the treated share (s * (a - p1)). V = A^-1 B A^-T with the analytic
#     Jacobian A; the derivative of the weights in the propensity coefficients
#     is dw/de * e (1 - e) * z (Lunceford and Davidian 2004 for ATE; the same
#     chain rule for ATT, ATC, and ATO). A central-difference Jacobian of the
#     stacked estimating equations checks it.
#   - vcov = "HC0" is the outcome block alone, at the returned coefficients.
#     It agrees with sandwich::vcovHC(type = "HC0") on the weighted glm, or
#     sandwich::vcovCL(type = "HC0") with a cluster (G / (G - 1) adjustment),
#     up to glm's convergence error; both are written.
#     The clustered "asympt" meat sums the stacked functions within clusters
#     with the same G / (G - 1).
#   - the marginal effects stack the weighted arm means (m1, m0) on the
#     propensity model (the saturated model y ~ treat) and use the delta
#     method for the difference, log risk ratio, and log odds ratio.
suppressPackageStartupMessages({
  library(jsonlite)
  library(sandwich)
})

data_dir <- "tests/r_oracle/data"
out_path <- "tests/r_oracle/fixtures/iptw_outcome.json"

# --- data -------------------------------------------------------------------
set.seed(20261012)
n <- 500
site <- sample(sprintf("s%02d", 1:30), n, replace = TRUE)
u <- rnorm(30, 0, 0.4)[as.integer(factor(site))]
age <- round(rnorm(n, 60, 9), 2)
bmi <- round(rlnorm(n, log(25), 0.15), 2)
smoke <- rbinom(n, 1, 0.3)
race <- sample(c("asian", "black", "white"), n, replace = TRUE, prob = c(0.2, 0.3, 0.5))
sw <- round(runif(n, 0.5, 2), 4)
lp <- -0.3 + 0.05 * (age - 60) + 0.06 * (bmi - 25) + 0.6 * smoke +
  0.5 * (race == "black") - 0.4 * (race == "asian") + u
treat <- rbinom(n, 1, plogis(lp))
ybin <- rbinom(n, 1, plogis(-1 + 0.6 * treat + 0.03 * (age - 60) + 0.5 * smoke + u))
ycont <- round(10 + 1.5 * treat + 0.1 * (age - 60) + 0.2 * (bmi - 25) + rnorm(n, 0, 2), 3)
ycount <- rpois(n, exp(0.3 + 0.4 * treat + 0.02 * (age - 60) + 0.3 * smoke))
ypos <- round(rgamma(n, shape = 3, rate = 3 / exp(1 + 0.3 * treat + 0.01 * (age - 60))), 4)
write.csv(data.frame(treat, age, bmi, smoke, race, site, sw, ybin, ycont, ycount, ypos),
          file.path(data_dir, "iptw_outcome_a.csv"), row.names = FALSE)

d <- read.csv(file.path(data_dir, "iptw_outcome_a.csv"))

# --- propensity score and weights ---------------------------------------------
ps_parts <- function(covs, estimand, stabilize, s_weights) {
  s <- if (is.null(s_weights)) rep(1, nrow(d)) else d[[s_weights]]
  fam <- if (is.null(s_weights)) binomial() else quasibinomial()
  f <- reformulate(covs, response = "treat")
  fit <- glm(f, data = d, family = fam, weights = s)
  Z <- model.matrix(fit)
  a <- d$treat
  list(Z = Z, a = a, s = s, gamma = coef(fit), estimand = estimand, stabilize = stabilize,
       p1 = sum(s * a) / sum(s))
}

w_from_ps <- function(e, a, estimand) {
  switch(estimand,
    ATE = ifelse(a == 1, 1 / e, 1 / (1 - e)),
    ATT = ifelse(a == 1, 1, e / (1 - e)),
    ATC = ifelse(a == 1, (1 - e) / e, 1),
    ATO = ifelse(a == 1, 1 - e, e)
  )
}

dw_de <- function(e, a, estimand) {
  switch(estimand,
    ATE = ifelse(a == 1, -1 / e^2, 1 / (1 - e)^2),
    ATT = ifelse(a == 1, 0, 1 / (1 - e)^2),
    ATC = ifelse(a == 1, -1 / e^2, 0),
    ATO = ifelse(a == 1, -1, 1)
  )
}

# Weights at (gamma, p1) and their derivatives.
weights_at <- function(P, gamma, p1) {
  e <- plogis(drop(P$Z %*% gamma))
  w0 <- w_from_ps(e, P$a, P$estimand)
  c <- if (P$stabilize) ifelse(P$a == 1, p1, 1 - p1) else rep(1, length(e))
  list(e = e, w = w0 * c,
       dg = (c * dw_de(e, P$a, P$estimand) * e * (1 - e)) * P$Z,
       dp = w0 * ifelse(P$a == 1, 1, -1))
}

# PS block of the stacked functions and its Jacobian.
ps_psi <- function(P, gamma, p1) {
  e <- plogis(drop(P$Z %*% gamma))
  out <- P$s * (P$a - e) * P$Z
  if (P$stabilize) out <- cbind(out, P$s * (P$a - p1))
  out
}

ps_jac <- function(P) {
  e <- plogis(drop(P$Z %*% P$gamma))
  J <- -crossprod(P$Z, P$Z * (P$s * e * (1 - e)))
  if (P$stabilize) {
    J <- rbind(cbind(J, 0), c(rep(0, ncol(J)), -sum(P$s)))
  }
  J
}

meat <- function(psi, cluster) {
  if (is.null(cluster)) return(crossprod(psi))
  g <- length(unique(cluster))
  g / (g - 1) * crossprod(rowsum(psi, cluster, reorder = FALSE))
}

sandwich_v <- function(J, psi, cluster) {
  Ji <- solve(J)
  Ji %*% meat(psi, cluster) %*% t(Ji)
}

# Central-difference Jacobian of the summed stacked functions.
num_jac <- function(f, theta) {
  k <- length(theta)
  J <- matrix(0, k, k)
  for (j in seq_len(k)) {
    h <- 1e-5 * max(1, abs(theta[j]))
    up <- theta; up[j] <- up[j] + h
    dn <- theta; dn[j] <- dn[j] - h
    J[, j] <- (colSums(f(up)) - colSums(f(dn))) / (2 * h)
  }
  J
}

check_jac <- function(J, Jn, id) {
  err <- max(abs(J - Jn)) / max(abs(J))
  if (err > 1e-6) stop(sprintf("%s: analytic and numeric Jacobians differ by %g", id, err))
}

fam_of <- function(family) switch(family,
  gaussian = gaussian(), binomial = quasibinomial(), poisson = quasipoisson(), gamma = Gamma())

# --- weighted GLM --------------------------------------------------------------
glm_cases <- list()
add_glm <- function(id, outcome, covs, estimand, family, stabilize = FALSE, s_weights = NULL,
                    cluster = NULL) {
  P <- ps_parts(covs, estimand, stabilize, s_weights)
  W <- weights_at(P, P$gamma, P$p1)
  d$.w <- W$w * P$s
  fam <- fam_of(family)
  fit <- glm(as.formula(outcome), data = d, family = fam, weights = .w)
  X <- model.matrix(fit)
  y <- fit$y
  beta <- coef(fit)
  cl <- if (is.null(cluster)) NULL else d[[cluster]]

  out_psi <- function(beta, w) {
    eta <- drop(X %*% beta)
    mu <- fam$linkinv(eta)
    (w * P$s * (y - mu) * fam$mu.eta(eta) / fam$variance(mu)) * X
  }
  eta <- drop(X %*% beta)
  mu <- fam$linkinv(eta)
  r <- fam$mu.eta(eta) / fam$variance(mu)
  U <- (P$s * (y - mu) * r) * X
  Jbb <- -crossprod(X, X * (W$w * P$s * fam$mu.eta(eta)^2 / fam$variance(mu)))
  cross <- crossprod(U, W$dg)
  if (stabilize) cross <- cbind(cross, crossprod(U, W$dp))
  Jps <- ps_jac(P)
  p <- length(beta)
  q <- ncol(Jps)
  J <- rbind(cbind(Jbb, cross), cbind(matrix(0, q, p), Jps))
  psi <- cbind(out_psi(beta, W$w), ps_psi(P, P$gamma, P$p1))

  stacked <- function(theta) {
    g <- theta[p + seq_len(ncol(P$Z))]
    p1 <- if (stabilize) theta[length(theta)] else P$p1
    cbind(out_psi(theta[seq_len(p)], weights_at(P, g, p1)$w), ps_psi(P, g, p1))
  }
  theta <- c(beta, P$gamma, if (stabilize) P$p1)
  check_jac(J, num_jac(stacked, theta), id)

  V <- sandwich_v(J, psi, cl)[seq_len(p), seq_len(p)]
  Vhc <- if (is.null(cl)) vcovHC(fit, type = "HC0") else vcovCL(fit, cluster = cl, type = "HC0")
  Vhc_hand <- sandwich_v(Jbb, out_psi(beta, W$w), cl)
  # vcovHC uses the working weights of glm's last IRLS step, so it differs
  # from the analytic value at the returned coefficients by glm's
  # convergence error (up to about 1e-5 relative here).
  if (max(abs(Vhc - Vhc_hand)) > 1e-4 * max(abs(Vhc))) stop(paste(id, "HC0 differs from sandwich"))
  glm_cases[[id]] <<- list(
    outcome = outcome, covs = covs, ps_formula = paste("treat ~", paste(covs, collapse = " + ")),
    estimand = estimand, family = family, stabilize = stabilize, s_weights = s_weights,
    cluster = cluster,
    terms = names(beta), coefficients = unname(beta),
    vcov_asympt = unname(V), vcov_hc0 = unname(Vhc_hand), vcov_hc0_sandwich = unname(Vhc)
  )
}

covs <- c("age", "bmi", "smoke", "race")
add_glm("bin_ate", "ybin ~ treat", covs, "ATE", "binomial")
add_glm("bin_att_adj", "ybin ~ treat + age + smoke", covs, "ATT", "binomial")
add_glm("cont_atc_adj", "ycont ~ treat + age", covs, "ATC", "gaussian")
add_glm("count_ato", "ycount ~ treat", covs, "ATO", "poisson")
add_glm("cont_ate_stab", "ycont ~ treat + bmi", covs, "ATE", "gaussian", stabilize = TRUE)
add_glm("bin_ate_sw", "ybin ~ treat", c("age", "smoke"), "ATE", "binomial", s_weights = "sw")
add_glm("bin_ate_cluster", "ybin ~ treat + age", covs, "ATE", "binomial", cluster = "site")
add_glm("pos_ate_gamma", "ypos ~ treat", covs, "ATE", "gamma")
add_glm("count_att_sw_cluster", "ycount ~ treat + smoke", covs, "ATT", "poisson",
        s_weights = "sw", cluster = "site")

# --- marginal effects --------------------------------------------------------
effect_cases <- list()
add_effect <- function(id, outcome, covs, estimand, stabilize = FALSE, s_weights = NULL,
                       cluster = NULL) {
  P <- ps_parts(covs, estimand, stabilize, s_weights)
  W <- weights_at(P, P$gamma, P$p1)
  y <- d[[outcome]]
  a <- P$a
  s <- P$s
  m1 <- sum(W$w * s * a * y) / sum(W$w * s * a)
  m0 <- sum(W$w * s * (1 - a) * y) / sum(W$w * s * (1 - a))
  mean_psi <- function(m, w) cbind(w * s * a * (y - m[1]), w * s * (1 - a) * (y - m[2]))
  U <- cbind(s * a * (y - m1), s * (1 - a) * (y - m0))
  Jbb <- diag(c(-sum(W$w * s * a), -sum(W$w * s * (1 - a))))
  cross <- crossprod(U, W$dg)
  if (stabilize) cross <- cbind(cross, crossprod(U, W$dp))
  Jps <- ps_jac(P)
  q <- ncol(Jps)
  J <- rbind(cbind(Jbb, cross), cbind(matrix(0, q, 2), Jps))
  psi <- cbind(mean_psi(c(m1, m0), W$w), ps_psi(P, P$gamma, P$p1))
  stacked <- function(theta) {
    g <- theta[2 + seq_len(ncol(P$Z))]
    p1 <- if (stabilize) theta[length(theta)] else P$p1
    cbind(mean_psi(theta[1:2], weights_at(P, g, p1)$w), ps_psi(P, g, p1))
  }
  check_jac(J, num_jac(stacked, c(m1, m0, P$gamma, if (stabilize) P$p1)), id)
  cl <- if (is.null(cluster)) NULL else d[[cluster]]
  Va <- sandwich_v(J, psi, cl)[1:2, 1:2]
  Vh <- sandwich_v(Jbb, mean_psi(c(m1, m0), W$w), cl)
  grads <- list(
    difference = c(1, -1),
    ratio = c(1 / m1, -1 / m0),
    odds_ratio = c(1 / (m1 * (1 - m1)), -1 / (m0 * (1 - m0)))
  )
  est <- c(difference = m1 - m0, ratio = m1 / m0,
           odds_ratio = (m1 / (1 - m1)) / (m0 / (1 - m0)))
  se <- function(V) sapply(grads, function(g) sqrt(drop(t(g) %*% V %*% g)))
  # Check the difference against the slope of the weighted linear model
  # y ~ treat (the Chapter 5 estimator) under HC0.
  d$.w <- W$w * s
  lin <- glm(as.formula(paste(outcome, "~ treat")), data = d, weights = .w)
  lin_hc0 <- if (is.null(cl)) vcovHC(lin, type = "HC0") else vcovCL(lin, cluster = cl, type = "HC0")
  if (abs(sqrt(lin_hc0[2, 2]) - se(Vh)[["difference"]]) > 1e-10) stop(paste(id, "HC0 difference"))
  effect_cases[[id]] <<- list(
    outcome = outcome, ps_formula = paste("treat ~", paste(covs, collapse = " + ")),
    estimand = estimand, stabilize = stabilize, s_weights = s_weights, cluster = cluster,
    mean_treated = m1, mean_control = m0,
    measure = names(est), estimate = unname(est),
    se_asympt = unname(se(Va)), se_hc0 = unname(se(Vh))
  )
}

add_effect("ate", "ybin", covs, "ATE")
add_effect("att", "ybin", covs, "ATT")
add_effect("atc", "ybin", covs, "ATC")
add_effect("ato", "ybin", covs, "ATO")
add_effect("ate_stab", "ybin", covs, "ATE", stabilize = TRUE)
add_effect("ate_sw", "ybin", c("age", "smoke"), "ATE", s_weights = "sw")
add_effect("ate_cluster", "ybin", covs, "ATE", cluster = "site")

writeLines(toJSON(list(
  r_version = R.version.string,
  source = "hand-coded M-estimation of WeightIt::glm_weightit (vcov = 'asympt' / 'HC0')",
  glm = glm_cases,
  effect = effect_cases
), auto_unbox = TRUE, digits = NA, null = "null"), out_path)
