# Likelihood comparison of gamm4 fits with and without the examiner intercept,
# and the profile interval of the examiner SD, for tests/r_oracle/test_gamm4.py.
# Run from the repository root: Rscript tests/r_oracle/scripts/gamm4_compare.R
#
# gamm4 without `random` fits the smooth alone through glmer, so logLik(g$mer)
# of both fits is on the same Laplace scale. statract's variance interval is a
# Wald interval for the log SD; the profile interval is kept to bound how far
# the two may differ.
suppressMessages({
  library(gamm4)
  library(jsonlite)
})

d <- read.csv("tests/r_oracle/data/gamm_binary_a.csv")
d$examiner <- factor(d$examiner)
out <- "tests/r_oracle/fixtures/gamm4_compare.json"

g0 <- gamm4(y ~ s(pre_size_mm, k = 10), family = binomial, data = d)
g1 <- gamm4(y ~ s(pre_size_mm, k = 10), random = ~(1 | examiner), family = binomial, data = d)
a <- anova(g0$mer, g1$mer)
ci <- confint(profile(g1$mer, which = "theta_", signames = FALSE))

fits <- lapply(list(g0, g1), function(g) {
  ll <- logLik(g$mer)
  list(log_likelihood = as.numeric(ll), n_parameters = attr(ll, "df"), aic = AIC(g$mer), bic = BIC(g$mer))
})
meta <- list(r = R.version.string, gamm4 = as.character(packageVersion("gamm4")),
             lme4 = as.character(packageVersion("lme4")))
write_json(list(meta = meta, fits = fits, statistic = a$Chisq[2], df = a$Df[2], p_value = a[["Pr(>Chisq)"]][2],
                profile_sd = unname(ci["sd_(Intercept)|examiner", ])),
           out, digits = NA, auto_unbox = TRUE, pretty = TRUE)
