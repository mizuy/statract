# stats::t.test, wilcox.test, mcnemar.test, binom.test, prop.test, p.adjust
# oracle for statract.models.htest. Run from the repo root:
#   Rscript tests/r_oracle/scripts/htest.R
suppressPackageStartupMessages(library(jsonlite))

out <- "tests/r_oracle/fixtures/htest.json"

htest_out <- function(h) {
  ci <- h$conf.int
  list(
    statistic = unname(h$statistic),
    parameter = if (is.null(h$parameter)) NULL else unname(h$parameter),
    p_value = h$p.value,
    conf_int = if (is.null(ci)) NULL else as.numeric(ci),
    conf_level = if (is.null(ci)) NULL else attr(ci, "conf.level"),
    estimates = if (is.null(h$estimate)) NULL else as.list(h$estimate),
    method = h$method,
    alternative = h$alternative
  )
}

cases <- list()
add <- function(fn, args, h) {
  cases[[length(cases) + 1]] <<- list(fn = fn, args = args, expect = htest_out(h))
}

# --- data -------------------------------------------------------------------
set.seed(20261008)
a <- round(rnorm(18, 1, 2), 3)
b <- round(rnorm(23, 0.2, 1.3), 3)
a_na <- a; a_na[c(3, 11)] <- NA
b_pair <- round(a + rnorm(18, 0.4, 1), 3); b_pair[7] <- NA
tied <- round(rnorm(30, 0.5, 1), 0)          # many ties and zeros
tied2 <- round(rnorm(26, -0.2, 1), 0)
big <- round(rnorm(64, 0.3, 1), 4)           # n >= 50, no ties
big2 <- round(rnorm(55, 0, 1), 4)
small <- c(1.83, 0.50, 1.62, 2.48, 1.68, 1.88, 1.55, 3.06, 1.30, 0.21, -0.4)
small2 <- c(0.878, 0.647, 0.598, 2.05, 1.06, 1.29, 1.06, 3.14, 1.29)

# --- t.test -----------------------------------------------------------------
for (alt in c("two.sided", "less", "greater")) {
  add("t_test", list(x = a, mu = 0.3, alternative = alt), t.test(a, mu = 0.3, alternative = alt))
  add("t_test", list(x = a_na, y = b, alternative = alt), t.test(a_na, b, alternative = alt))
}
add("t_test", list(x = a, y = b, var_equal = TRUE), t.test(a, b, var.equal = TRUE))
add("t_test", list(x = a, y = b, var_equal = TRUE, conf_level = 0.9, mu = -0.5),
    t.test(a, b, var.equal = TRUE, conf.level = 0.9, mu = -0.5))
add("t_test", list(x = a, y = b_pair, paired = TRUE), t.test(a, b_pair, paired = TRUE))
add("t_test", list(x = a_na, y = b_pair, paired = TRUE, alternative = "greater"),
    t.test(a_na, b_pair, paired = TRUE, alternative = "greater"))

# --- wilcox.test ------------------------------------------------------------
w <- function(args, ...) suppressWarnings(wilcox.test(...))
for (alt in c("two.sided", "less", "greater")) {
  add("wilcox_test", list(x = small, mu = 0.5, alternative = alt, conf_int = TRUE),
      w(NULL, small, mu = 0.5, alternative = alt, conf.int = TRUE))
  add("wilcox_test", list(x = small, y = small2, alternative = alt, conf_int = TRUE),
      w(NULL, small, small2, alternative = alt, conf.int = TRUE))
  add("wilcox_test", list(x = tied, alternative = alt, conf_int = TRUE),
      w(NULL, tied, alternative = alt, conf.int = TRUE))
  add("wilcox_test", list(x = tied, y = tied2, alternative = alt, conf_int = TRUE),
      w(NULL, tied, tied2, alternative = alt, conf.int = TRUE))
  add("wilcox_test", list(x = big, y = big2, alternative = alt, conf_int = TRUE),
      w(NULL, big, big2, alternative = alt, conf.int = TRUE))
}
add("wilcox_test", list(x = tied, correct = FALSE, conf_int = TRUE), w(NULL, tied, correct = FALSE, conf.int = TRUE))
add("wilcox_test", list(x = tied, y = tied2, correct = FALSE, conf_int = TRUE),
    w(NULL, tied, tied2, correct = FALSE, conf.int = TRUE))
add("wilcox_test", list(x = big, mu = 0.1, conf_int = TRUE), w(NULL, big, mu = 0.1, conf.int = TRUE))
add("wilcox_test", list(x = a, y = b, exact = FALSE), w(NULL, a, b, exact = FALSE))
add("wilcox_test", list(x = a, y = b, mu = 0.4, conf_int = TRUE, conf_level = 0.9),
    w(NULL, a, b, mu = 0.4, conf.int = TRUE, conf.level = 0.9))
add("wilcox_test", list(x = a_na, y = b_pair, paired = TRUE, conf_int = TRUE),
    w(NULL, a_na, b_pair, paired = TRUE, conf.int = TRUE))
add("wilcox_test", list(x = small[1:4], conf_int = TRUE), w(NULL, small[1:4], conf.int = TRUE))

# --- mcnemar.test -----------------------------------------------------------
t2 <- matrix(c(794, 86, 150, 570), 2)
t3 <- matrix(c(20, 5, 3, 9, 30, 4, 2, 11, 25), 3)
add("mcnemar_test", list(x = t2), mcnemar.test(t2))
add("mcnemar_test", list(x = t2, correct = FALSE), mcnemar.test(t2, correct = FALSE))
add("mcnemar_test", list(x = t3), mcnemar.test(t3))
set.seed(11)
before <- sample(c("neg", "pos"), 60, TRUE, prob = c(0.6, 0.4))
after <- ifelse(runif(60) < 0.25, ifelse(before == "pos", "neg", "pos"), before)
before[c(4, 9)] <- NA
add("mcnemar_test", list(x = before, y = after), mcnemar.test(before, after))

# --- binom.test -------------------------------------------------------------
for (alt in c("two.sided", "less", "greater")) {
  add("binom_test", list(x = 7, n = 20, p = 0.3, alternative = alt), binom.test(7, 20, p = 0.3, alternative = alt))
}
add("binom_test", list(x = 0, n = 15), binom.test(0, 15))
add("binom_test", list(x = 15, n = 15, conf_level = 0.99), binom.test(15, 15, conf.level = 0.99))
add("binom_test", list(x = 9, n = 18), binom.test(9, 18))
add("binom_test", list(x = 31, n = 40, p = 0.62), binom.test(31, 40, p = 0.62))
add("binom_test", list(x = c(12, 30), p = 0.2), binom.test(c(12, 30), p = 0.2))

# --- prop.test --------------------------------------------------------------
pt <- function(...) suppressWarnings(prop.test(...))
for (alt in c("two.sided", "less", "greater")) {
  add("prop_test", list(x = 17, n = 40, p = 0.3, alternative = alt), pt(17, 40, p = 0.3, alternative = alt))
  add("prop_test", list(x = c(15, 25), n = c(50, 50), alternative = alt), pt(c(15, 25), c(50, 50), alternative = alt))
}
add("prop_test", list(x = 17, n = 40, correct = FALSE), pt(17, 40, correct = FALSE))
add("prop_test", list(x = 20, n = 40, p = 0.49), pt(20, 40, p = 0.49))
add("prop_test", list(x = 0, n = 12), pt(0, 12))
add("prop_test", list(x = c(15, 25), n = c(50, 50), correct = FALSE), pt(c(15, 25), c(50, 50), correct = FALSE))
add("prop_test", list(x = c(15, 16), n = c(50, 52)), pt(c(15, 16), c(50, 52)))
add("prop_test", list(x = c(83, 90, 129, 70), n = c(86, 93, 136, 82)), pt(c(83, 90, 129, 70), c(86, 93, 136, 82)))
add("prop_test", list(x = c(15, 25), n = c(50, 50), p = c(0.3, 0.4)), pt(c(15, 25), c(50, 50), p = c(0.3, 0.4)))

# --- p.adjust ---------------------------------------------------------------
pv <- c(0.01, 0.02, 0.02, 0.04, NA, 0.2, 0.001, 0.5, 0.03, 0.8)
padj <- list()
for (m in c("holm", "hochberg", "hommel", "bonferroni", "BH", "BY", "fdr", "none")) {
  padj[[length(padj) + 1]] <- list(method = m, n = NULL, value = p.adjust(pv, m))
  padj[[length(padj) + 1]] <- list(method = m, n = 15, value = p.adjust(pv, m, n = 15))
}
padj[[length(padj) + 1]] <- list(method = "hommel", n = NULL, value = p.adjust(c(0.03, 0.04), "hommel"))
padj[[length(padj) + 1]] <- list(method = "hommel", n = NULL, value = p.adjust(c(0.04, 0.01, 0.04), "hommel"))

# --- proportion intervals ---------------------------------------------------
props <- list()
for (k in c(0, 3, 17, 40)) {
  props[[length(props) + 1]] <- list(
    x = k, n = 40, alpha = 0.05,
    wilson = as.numeric(prop.test(k, 40, correct = FALSE)$conf.int),
    clopper_pearson = as.numeric(binom.test(k, 40)$conf.int)
  )
}
props[[length(props) + 1]] <- list(
  x = 6, n = 25, alpha = 0.1,
  wilson = as.numeric(prop.test(6, 25, correct = FALSE, conf.level = 0.9)$conf.int),
  clopper_pearson = as.numeric(binom.test(6, 25, conf.level = 0.9)$conf.int)
)

res <- list(
  r_version = R.version.string,
  cases = cases,
  p_adjust = list(p = pv, results = padj),
  proportion_ci = props
)
write(toJSON(res, auto_unbox = TRUE, digits = NA, na = "string", null = "null", pretty = FALSE), out)
cat("wrote", out, "\n")
