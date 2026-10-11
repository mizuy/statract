# IPTW oracle with WeightIt and cobalt themselves, for a machine with CRAN.
# Run from the repository root after tests/r_oracle/scripts/iptw.R has written
# the data CSVs:
#   Rscript tests/r_oracle/scripts/iptw_weightit.R
#
# It writes tests/r_oracle/fixtures/iptw.json in the same layout as iptw.R,
# which is the base R version of the same formulas for machines without the
# packages. The committed fixture comes from this script (WeightIt 2.1.0,
# cobalt 5.0.0), and iptw.R reproduces it exactly.
suppressPackageStartupMessages({
  library(WeightIt)
  library(cobalt)
  library(jsonlite)
})

data_dir <- "tests/r_oracle/data"
out_path <- "tests/r_oracle/fixtures/iptw.json"

read_sample <- function(name) read.csv(file.path(data_dir, paste0(name, ".csv")))

cases <- list()
add_case <- function(id, sample, covs, estimand, stabilize = FALSE, trim = NULL,
                     trim_lower = FALSE, s_weights = NULL, binary = "raw") {
  d0 <- read_sample(sample)
  used <- c("treat", covs, s_weights)
  # Complete cases first: weightit() would add missingness indicators.
  rows <- which(complete.cases(d0[, used]))
  d <- d0[rows, ]
  f <- reformulate(covs, response = "treat")
  W <- weightit(f, data = d, method = "glm", estimand = estimand,
                stabilize = stabilize, s.weights = s_weights)
  if (!is.null(trim)) W <- trim(W, at = trim, lower = trim_lower)
  bt <- bal.tab(W, un = TRUE, binary = binary, continuous = "std",
                disp.means = TRUE, disp.v.ratio = TRUE)
  B <- bt$Balance
  num <- function(col) if (col %in% names(B)) unname(B[[col]]) else rep(NA_real_, nrow(B))
  obs <- bt$Observations
  ctrl <- if ("Control" %in% names(obs)) "Control" else names(obs)[1]
  trt <- if ("Treated" %in% names(obs)) "Treated" else names(obs)[2]
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
    ps = unname(W$ps),
    weights = unname(W$weights),
    balance = list(
      term = rownames(B),
      type = as.character(B$Type),
      mean_treated_unadjusted = num("M.1.Un"),
      mean_control_unadjusted = num("M.0.Un"),
      diff_unadjusted = num("Diff.Un"),
      variance_ratio_unadjusted = num("V.Ratio.Un"),
      mean_treated_adjusted = num("M.1.Adj"),
      mean_control_adjusted = num("M.0.Adj"),
      diff_adjusted = num("Diff.Adj"),
      variance_ratio_adjusted = num("V.Ratio.Adj")
    ),
    ess = list(
      control = unname(obs[c("Unadjusted", "Adjusted"), ctrl]),
      treated = unname(obs[c("Unadjusted", "Adjusted"), trt])
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
  source = paste0("WeightIt ", packageVersion("WeightIt"), " / cobalt ", packageVersion("cobalt")),
  cases = cases
), auto_unbox = TRUE, digits = NA, na = "string", null = "null"), out_path)
