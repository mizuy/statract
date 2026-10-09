# Time the R estimators on the prepared slices and write python-comparable JSON.

Sys.setenv(
  OPENBLAS_NUM_THREADS = "1",
  OMP_NUM_THREADS = "1",
  MKL_NUM_THREADS = "1",
  NUMEXPR_NUM_THREADS = "1"
)

suppressPackageStartupMessages({
  library(jsonlite)
  library(sandwich)
  library(lmtest)
  library(survival)
  library(MatchIt)
  library(mgcv)
  library(lme4)
})

cache <- "/tmp/statract-bench"
manifest <- fromJSON(file.path(cache, "prepared", "manifest.json"), simplifyVector = FALSE)
repeats <- 5

ols_x <- c(
  "age", "education_num", "capital_gain", "capital_loss", "sex", "race",
  "marital3", "workclass4", "relationship3", "us_native"
)
glm_x <- c(
  "age", "education_num", "capital_gain", "capital_loss", "sex", "race",
  "marital3", "workclass4", "us_native", "hours_per_week"
)
match_x <- c(
  "age", "education_num", "capital_gain", "capital_loss", "race", "marital3",
  "workclass4", "relationship3", "us_native", "hours_per_week"
)
surv_x <- c("age", "sex", "num_co", "scoma", "meanbp", "hrt", "temp", "resp", "diabetes", "ca")
# The tasks added for the later functions call pROC, rms, stdReg2, cmprsk,
# gamm4, glmmTMB, partykit and mice through `::` without attaching them, so a
# missing package fails only its own task. pROC would otherwise mask var().
roc_x2 <- c("age", "sex", "num_co", "ca")
# The logistic validate/calibrate tasks leave out temp, as run_python.py explains.
lrm_x <- setdiff(surv_x, "temp")
htest_x <- c("age", "num_co", "scoma", "meanbp", "hrt", "temp", "resp")
std_cox_x <- c("diabetes", "age", "sex", "num_co", "meanbp", "ca")
gamm_grid <- c(50, 70, 90, 110, 130)
glmm_formula <- list(
  poisson = "num_co ~ age + sex + meanbp + ca + (1 | dzgroup)",
  negative_binomial = "slos ~ age + sex + num_co + meanbp + ca + (1 | dzgroup)"
)
mi_columns <- c("hospdead", "age", "sex", "num_co", "meanbp", "hrt", "alb", "bili", "pafi", "wblc", "income4")
mi_x <- c("age", "sex", "num_co", "meanbp", "hrt", "alb", "bili", "pafi", "wblc", "income4")
star_x <- c(
  "gender", "race", "freelunch", "birthyear", "grade", "classtype", "urban",
  "tyears", "tgen", "classsize"
)

time_call <- function(fun, n = repeats) {
  fun()
  samples <- numeric(n)
  result <- NULL
  for (i in seq_len(n)) {
    start <- proc.time()[["elapsed"]]
    result <- fun()
    samples[[i]] <- proc.time()[["elapsed"]] - start
  }
  list(seconds = median(samples), result = result)
}

rhs <- function(names) paste(unlist(names), collapse = " + ")

apply_factors <- function(frame, factors) {
  for (name in factors) {
    if (!name %in% names(frame)) next
    values <- as.character(frame[[name]])
    frame[[name]] <- factor(values, levels = sort(unique(values)))
  }
  frame
}

load_slice <- function(info) {
  # Factor columns stay character. read.csv would otherwise turn zero-padded
  # codes such as "01" into integers and change the dummy names.
  factors <- unlist(info$factors)
  header <- names(read.csv(info$csv, nrows = 0, check.names = FALSE))
  classes <- ifelse(header %in% factors, "character", NA)
  # Empty fields are missing (support-mi keeps its missing laboratory values).
  frame <- read.csv(
    info$csv,
    stringsAsFactors = FALSE,
    check.names = FALSE,
    colClasses = classes,
    na.strings = c("", "NA")
  )
  apply_factors(frame, factors)
}

fit_terms <- function(fit) {
  coefs <- coef(fit)
  se <- sqrt(diag(vcov(fit)))
  list(
    terms = names(coefs),
    coef = as.numeric(coefs),
    se = as.numeric(se),
    loglik = as.numeric(logLik(fit))
  )
}

ols_terms <- function(fit) {
  tab <- summary(fit)$coefficients
  list(
    terms = rownames(tab),
    coef = as.numeric(tab[, "Estimate"]),
    se = as.numeric(tab[, "Std. Error"]),
    t = as.numeric(tab[, "t value"]),
    p = as.numeric(tab[, "Pr(>|t|)"]),
    loglik = as.numeric(logLik(fit))
  )
}

cov_payload <- function(fit, cov) {
  list(terms = names(coef(fit)), cov = unname(as.matrix(cov)))
}

curve_rows <- function(fit, times, grouped) {
  summary_fit <- summary(fit, times = times, extend = TRUE)
  groups <- if (grouped) sub("^[^=]+=", "", summary_fit$strata) else rep(NA_character_, length(summary_fit$time))
  lapply(seq_along(summary_fit$time), function(i) {
    list(
      group = if (is.na(groups[[i]])) NULL else groups[[i]],
      time = summary_fit$time[[i]],
      estimate = summary_fit$surv[[i]],
      low = summary_fit$lower[[i]],
      high = summary_fit$upper[[i]]
    )
  })
}

pair_payload <- function(matched) {
  matrix <- matched$match.matrix
  pairs <- list()
  if (!is.null(matrix) && length(matrix)) {
    treated <- as.integer(rownames(matrix))
    controls <- as.integer(matrix[, 1])
    keep <- !is.na(controls)
    if (any(keep)) {
      pairs <- lapply(which(keep), function(i) c(treated[[i]], controls[[i]]))
    }
  }
  list(weights = as.numeric(matched$weights), pairs = pairs)
}

htest_payload <- function(h) {
  out <- list(stat = unname(h$statistic), p = h$p.value)
  if (!is.null(h$parameter)) out$df <- unname(h$parameter)
  if (!is.null(h$conf.int)) out$ci <- as.numeric(h$conf.int)
  out$estimate <- I(unname(as.numeric(h$estimate)))
  out
}

# Splits of a ctree depth first, written as run_python.py writes them.
ctree_splits <- function(tree) {
  data <- tree$data
  out <- character(0)
  walk <- function(node) {
    split <- partykit::split_node(node)
    if (is.null(split)) return(invisible(NULL))
    var <- names(data)[partykit::varid_split(split)]
    breaks <- partykit::breaks_split(split)
    if (!is.null(breaks)) {
      out[[length(out) + 1L]] <<- sprintf("%s<=%.10g", var, breaks)
    } else {
      index <- partykit::index_split(split)
      left <- levels(data[[var]])[!is.na(index) & index == 1L]
      out[[length(out) + 1L]] <<- sprintf("%s in {%s}", var, paste(sort(left, method = "radix"), collapse = ","))
    }
    for (kid in partykit::kids_node(node)) walk(kid)
  }
  walk(partykit::node_party(tree))
  out
}

run_task <- function(task, frame, manifest) {
  kind <- task$kind
  option <- task$option
  record <- list(id = task$id, slice = task$slice, n = nrow(frame), error = NULL, parts = list())
  tryCatch({
    if (kind == "ols") {
      timed <- time_call(function() lm(as.formula(paste("hours_per_week ~", rhs(ols_x))), data = frame))
      record$parts$main <- c(ols_terms(timed$result), list(seconds = timed$seconds))
    } else if (kind == "ols_w") {
      timed <- time_call(function() lm(as.formula(paste("hours_per_week ~", rhs(ols_x))), data = frame, weights = fnlwgt))
      record$parts$main <- c(ols_terms(timed$result), list(seconds = timed$seconds))
    } else if (kind == "glm_bin") {
      timed <- time_call(function() glm(as.formula(paste("income_gt_50k ~", rhs(glm_x))), data = frame, family = binomial()))
      record$parts$main <- c(fit_terms(timed$result), list(seconds = timed$seconds))
    } else if (kind == "glm_gamma") {
      timed <- time_call(function() glm(
        as.formula(paste("hours_per_week ~", rhs(ols_x))),
        data = frame,
        family = Gamma(link = "inverse")
      ))
      record$parts$main <- c(fit_terms(timed$result), list(seconds = timed$seconds))
    } else if (kind == "glm_pois") {
      predictors <- manifest$slices[[task$slice]]$predictors
      timed <- time_call(function() glm(as.formula(paste("cnt ~", rhs(predictors))), data = frame, family = poisson()))
      record$parts$main <- c(fit_terms(timed$result), list(seconds = timed$seconds))
    } else if (kind == "hc") {
      fit <- lm(as.formula(paste("hours_per_week ~", rhs(ols_x))), data = frame)
      for (hc_type in option$types) {
        timed <- time_call(function() vcovHC(fit, type = hc_type))
        record$parts[[hc_type]] <- c(cov_payload(fit, timed$result), list(seconds = timed$seconds))
      }
    } else if (kind == "cl1") {
      fit <- lm(as.formula(paste("read ~", rhs(star_x))), data = frame)
      timed <- time_call(function() vcovCL(fit, cluster = frame$school, type = "HC1"))
      record$parts$main <- c(cov_payload(fit, timed$result), list(seconds = timed$seconds))
    } else if (kind == "cl2") {
      fit <- lm(as.formula(paste("read ~", rhs(star_x))), data = frame)
      cluster <- cbind(frame$school, frame$grade)
      timed <- time_call(function() vcovCL(fit, cluster = cluster, type = "HC1"))
      record$parts$main <- c(cov_payload(fit, timed$result), list(seconds = timed$seconds))
    } else if (kind == "nw") {
      predictors <- manifest$slices[[task$slice]]$predictors
      fit <- lm(as.formula(paste("cnt ~", rhs(predictors))), data = frame)
      lag <- floor(4 * (nrow(frame) / 100) ^ (2 / 9))
      timed <- time_call(function() NeweyWest(fit, lag = lag, prewhite = FALSE, adjust = FALSE))
      record$parts$main <- c(cov_payload(fit, timed$result), list(seconds = timed$seconds))
    } else if (kind %in% c("wald", "lr")) {
      full <- lm(as.formula(paste("hours_per_week ~", rhs(ols_x))), data = frame)
      reduced_x <- ols_x[ols_x != "us_native"]
      reduced <- lm(as.formula(paste("hours_per_week ~", rhs(reduced_x))), data = frame)
      if (kind == "wald") {
        timed <- time_call(function() waldtest(full, reduced))
        record$parts$main <- list(stat = timed$result[2, "F"], p = timed$result[2, "Pr(>F)"], seconds = timed$seconds)
      } else {
        timed <- time_call(function() lrtest(full, reduced))
        record$parts$main <- list(
          stat = timed$result[2, "Chisq"],
          p = timed$result[2, "Pr(>Chisq)"],
          seconds = timed$seconds
        )
      }
    } else if (kind == "bp") {
      fit <- lm(as.formula(paste("hours_per_week ~", rhs(ols_x))), data = frame)
      for (studentize in option$studentize) {
        label <- if (isTRUE(studentize)) "student" else "raw"
        timed <- time_call(function() bptest(fit, studentize = studentize))
        record$parts[[label]] <- list(stat = unname(timed$result$statistic), p = unname(timed$result$p.value), seconds = timed$seconds)
      }
    } else if (kind == "reset") {
      fit <- lm(as.formula(paste("hours_per_week ~", rhs(ols_x))), data = frame)
      timed <- time_call(function() resettest(fit, power = 2:3, type = "fitted"))
      record$parts$main <- list(stat = unname(timed$result$statistic), p = unname(timed$result$p.value), seconds = timed$seconds)
    } else if (kind == "dw") {
      predictors <- manifest$slices[[task$slice]]$predictors
      fit <- lm(as.formula(paste("cnt ~", rhs(predictors))), data = frame)
      timed <- time_call(function() dwtest(fit, alternative = "greater"))
      record$parts$main <- list(stat = unname(timed$result$statistic), p = unname(timed$result$p.value), seconds = timed$seconds)
    } else if (kind == "bg") {
      predictors <- manifest$slices[[task$slice]]$predictors
      fit <- lm(as.formula(paste("cnt ~", rhs(predictors))), data = frame)
      for (order in option$orders) {
        timed <- time_call(function() bgtest(fit, order = order, type = "Chisq"))
        record$parts[[as.character(order)]] <- list(
          stat = unname(timed$result$statistic),
          p = unname(timed$result$p.value),
          seconds = timed$seconds
        )
      }
    } else if (kind %in% c("km", "km_sex", "na")) {
      times <- unlist(manifest$km_times)
      grouped <- kind == "km_sex"
      formula <- if (grouped) Surv(d_time, death) ~ sex else Surv(d_time, death) ~ 1
      timed <- time_call(function() {
        if (kind == "na") survfit(formula, data = frame, stype = 2) else survfit(formula, data = frame)
      })
      record$parts$main <- list(rows = curve_rows(timed$result, times, grouped), seconds = timed$seconds)
    } else if (kind == "lrk") {
      specs <- list(
        rho0 = list(formula = Surv(d_time, death) ~ sex, rho = 0),
        rho1 = list(formula = Surv(d_time, death) ~ sex, rho = 1),
        strata = list(formula = Surv(d_time, death) ~ sex + strata(ca), rho = 0)
      )
      for (label in names(specs)) {
        spec <- specs[[label]]
        timed <- time_call(function() survdiff(spec$formula, data = frame, rho = spec$rho))
        stat <- timed$result$chisq
        df <- length(timed$result$n) - 1
        record$parts[[label]] <- list(stat = stat, p = pchisq(stat, df, lower.tail = FALSE), seconds = timed$seconds)
      }
    } else if (kind == "cox") {
      formula_text <- paste("Surv(d_time, death) ~", rhs(surv_x))
      if (isTRUE(option$strata)) formula_text <- paste(formula_text, "+ strata(dzgroup)")
      ties <- option$ties
      timed <- time_call(function() coxph(as.formula(formula_text), data = frame, ties = ties))
      record$parts$main <- c(fit_terms(timed$result), list(seconds = timed$seconds))
    } else if (kind == "aft") {
      dist <- option$distribution
      formula <- as.formula(paste("Surv(d_time, death) ~", rhs(surv_x)))
      timed <- time_call(function() survreg(formula, data = frame, dist = dist))
      fit <- timed$result
      payload <- fit_terms(fit)
      # coef() omits the scale. Weibull and lognormal report it as Log(scale),
      # and vcov() already carries that extra row.
      if (dist != "exponential") {
        coefs <- coef(fit)
        vc <- vcov(fit)
        if (nrow(vc) != length(coefs) + 1L) {
          stop(paste("survreg vcov rows", nrow(vc), "coef", length(coefs)))
        }
        payload$terms <- c(names(coefs), "Log(scale)")
        payload$coef <- c(as.numeric(coefs), unname(log(fit$scale)))
        payload$se <- as.numeric(sqrt(diag(vc)))
      }
      record$parts$main <- c(payload, list(seconds = timed$seconds))
    } else if (kind == "match") {
      work <- frame
      work$treat <- as.integer(work$sex == "Female")
      rownames(work) <- as.character(seq_len(nrow(work)) - 1L)
      distance <- option$distance
      timed <- time_call(function() {
        if (distance == "exact") {
          matchit(treat ~ race, data = work, method = "exact", estimand = "ATT")
        } else if (distance == "cem") {
          matchit(as.formula(paste("treat ~", rhs(match_x))), data = work, method = "cem", estimand = "ATT")
        } else if (distance == "logit") {
          matchit(
            as.formula(paste("treat ~", rhs(match_x))),
            data = work, method = "nearest", distance = "glm", link = "logit",
            m.order = "data", estimand = "ATT", replace = FALSE
          )
        } else {
          matchit(
            as.formula(paste("treat ~", rhs(match_x))),
            data = work, method = "nearest", distance = "mahalanobis",
            m.order = "data", estimand = "ATT", replace = FALSE
          )
        }
      })
      payload <- pair_payload(timed$result)
      if (distance == "logit") {
        balance <- summary(timed$result)$sum.matched
        payload$smd_age <- unname(balance["age", "Std. Mean Diff."])
      }
      payload$seconds <- timed$seconds
      record$parts$main <- payload
    } else if (kind == "gam") {
      k <- option$k
      timed <- time_call(function() gam(cnt ~ s(temp, bs = "cr", k = k), data = frame, method = "REML"))
      # The default outer Newton stops with a gradient near 1e-3 in log sp,
      # which moves the coefficients by up to 1e-3 on 10,000 rows. The time is
      # the default call; the numbers come from a refit run to convergence.
      converged <- gam(
        cnt ~ s(temp, bs = "cr", k = k), data = frame, method = "REML",
        control = gam.control(newton = list(conv.tol = 1e-12), epsilon = 1e-12)
      )
      record$parts$main <- list(
        coef = as.numeric(coef(converged)),
        sp = unname(converged$sp[[1]]),
        # sum(edf) is the trace of the hat matrix, including the intercept.
        # summary()$s.table is the smooth alone and is short by 1.
        edf = sum(converged$edf),
        reml = unname(converged$gcv.ubre),
        seconds = timed$seconds
      )
    } else if (kind == "lmm") {
      if (isTRUE(option$slopes)) {
        formula <- as.formula(paste("read ~", rhs(star_x), "+ (1 + grade | student)"))
      } else {
        formula <- as.formula(paste("read ~", rhs(star_x), "+ (1 | class_id)"))
      }
      reml <- option$method == "reml"
      timed <- time_call(function() lmer(formula, data = frame, REML = reml))
      fit <- timed$result
      re <- as.matrix(VarCorr(fit)[[1]])
      record$parts$main <- list(
        terms = names(fixef(fit)),
        coef = as.numeric(fixef(fit)),
        se = as.numeric(sqrt(diag(vcov(fit)))),
        re = unname(re),
        sigma2 = unname(sigma(fit)^2),
        loglik = as.numeric(logLik(fit)),
        seconds = timed$seconds
      )
    } else if (kind == "ttest") {
      x <- frame$meanbp[frame$hospdead == 1]
      y <- frame$meanbp[frame$hospdead != 1]
      for (label in c("welch", "pooled")) {
        equal <- label == "pooled"
        timed <- time_call(function() t.test(x, y, var.equal = equal))
        record$parts[[label]] <- c(htest_payload(timed$result), list(seconds = timed$seconds))
      }
    } else if (kind == "wilcox") {
      x <- frame$meanbp[frame$hospdead == 1]
      y <- frame$meanbp[frame$hospdead != 1]
      timed <- time_call(function() suppressWarnings(wilcox.test(x, y, conf.int = TRUE)))
      record$parts$main <- c(htest_payload(timed$result), list(seconds = timed$seconds))
    } else if (kind == "prop") {
      died <- frame$hospdead == 1
      female <- frame$sex == "female"
      counts <- c(sum(died & female), sum(died & !female))
      totals <- c(sum(female), sum(!female))
      timed <- time_call(function() prop.test(counts, totals))
      record$parts$main <- c(htest_payload(timed$result), list(seconds = timed$seconds))
    } else if (kind == "padj") {
      raw <- vapply(htest_x, function(name) {
        t.test(frame[[name]][frame$hospdead == 1], frame[[name]][frame$hospdead != 1])$p.value
      }, numeric(1))
      for (method in unlist(option$methods)) {
        timed <- time_call(function() p.adjust(raw, method))
        record$parts[[method]] <- list(adjusted = unname(timed$result), seconds = timed$seconds)
      }
    } else if (kind == "roc") {
      score1 <- predict(glm(as.formula(paste("hospdead ~", rhs(surv_x))), data = frame, family = binomial()), type = "link")
      score2 <- predict(glm(as.formula(paste("hospdead ~", rhs(roc_x2))), data = frame, family = binomial()), type = "link")
      timed <- time_call(function() {
        r1 <- pROC::roc(frame$hospdead, unname(score1), quiet = TRUE)
        r2 <- pROC::roc(frame$hospdead, unname(score2), quiet = TRUE)
        list(
          r1 = r1, r2 = r2,
          ci1 = pROC::ci.auc(r1, method = "delong"),
          ci2 = pROC::ci.auc(r2, method = "delong"),
          test = pROC::roc.test(r1, r2, method = "delong")
        )
      })
      res <- timed$result
      record$parts$main <- list(
        auc = c(as.numeric(res$r1$auc), as.numeric(res$r2$auc)),
        var = c(as.numeric(pROC::var(res$r1, method = "delong")), as.numeric(pROC::var(res$r2, method = "delong"))),
        ci = c(as.numeric(res$ci1)[c(1, 3)], as.numeric(res$ci2)[c(1, 3)]),
        stat = unname(as.numeric(res$test$statistic)),
        p = res$test$p.value,
        seconds = timed$seconds
      )
    } else if (kind %in% c("val_lrm", "val_cph")) {
      b <- option$B
      if (kind == "val_lrm") {
        fit <- rms::lrm(as.formula(paste("hospdead ~", rhs(lrm_x))), data = frame, x = TRUE, y = TRUE)
      } else {
        fit <- rms::cph(as.formula(paste("Surv(d_time, death) ~", rhs(surv_x))), data = frame, x = TRUE, y = TRUE)
      }
      # The resamples differ from numpy's, so only index.orig is compared. The
      # time covers the same B repetitions on both sides.
      timed <- time_call(function() {
        set.seed(manifest$seed)
        rms::validate(fit, B = b)
      })
      tab <- unclass(timed$result)
      record$parts$main <- list(
        terms = rownames(tab),
        index_orig = unname(as.numeric(tab[, "index.orig"])),
        index_corrected = unname(as.numeric(tab[, "index.corrected"])),
        seconds = timed$seconds
      )
    } else if (kind == "cal_lrm") {
      b <- option$B
      fit <- rms::lrm(as.formula(paste("hospdead ~", rhs(lrm_x))), data = frame, x = TRUE, y = TRUE)
      timed <- time_call(function() {
        set.seed(manifest$seed)
        rms::calibrate(fit, B = b)
      })
      tab <- unclass(timed$result)
      record$parts$main <- list(
        predy = unname(as.numeric(tab[, "predy"])),
        calibrated_orig = unname(as.numeric(tab[, "calibrated.orig"])),
        seconds = timed$seconds
      )
    } else if (kind == "cal_cph") {
      b <- option$B
      u <- option$u
      m <- option$m
      fit <- rms::cph(
        as.formula(paste("Surv(d_time, death) ~", rhs(surv_x))),
        data = frame, x = TRUE, y = TRUE, surv = TRUE, time.inc = u
      )
      timed <- time_call(function() {
        set.seed(manifest$seed)
        rms::calibrate(fit, cmethod = "KM", u = u, m = m, B = b)
      })
      tab <- unclass(timed$result)
      record$parts$main <- list(
        mean_predicted = unname(as.numeric(tab[, "mean.predicted"])),
        KM = unname(as.numeric(tab[, "KM"])),
        std_err = unname(as.numeric(tab[, "std.err"])),
        index_orig = unname(as.numeric(tab[, "index.orig"])),
        seconds = timed$seconds
      )
    } else if (kind == "std_cox") {
      times <- unlist(manifest$km_times)
      values <- setNames(list(unlist(option$values)), option$exposure)
      formula <- as.formula(paste("Surv(d_time, death) ~", rhs(std_cox_x)))
      timed <- time_call(function() {
        stdReg2::standardize_coxph(formula = formula, data = frame, values = values, times = times, measure = "survival")
      })
      res <- timed$result$res
      est <- as.matrix(res$est)
      record$parts$main <- list(
        # Row-major (time, value), as numpy's ravel of the times x values matrix.
        estimate = as.numeric(t(unname(est))),
        cov = unlist(lapply(res$vcov, function(v) as.numeric(t(unname(as.matrix(v)))))),
        seconds = timed$seconds
      )
    } else if (kind == "cuminc") {
      times <- unlist(manifest$km_times)
      for (label in c("main", "strata")) {
        timed <- time_call(function() {
          if (label == "main") {
            cmprsk::cuminc(frame$d_time, frame$cause, frame$sex)
          } else {
            cmprsk::cuminc(frame$d_time, frame$cause, frame$sex, strata = frame$ca)
          }
        })
        fit <- timed$result
        tests <- fit$Tests[order(rownames(fit$Tests), method = "radix"), , drop = FALSE]
        part <- list(stat = unname(tests[, "stat"]), p = unname(tests[, "pv"]), seconds = timed$seconds)
        if (label == "main") {
          tp <- cmprsk::timepoints(fit, times)
          keep <- order(rownames(tp$est), method = "radix")
          # Row by row ("female 1", "female 2", "male 1", ...), each across the times.
          part$estimate <- as.numeric(t(unname(tp$est[keep, , drop = FALSE])))
          part$variance <- as.numeric(t(unname(tp$var[keep, , drop = FALSE])))
        }
        record$parts[[label]] <- part
      }
    } else if (kind == "crr") {
      mm <- model.matrix(as.formula(paste("~", rhs(surv_x))), frame)[, -1, drop = FALSE]
      timed <- time_call(function() cmprsk::crr(frame$d_time, frame$cause, mm, failcode = option$cause, cencode = 0))
      fit <- timed$result
      record$parts$main <- list(
        terms = colnames(mm),
        coef = unname(fit$coef),
        se = unname(sqrt(diag(fit$var))),
        loglik = fit$loglik,
        seconds = timed$seconds
      )
    } else if (kind == "gamm") {
      k <- option$k
      formula <- as.formula(sprintf("hospdead ~ age + s(meanbp, bs = \"cr\", k = %d)", as.integer(k)))
      timed <- time_call(function() gamm4::gamm4(formula, random = ~ (1 | dzgroup), family = binomial(), data = frame))
      # The time is the default call. The numbers come from a refit with glmer
      # run to its optimum, as in tests/r_oracle/scripts/gamm4.R. edf and SEs
      # are not compared: gamm4 0.2-6 builds them from chol(V, pivot = TRUE),
      # whose pivot Matrix >= 1.6 no longer reports, so they are about 1% off.
      tight <- glmerControl(
        optimizer = "nloptwrap", tolPwrss = 1e-13,
        optCtrl = list(xtol_abs = 1e-12, ftol_abs = 1e-14, xtol_rel = 0, ftol_rel = 0, maxeval = 1e5)
      )
      fit <- gamm4::gamm4(formula, random = ~ (1 | dzgroup), family = binomial(), data = frame, control = tight)
      s <- summary(fit$gam)
      nd <- data.frame(meanbp = gamm_grid, age = median(frame$age))
      terms <- predict(fit$gam, newdata = nd, type = "terms")
      record$parts$main <- list(
        terms = rownames(s$p.table),
        coef = unname(s$p.table[, 1]),
        re = as.numeric(VarCorr(fit$mer)$dzgroup),
        loglik = as.numeric(logLik(fit$mer)),
        smooth = unname(terms[, rownames(s$s.table)[1]]),
        edf = unname(s$edf),
        seconds = timed$seconds
      )
    } else if (kind == "glmm") {
      family <- if (option$family == "poisson") poisson() else glmmTMB::nbinom2()
      formula <- as.formula(glmm_formula[[option$family]])
      timed <- time_call(function() glmmTMB::glmmTMB(formula, data = frame, family = family))
      fit <- timed$result
      payload <- list(
        terms = names(fixef(fit)$cond),
        coef = unname(fixef(fit)$cond),
        se = unname(sqrt(diag(vcov(fit)$cond))),
        loglik = as.numeric(logLik(fit)),
        re = unname(as.numeric(VarCorr(fit)$cond$dzgroup)[1]),
        seconds = timed$seconds
      )
      if (option$family != "poisson") payload$theta <- unname(sigma(fit))
      record$parts$main <- payload
    } else if (kind == "ctree") {
      formula <- as.formula(paste("hospdead ~", rhs(surv_x)))
      control <- partykit::ctree_control(
        teststat = "quadratic", testtype = "Bonferroni", alpha = 0.05, minsplit = 20L, minbucket = 7L
      )
      timed <- time_call(function() partykit::ctree(formula, data = frame, control = control))
      tree <- timed$result
      tests <- partykit:::sctest.constparty(tree, node = 1L)
      record$parts$main <- list(
        splits = I(ctree_splits(tree)),
        n_terminal = length(partykit::nodeids(tree, terminal = TRUE)),
        stat = unname(as.numeric(tests["statistic", ])),
        p = unname(as.numeric(tests["p.value", ])),
        pred = unname(as.numeric(predict(tree, newdata = frame))),
        seconds = timed$seconds
      )
    } else if (kind == "mice") {
      work <- frame[, mi_columns]
      m <- option$m
      maxit <- option$maxit
      mi_formula <- as.formula(paste("hospdead ~", rhs(mi_x)))
      n_time <- if (is.null(option$repeats)) repeats else option$repeats
      timed <- time_call(function() {
        imp <- mice::mice(work, m = m, maxit = maxit, seed = manifest$seed, printFlag = FALSE)
        fits <- lapply(seq_len(m), function(i) glm(mi_formula, data = mice::complete(imp, i), family = binomial()))
        list(imp = imp, pooled = mice::pool(fits))
      }, n = n_time)
      imp <- timed$result$imp
      used <- imp$method[imp$method != ""]
      nmis <- imp$nmis[names(used)]
      record$parts$main <- list(
        # Only which columns are imputed, by which method, and how many cells
        # are compared. mice's draws come from R's generator.
        methods = I(sort(paste0(names(used), ":", unname(used)), method = "radix")),
        nmis = I(unname(as.integer(nmis[order(names(nmis), method = "radix")]))),
        pooled_terms = I(as.character(timed$result$pooled$pooled$term)),
        pooled_estimate = timed$result$pooled$pooled$estimate,
        seconds = timed$seconds
      )
    } else {
      stop(paste("unknown kind", kind))
    }
    record
  }, error = function(err) {
    record$error <- conditionMessage(err)
    record
  })
}

slices <- list()
for (name in names(manifest$slices)) {
  message("load ", name)
  slices[[name]] <- load_slice(manifest$slices[[name]])
}

out <- list()
for (task in manifest$tasks) {
  message("R ", task$id, " ", task$slice)
  out[[length(out) + 1]] <- run_task(task, slices[[task$slice]], manifest)
}

dir.create(file.path(cache, "results"), showWarnings = FALSE, recursive = TRUE)
write_json(out, file.path(cache, "results", "r.json"), auto_unbox = TRUE, digits = 16, null = "null")
failed <- Filter(function(row) !is.null(row$error), out)
message("R failed ", length(failed))
for (row in failed) message("  ", row$id, " ", row$slice, " ", row$error)
