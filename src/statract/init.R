library(tidyverse)
# library(magick)
library(ggplot2)
# library(MatchIt)
# library(mgcv)
# library(gratia)
# library(gamm4)
# library(ggforce)
# library(patchwork)
# library(gam)
library(polars)


format_pvalue <- function(pvalue) {
  if (is.na(pvalue)) {
    return("-")
  }
  if (pvalue >= 0.01) {
    return(sprintf(pvalue, fmt = "%.2f"))
  } else if (pvalue >= 0.001) {
    return(sprintf(pvalue, fmt = "%.3f"))
  } else {
    return("P<.001")
  }
}

get_glm_plot <- function(glm_table) {
  target <- glm_table %>% mutate(
    row = row_number(),
    hr = exp(coef),
    lo = exp(coef - 1.96 * se),
    hi = exp(coef + 1.96 * se)
  )
  ormax <- ceiling(max(max(target$hr), max(target$hi)))
  return(target %>%
    ggplot(aes(x = reorder(name, -row), y = hr)) +
    geom_point(shape = 15, size = 4, position = "dodge", color = "black") +
    geom_errorbar(aes(ymin = lo, ymax = hi), width = 0.2, size = 0.7, position = "dodge", color = "black") +
    ylab("Hazard ratio with 95% CI") +
    xlab("") +
    coord_flip(ylim = c(0, ormax)) +
    geom_hline(yintercept = 1, color = "red", size = 1) +
    theme(axis.title = element_text(size = 17)) +
    theme(axis.text = element_text(size = 14)))
}

# Note: forest.R is loaded separately via rpy2's init() function
# For direct R usage, source forest.R explicitly:
#   source("path/to/forest.R")
