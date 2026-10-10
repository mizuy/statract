# Forest Plot Generation Library
# This library provides functions for creating forest plots with customizable columns
# and odds ratio visualizations.
#
# Usage:
#   source("path/to/forest.R")
#   # or from Python via rpy2:
#   # from statract.r import init; init()
#
# Exported functions:
#   - forest_col(): Create a text column generator
#   - forest_or(): Create an odds ratio column generator
#   - forest_plot(): Create a complete forest plot
#   - add_group_headers(): Add group headers to data
#
# Dependencies:
#   - tidyverse
#   - ggfittext
#   - patchwork
#   - showtext

# Load required libraries
if (!requireNamespace("tidyverse", quietly = TRUE)) {
  stop("tidyverse package is required. Install it with: install.packages('tidyverse')")
}
if (!requireNamespace("ggfittext", quietly = TRUE)) {
  stop("ggfittext package is required. Install it with: install.packages('ggfittext')")
}
if (!requireNamespace("patchwork", quietly = TRUE)) {
  stop("patchwork package is required. Install it with: install.packages('patchwork')")
}
if (!requireNamespace("showtext", quietly = TRUE)) {
  stop("showtext package is required. Install it with: install.packages('showtext')")
}

library(tidyverse)
library(ggfittext)
library(patchwork)
library(showtext)
showtext_auto()

# DEBUG <- TRUE
DEBUG <- FALSE

# Validation helper functions for better error messages
validate_character <- function(x, param_name) {
  if (!is.character(x)) {
    stop(sprintf("'%s' must be a character vector, got: %s", param_name, class(x)[1]))
  }
  if (length(x) == 0) {
    stop(sprintf("'%s' cannot be empty", param_name))
  }
  return(TRUE)
}

validate_numeric_range <- function(x, param_name, min_val = -Inf, max_val = Inf) {
  if (!is.numeric(x)) {
    stop(sprintf("'%s' must be numeric, got: %s", param_name, class(x)[1]))
  }
  if (length(x) != 1) {
    stop(sprintf("'%s' must be a single value, got length: %d", param_name, length(x)))
  }
  if (x < min_val || x > max_val) {
    stop(sprintf("'%s' must be between %g and %g, got: %g", param_name, min_val, max_val, x))
  }
  return(TRUE)
}

validate_positive_numeric <- function(x, param_name) {
  validate_numeric_range(x, param_name, min_val = 0, max_val = Inf)
}

validate_fontsize <- function(x, param_name) {
  if (!is.null(x)) {
    validate_positive_numeric(x, param_name)
  }
  return(TRUE)
}

validate_data_frame <- function(x, param_name) {
  if (!is.data.frame(x)) {
    stop(sprintf("'%s' must be a data frame, got: %s", param_name, class(x)[1]))
  }
  if (nrow(x) == 0) {
    stop(sprintf("'%s' cannot be empty (0 rows)", param_name))
  }
  return(TRUE)
}

# Base theme for forest plots
# Creates a clean, minimal theme suitable for forest plot visualizations
#
# @param fontsize Numeric, base font size for the theme
# @return ggplot2 theme object
get_theme_base <- function(fontsize = 14) {
  theme_classic(base_size = fontsize, base_family = "Helvetica") +
    theme(
      panel.background = element_rect(fill = "transparent", colour = NA),
      panel.grid.major = element_blank(),
      panel.grid.minor = element_blank(),
      plot.background = element_rect(fill = "transparent", colour = NA),
      plot.title = element_text(size = fontsize, hjust = 0.5, margin = margin(10, 10, 10, 10)),
      axis.line.y = element_blank(),
      axis.ticks.y = element_blank(),
      axis.text.y = element_blank(),
      axis.title.x = element_blank(),
      axis.title.y = element_blank(),
      plot.margin = unit(c(10, 0, 10, 0), "pt"),
      axis.ticks.length = unit(10, "pt"),
      legend.title = element_blank(),
      legend.position = "none",
      legend.margin = margin(10, 10, 10, 10)
    )
}

# Theme for text columns in forest plots
# Extends base theme by hiding x-axis elements for text-only columns
#
# @param fontsize Numeric, base font size for the theme
# @return ggplot2 theme object
get_theme_col <- function(fontsize = 14) {
  get_theme_base(fontsize) +
    theme(
      axis.ticks.x = element_blank(),
      axis.line.x = element_blank(),
      axis.text.x = element_blank(),
    )
}

# Theme for odds ratio columns in forest plots
# Extends base theme by adding x-axis line for odds ratio visualization
#
# @param fontsize Numeric, base font size for the theme
# @return ggplot2 theme object
get_theme_or <- function(fontsize = 14) {
  get_theme_base(fontsize) +
    theme(
      axis.line.x = element_line(linewidth = 0.5),
    )
}

# Create a text column generator for forest plots
# Generates a column that displays text content with customizable alignment and formatting
#
# @param col_label Character, name of the column in data to display
# @param title Character, optional title for the column
# @param hjust Numeric, horizontal justification (0=left, 0.5=center, 1=right)
# @param fontsize Numeric, font size for this column (NULL uses plot default)
# @param width Numeric, relative width of this column
# @return List containing plot generator and column metadata
forest_col <- function(col_label, title = NULL, hjust = 0.5, fontsize = NULL, width = 1, fittext = TRUE) {
  # Input validation with detailed error messages
  validate_character(col_label, "col_label")
  validate_numeric_range(hjust, "hjust", min_val = 0, max_val = 1)
  validate_fontsize(fontsize, "fontsize")
  validate_positive_numeric(width, "width")

  # Convert hjust to ggfittext place parameter
  place <- switch(as.character(hjust),
    "0" = "left",
    "1" = "right",
    "centre" # default for 0.5 and other values
  )

  # Store the fontsize for this column
  stored_fontsize <- fontsize

  # Create plot generator function
  plot_gen <- function(data, plot_fontsize = 14) {
    # Font size priority: stored_fontsize > plot_fontsize > default 14
    final_fontsize <- if (!is.null(stored_fontsize)) stored_fontsize else plot_fontsize

    t <- if (fittext) {
      ggfittext::geom_fit_text(
        aes(label = .data[[col_label]]),
        size = final_fontsize, place = place, grow = FALSE,
        padding.x = grid::unit(0, "mm"),
        padding.y = grid::unit(0, "mm")
      )
    } else {
      geom_text(aes(label = .data[[col_label]]), size = final_fontsize / 3, hjust = hjust)
    }
    plot <- data |> ggplot(aes(y = fct_reorder(as_factor(loc_), -loc_), x = 1)) +
      t +
      ggtitle(title) +
      ylab(NULL) +
      xlab(NULL) +
      get_theme_col(final_fontsize)

    # Add debug borders if enabled
    if (DEBUG) {
      plot <- plot +
        geom_tile(fill = "transparent", colour = "black")
    }
    return(plot)
  }

  return(
    list(
      plot_gen = plot_gen,
      name = col_label,
      width = width,
      title = title,
      hjust = hjust,
      fontsize = stored_fontsize
    )
  )
}




# Calculate maximum value excluding outliers
# Uses IQR method to identify and exclude extreme outliers
#
# @param x Numeric vector
# @param iqr_factor Numeric, multiplier for IQR to define outliers
# @return Numeric, maximum value excluding outliers
max_without_outliers <- function(x, iqr_factor = 1.5) {
  if (all(is.na(x))) {
    return(NA)
  }
  highest <- max(x, na.rm = TRUE)
  q1 <- quantile(x, 0.25, na.rm = TRUE)
  q3 <- quantile(x, 0.75, na.rm = TRUE)
  iqr <- q3 - q1
  upper_bound <- min(highest, q3 + iqr_factor * iqr + 0.1)
  return(upper_bound)
}

# Calculate minimum value excluding outliers
# Uses IQR method to identify and exclude extreme outliers
#
# @param x Numeric vector
# @param iqr_factor Numeric, multiplier for IQR to define outliers
# @return Numeric, minimum value excluding outliers
min_without_outliers <- function(x, iqr_factor = 1.5) {
  if (all(is.na(x))) {
    return(NA)
  }
  lowest <- min(x, na.rm = TRUE)
  q1 <- quantile(x, 0.25, na.rm = TRUE)
  q3 <- quantile(x, 0.75, na.rm = TRUE)
  iqr <- q3 - q1
  lower_bound <- max(lowest, q1 - iqr_factor * iqr - 0.1)
  return(lower_bound)
}

# Create an odds ratio column generator for forest plots
# Generates a column that displays odds ratios with confidence intervals
#
# @param col_or Character, name of the odds ratio column in data
# @param col_lo Character, name of the lower confidence interval column
# @param col_hi Character, name of the upper confidence interval column
# @param title Character, optional title for the column
# @param width Numeric, relative width of this column
# @param range Numeric vector of length 2, optional x-axis range [min, max]
# @param col_color Character, optional column name for color mapping
# @param fontsize Numeric, font size for this column (NULL uses plot default)
# @return List containing plot generator and column metadata
forest_or <- function(col_or, col_lo, col_hi, title = NULL, width = 1, range = NA, col_color = NULL, fontsize = NULL) {
  # Validate input parameters with detailed error messages
  validate_character(col_or, "col_or")
  validate_character(col_lo, "col_lo")
  validate_character(col_hi, "col_hi")
  validate_positive_numeric(width, "width")
  if (!is.null(col_color)) {
    validate_character(col_color, "col_color")
  }
  validate_fontsize(fontsize, "fontsize")

  # Normalize range parameter with detailed validation
  if (length(range) == 1 && is.na(range)) {
    range <- c(NA, NA)
  } else if (is.numeric(range)) {
    if (length(range) != 2) {
      stop("'range' must be a numeric vector of length 2, got length: ", length(range))
    }
    if (any(is.na(range))) {
      range <- c(NA, NA)
    } else if (range[1] >= range[2]) {
      stop("'range[1]' must be less than 'range[2]', got: [", range[1], ", ", range[2], "]")
    }
    # range is valid, keep as is
  } else {
    stop("'range' must be NA or a numeric vector of length 2, got: ", class(range)[1], " of length ", length(range))
  }

  # Store the fontsize for this column
  stored_fontsize <- fontsize

  # Create plot generator function
  plot_gen <- function(data, plot_fontsize = 14) {
    # Font size priority: stored_fontsize > plot_fontsize > default 14
    final_fontsize <- if (!is.null(stored_fontsize)) stored_fontsize else plot_fontsize

    # Prepare data for plotting
    data <- data |> dplyr::mutate(
      y = fct_reorder(as_factor(loc_), -loc_),
      x = .data[[col_or]],
      hi = .data[[col_hi]],
      lo = .data[[col_lo]]
    )

    # Add color mapping if specified
    if (!is.null(col_color)) {
      data <- data |> dplyr::mutate(color = .data[[col_color]])
    }

    # Calculate plot limits
    or_lim_high <- max_without_outliers(data$x, iqr_factor = 2)
    data <- data |> dplyr::mutate(disable = or_lim_high < x)
    data_active <- data |> dplyr::filter(!disable)
    lim_lo <- dplyr::if_else(is.na(range[1]), min(data_active$lo), range[1])
    lim_hi <- dplyr::if_else(is.na(range[2]), max_without_outliers(data_active$hi, iqr_factor = 2), range[2])

    # Prepare plot data with truncated values for out-of-range points
    plot_data <- data |> dplyr::mutate(
      disable = x < lim_lo | lim_hi < x,
      x = dplyr::if_else(disable, NA, x),
      lo = dplyr::if_else(disable, NA, lo),
      hi = dplyr::if_else(disable, NA, pmin(hi, lim_hi)),
      lo_trunc = dplyr::if_else(disable, NA, dplyr::if_else(lo < lim_lo, lim_lo, NA)),
      hi_trunc = dplyr::if_else(disable, NA, dplyr::if_else(hi > lim_hi, lim_hi, NA)),
    )

    # Create base plot with or without color mapping
    if (!is.null(col_color)) {
      plot <- plot_data |> ggplot() +
        geom_pointrange(aes(y = y, x = x, xmin = lo, xmax = hi, color = color), size = 1, linewidth = 1)
    } else {
      plot <- plot_data |> ggplot() +
        geom_pointrange(aes(y = y, x = x, xmin = lo, xmax = hi), size = 1)
    }

    # Add final plot elements
    plot +
      coord_cartesian(xlim = c(lim_lo, lim_hi)) +
      geom_point(aes(y = y, x = hi_trunc), shape = 21, fill = "white", size = 3) +
      geom_point(aes(y = y, x = lo_trunc), shape = 21, fill = "white", size = 3) +
      geom_vline(xintercept = 1, linewidth = 0.5, linetype = "dashed") +
      ggtitle(title) + ylab(NULL) + xlab(NULL) + get_theme_or(final_fontsize)
  }

  list(
    plot_gen = plot_gen,
    name = col_or,
    col_or = col_or,
    col_hi = col_hi,
    col_lo = col_lo,
    col_color = col_color,
    width = width,
    title = title,
    fontsize = stored_fontsize
  )
}


# Create a complete forest plot
# Combines multiple column generators into a single forest plot visualization
#
# @param data Data frame containing the data to plot
# @param col_generators List of column generator objects
# @param title Character, optional main title for the plot
# @param caption Character, optional caption for the plot
# @param fontsize Numeric, base font size for the plot
# @param return_data Logical, whether to return data along with plot
# @return ggplot2 object or list containing plot and data
forest_plot <- function(
    data,
    col_generators = list(
      forest_col("name", title = "", hjust = 0, width = 2),
      forest_col("n", hjust = 1),
      forest_col("event", title = "event", hjust = 1, width = 0.5),
      forest_or("oddsratio", "lo", "hi", width = 4),
      forest_col("desc", title = "OR (95% CI)", width = 2),
      forest_col("pvalue", title = "P value")
    ),
    title = NULL,
    caption = NULL,
    fontsize = 14,
    return_data = FALSE) {
  # Input validation with detailed error messages
  validate_data_frame(data, "data")
  if (!is.list(col_generators)) {
    stop("'col_generators' must be a list, got: ", class(col_generators)[1])
  }
  if (length(col_generators) == 0) {
    stop("'col_generators' cannot be empty")
  }
  validate_positive_numeric(fontsize, "fontsize")
  if (!is.logical(return_data)) {
    stop("'return_data' must be logical (TRUE/FALSE), got: ", class(return_data)[1])
  }

  # Add row numbers for ordering
  data <- data |> dplyr::mutate(loc_ = dplyr::row_number())

  # Pre-allocate vectors for better performance
  n_cols <- length(col_generators)
  plots <- vector("list", n_cols)
  widths <- numeric(n_cols)

  # Generate plots for each column
  for (i in seq_along(col_generators)) {
    col <- col_generators[[i]]

    # Validate column generator structure
    if (!is.list(col)) {
      stop("Column generator ", i, " must be a list, got: ", class(col)[1])
    }
    if (is.null(col$plot_gen) || !is.function(col$plot_gen)) {
      stop("Column generator ", i, " must have a 'plot_gen' function")
    }
    if (is.null(col$width) || !is.numeric(col$width)) {
      stop("Column generator ", i, " must have a numeric 'width'")
    }

    widths[i] <- col$width
    plots[[i]] <- col$plot_gen(data, fontsize)
  }

  # Combine plots using patchwork
  plot <- patchwork::wrap_plots(plots, widths = widths, nrow = 1, axes = "collect") +
    patchwork::plot_annotation(
      title = title,
      caption = caption,
      theme = theme(
        plot.title = element_text(size = fontsize * 0.86, margin = margin(0, 0, 0, 0)),
        plot.caption = element_text(size = fontsize * 0.86, margin = margin(0, 0, 0, 0))
      )
    ) &
    coord_cartesian(ylim = c(1, max(data$loc_)))

  # Return plot with optional data
  if (return_data) {
    return(list(plot = plot, plots = plots, data = data))
  } else {
    return(plot)
  }
}


# Inserts displayname column before each group to create visual separation
#
# @param data Data frame with a grouping column
# @param col Character, name of the column to use for grouping and placeholder (default: "name")
# @param display_col Character, name of the column to use for displayname (default: "displayname")
# @param style Character, style of placeholder insertion ("separate" or "inline", default: "inline")
# @return Data frame with displayname column added

add_group_headers <- function(data, col_name = "name", col_groupheader = "displayname", style = "inline") {
  # Input validation
  validate_data_frame(data, "data")
  validate_character(col_name, "col")
  validate_character(style, "style")
  if (!col_name %in% names(data)) {
    stop(sprintf("'data' must contain a '%s' column for grouping", col_name))
  }
  if (!style %in% c("separate", "inline")) {
    stop("'style' must be either 'separate' or 'inline', got: ", style)
  }


  if (style == "separate") {
    # Original style: separate placeholder rows before each group
    data |>
      dplyr::group_by(.data[[col_name]]) |>
      dplyr::group_modify(function(df, groupname, ...) {
        df |>
          mutate(!!col_groupheader := NA_character_) |>
          select(-col_name) |>
          add_row(
            !!col_groupheader := groupname[[col_name]][1],
            .before = 0
          )
      }, .keep = TRUE) |>
      dplyr::ungroup()
  } else {
    # Inline style: add placeholder column and empty rows between groups
    # First add placeholder column with group name only in first row
    data |>
      dplyr::group_by(Key = fct_inorder(.data[[col_name]])) |>
      dplyr::mutate(
        !!col_groupheader := dplyr::if_else(
          dplyr::row_number() == 1,
          .data[[col_name]][1],
          NA_character_
        )
      ) |>
      dplyr::group_modify(~ add_row(.x)) |>
      ungroup() |>
      head(-1)
  }
}

# Example usage:
#
# # From R directly:
# source("path/to/forest.R")
#
# # From Python via rpy2:
# # from statract.r import init
# # init()  # This will load forest.R automatically
#
# # Create a forest plot:
# data <- data.frame(
#   name = c("Variable 1", "Variable 2", "Variable 3"),
#   n = c(100, 200, 150),
#   event = c(10, 20, 15),
#   oddsratio = c(1.5, 2.0, 0.8),
#   lo = c(1.2, 1.5, 0.6),
#   hi = c(1.8, 2.5, 1.0),
#   desc = c("1.5 (1.2-1.8)", "2.0 (1.5-2.5)", "0.8 (0.6-1.0)"),
#   pvalue = c(0.01, 0.001, 0.05)
# )
#
# plot <- forest_plot(data)
# print(plot)
