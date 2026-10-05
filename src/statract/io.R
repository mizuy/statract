#' Load all parquet files from a directory using polars
#'
#' This function loads all parquet files from the specified directory using polars,
#' using their file names (without extension) as variable names.
#' Dictionary-encoded columns are automatically preserved as Enum/Categorical.
#'
#' @param dir_path Path to the directory containing parquet files
#' @param pattern Optional pattern to filter files (default: "*.parquet")
#' @param ... Additional arguments passed to \code{pl$read_parquet}
#'
#' @return A named list where each element is a polars DataFrame loaded from a parquet file.
#'         The names are the file names without the .parquet extension.
#'
#' @examples
#' \dontrun{
#' data_list <- load_data("cache/build")
#' # Access individual data frames
#' target <- data_list$target
#' pccrc_candidate <- data_list$pccrc_candidate
#' }
#'
#' @export
load_data <- function(dir_path, pattern = "*.parquet", ...) {
    if (!requireNamespace("polars", quietly = TRUE)) {
        stop("polars package is required. Install it with: install.packages('polars')")
    }

    pl <- polars::pl

    # Check if directory exists
    if (!dir.exists(dir_path)) {
        stop(paste("Directory does not exist:", dir_path))
    }

    # Find all parquet files
    parquet_files <- list.files(
        path = dir_path,
        pattern = pattern,
        full.names = TRUE,
        recursive = FALSE
    )

    if (length(parquet_files) == 0) {
        warning(paste("No parquet files found in", dir_path))
        return(list())
    }

    # Load each parquet file using polars
    data_list <- list()
    for (file_path in parquet_files) {
        # Get file name without extension
        file_name <- tools::file_path_sans_ext(basename(file_path))

        # Skip if file name is empty or invalid
        if (file_name == "" || !grepl("^[a-zA-Z_][a-zA-Z0-9_]*$", file_name)) {
            warning(paste("Skipping file with invalid name:", file_path))
            next
        }

        # Load the parquet file using polars
        tryCatch(
            {
                data_list[[file_name]] <- pl$read_parquet(file_path, ...)
                message(paste("Loaded:", file_name, "from", basename(file_path)))
            },
            error = function(e) {
                warning(paste("Failed to load", file_path, ":", e$message))
            }
        )
    }

    return(data_list)
}
