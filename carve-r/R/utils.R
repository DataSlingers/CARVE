# Shared helpers. Mirrors _utils.py. A leaf: nothing here calls into the
# rest of the package.

# Evaluates code under a fixed seed and restores the caller's RNG state
# afterwards. The RNG kind is pinned, so a user's RNGkind() setting cannot
# change the draws. With seed = NULL the code draws from the caller's stream.
seeded <- function(seed, code) {
  if (is.null(seed)) {
    return(code)
  }
  withr::with_seed(
    seed,
    code,
    .rng_kind = "Mersenne-Twister",
    .rng_normal_kind = "Inversion",
    .rng_sample_kind = "Rejection"
  )
}

split_subsample_indices <- function(n_samples, subsample_ratio, random_state) {
  train_size <- floor(subsample_ratio * n_samples)
  train <- seeded(random_state, sample.int(n_samples, train_size))
  list(train = train, test = setdiff(seq_len(n_samples), train))
}

# Density-based methods mark unassigned points with a negative label; those
# are not counted as a cluster.
count_clusters <- function(labels) {
  if (is.null(labels) || length(labels) == 0L) {
    return(0L)
  }
  length(unique(labels[labels >= 0]))
}

# A single number K means 2:K, as in Python. expand_scalar = FALSE keeps a
# single value as it is, for values read from an estimator grid.
coerce_n_clusters <- function(value, expand_scalar = TRUE) {
  if (!is.numeric(value) || !is.null(dim(value))) {
    stop("n_clusters must be a numeric vector.", call. = FALSE)
  }
  if (anyNA(value) || any(value != round(value))) {
    stop("n_clusters values must be whole numbers.", call. = FALSE)
  }
  if (expand_scalar && length(value) == 1L) {
    if (value < 2) {
      stop("n_clusters int must be >= 2.", call. = FALSE)
    }
    return(seq.int(2L, as.integer(value)))
  }
  if (any(value < 2)) {
    stop("All n_clusters values must be >= 2.", call. = FALSE)
  }
  as.integer(value)
}

# Mean, standard error and the 95th and 5th percentiles of the resampled
# ARI scores, ignoring NaN. Quantile type 7 is numpy's default.
summarize_ari_scores <- function(x) {
  x <- as.numeric(x)
  ok <- x[!is.na(x)]
  if (length(ok) == 0L) {
    return(c(mean = NaN, se = NaN, upper = NaN, lower = NaN))
  }
  se <- if (length(ok) > 1L) stats::sd(ok) / sqrt(length(ok)) else NaN
  c(
    mean = mean(ok),
    se = se,
    upper = unname(stats::quantile(ok, 0.95, type = 7)),
    lower = unname(stats::quantile(ok, 0.05, type = 7))
  )
}

# Sparse input is densified: several estimators need dense input, and a
# sparse matrix would only move the failure into a worker.
as_data_matrix <- function(X, dense_warn_elements = 5e7) {
  if (is.data.frame(X)) {
    numeric_cols <- vapply(X, is.numeric, logical(1))
    if (!all(numeric_cols)) {
      stop(sprintf(
        "X has non-numeric columns: %s.",
        paste(names(X)[!numeric_cols], collapse = ", ")
      ), call. = FALSE)
    }
    X <- as.matrix(X)
  } else if (inherits(X, "Matrix")) {
    n_elements <- as.numeric(nrow(X)) * ncol(X)
    if (n_elements > dense_warn_elements) {
      warning(sprintf(
        "Densifying a sparse matrix with %s entries (~%.1f GB dense). Pass a reduced representation instead, such as principal components.",
        format(n_elements, big.mark = ",", scientific = FALSE),
        n_elements * 8 / 1e9
      ), call. = FALSE)
    }
    X <- as.matrix(X)
  } else if (is.numeric(X) && is.null(dim(X))) {
    X <- matrix(X, ncol = 1L)
  } else if (!is.matrix(X) || !is.numeric(X)) {
    stop(
      "X must be a numeric matrix, a numeric data frame, a sparse Matrix or a numeric vector.",
      call. = FALSE
    )
  }
  storage.mode(X) <- "double"
  if (any(!is.finite(X))) {
    stop("X contains missing or infinite values.", call. = FALSE)
  }
  X
}

# Runs expr, muffles its warnings and returns them with the value. The
# runner uses it inside each resample, so warnings from parallel workers
# reach the main process.
collect_warnings <- function(expr) {
  messages <- character()
  value <- withCallingHandlers(expr, warning = function(w) {
    messages <<- c(messages, conditionMessage(w))
    invokeRestart("muffleWarning")
  })
  list(value = value, warnings = messages)
}

check_dots <- function(...) {
  if (...length() == 0L) {
    return(invisible(NULL))
  }
  arg_names <- names(list(...))
  if (is.null(arg_names)) {
    arg_names <- rep("", ...length())
  }
  arg_names[arg_names == ""] <- "<unnamed>"
  stop(sprintf(
    "Unknown argument%s: %s.",
    if (length(arg_names) > 1L) "s" else "",
    paste(arg_names, collapse = ", ")
  ), call. = FALSE)
}

# Formatting that reproduces the Python package's labels and messages.
format_repr <- function(x) {
  if (is.null(x)) {
    return("None")
  }
  if (is.character(x) && length(x) == 1L) {
    return(paste0("'", x, "'"))
  }
  if (is.logical(x) && length(x) == 1L && !is.na(x)) {
    return(if (x) "True" else "False")
  }
  if (is.numeric(x) && length(x) == 1L) {
    return(format(x, digits = 15))
  }
  paste(deparse(x), collapse = "")
}

python_list <- function(x) {
  paste0("[", paste0("'", x, "'", collapse = ", "), "]")
}

format_params <- function(params) {
  if (length(params) == 0L) {
    return("{}")
  }
  parts <- vapply(
    names(params),
    function(name) paste0("'", name, "': ", format_repr(params[[name]])),
    character(1)
  )
  paste0("{", paste(parts, collapse = ", "), "}")
}

format_param_value <- function(val) {
  if (is.logical(val) && length(val) == 1L && !is.na(val)) {
    return(if (val) "True" else "False")
  }
  if (is.numeric(val) && length(val) == 1L && !is.na(val)) {
    if (val == round(val)) {
      return(sprintf("%.0f", val))
    }
    return(sprintf("%.3g", val))
  }
  if (is.character(val) && length(val) == 1L) {
    return(val)
  }
  paste(deparse(val), collapse = "")
}

title_case <- function(x) {
  words <- strsplit(x, " ", fixed = TRUE)[[1L]]
  paste(
    paste0(toupper(substring(words, 1L, 1L)), tolower(substring(words, 2L))),
    collapse = " "
  )
}
