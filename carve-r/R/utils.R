# Shared helpers. Mirrors _utils.py. A leaf: nothing here calls into the
# rest of the package.

# Evaluates code under a fixed seed and restores the caller's RNG state
# afterwards. The RNG kind is pinned, so a user's RNGkind() setting cannot
# change the draws. With seed = NULL the code draws from the caller's stream.
# The anchor draw and the pipeline allocation pass kind = "L'Ecuyer-CMRG":
# Python takes them from a numpy Generator, apart from the subsample draws,
# and a Mersenne-Twister draw from the same seed would repeat the start of
# resample 0's first subsample.
seeded <- function(seed, code, kind = "Mersenne-Twister") {
  if (is.null(seed)) {
    return(code)
  }
  withr::with_seed(
    seed,
    code,
    .rng_kind = kind,
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

# A suggested package is loaded only when a function needs it. has_package()
# is its own function so that tests can pretend a package is missing.
has_package <- function(package) {
  requireNamespace(package, quietly = TRUE)
}

require_package <- function(package, needed_for) {
  if (!has_package(package)) {
    stop(sprintf(
      "%s is required for %s. Install it with: install.packages(\"%s\")",
      package, needed_for, package
    ), call. = FALSE)
  }
  invisible(TRUE)
}

NOISE_POLICIES <- c("drop", "as_cluster", "singleton")

check_noise_policy <- function(policy) {
  if (!is.character(policy) || length(policy) != 1L || !policy %in% NOISE_POLICIES) {
    stop(sprintf(
      "Unknown noise_policy %s. Expected 'drop', 'as_cluster', or 'singleton'.",
      format_repr(policy)
    ), call. = FALSE)
  }
  invisible(policy)
}

# Resolves the negative labels that density-based methods give unassigned
# samples. "drop" removes them from the resample, so they count as not
# drawn; "as_cluster" keeps -1 as an ordinary label; "singleton" gives each
# its own cluster. noise_fraction is the share of noise before the policy.
apply_noise_policy <- function(indices, labels, policy = "drop") {
  check_noise_policy(policy)
  if (length(labels) == 0L) {
    return(list(indices = indices, labels = labels, noise_fraction = 0))
  }
  noise <- labels < 0
  fraction <- mean(noise)
  if (!any(noise) || policy == "as_cluster") {
    return(list(indices = indices, labels = labels, noise_fraction = fraction))
  }
  if (policy == "drop") {
    return(list(indices = indices[!noise], labels = labels[!noise], noise_fraction = fraction))
  }
  assigned <- labels[!noise]
  start <- if (length(assigned) > 0L) max(assigned) + 1L else 0L
  labels[noise] <- seq.int(start, length.out = sum(noise))
  list(indices = indices, labels = labels, noise_fraction = fraction)
}

# How far below the median sample's score, on the [0, 1] score scale, a
# sample must lie before noise_mask() may flag it.
NOISE_MARGIN <- 0.05

# Flags the samples whose score marks them as ambiguous: a NaN score, or a
# score strictly below the quantile of the finite scores and more than
# margin below their median, so that tied scores flag nothing. n_target is
# how many finite samples the quantile would flag if every score were
# distinct, counted through the quantile itself so that it agrees with the
# cutoff. Quantile type 7 is numpy's default.
noise_mask <- function(scores, quantile, margin = NOISE_MARGIN) {
  finite <- is.finite(scores)
  if (!any(finite)) {
    stop("scores contains no finite value.", call. = FALSE)
  }
  values <- scores[finite]
  cutoff <- stats::quantile(values, quantile, type = 7, names = FALSE)
  median <- stats::median(values)
  mask <- !finite
  mask[finite] <- values < cutoff & values < median - margin
  ranks <- seq_along(values) - 1
  n_target <- sum(ranks < stats::quantile(ranks, quantile, type = 7, names = FALSE))
  list(mask = mask, n_target = as.integer(n_target), cutoff = cutoff, median = median)
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
  if (is.null(val)) {
    return("None")
  }
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

# sklearn's adjusted_rand_score, through the pair confusion matrix. Counts
# are doubles, so very large n loses exactness but not accuracy.
adjusted_rand_index <- function(a, b) {
  n <- length(a)
  counts <- as.numeric(table(a, b))
  sum_squares <- sum(counts^2)
  n_rows <- as.numeric(table(a))
  n_cols <- as.numeric(table(b))
  tp <- sum_squares - n
  fp <- sum(n_cols^2) - sum_squares
  fn <- sum(n_rows^2) - sum_squares
  tn <- n^2 - fp - fn - sum_squares
  if (fn == 0 && fp == 0) {
    return(1)
  }
  2 * (tp * tn - fn * fp) / ((tp + fn) * (fn + tn) + (tp + fp) * (fp + tn))
}

# Renames labels to the reference values by maximum-overlap (Hungarian)
# matching. Labels with no partner, when there are more clusters than
# reference values, take fresh ids past the largest matched reference
# value, so they never merge with a matched cluster. Only positions where
# keep is TRUE take part in the matching. With no position kept there is
# nothing to match, and the labels come back unchanged.
align_cluster_labels <- function(reference_labels, labels, keep = NULL) {
  if (is.null(keep)) {
    keep <- rep(TRUE, length(labels))
  }
  if (!any(keep)) {
    return(labels)
  }
  ref <- reference_labels[keep]
  lab <- labels[keep]
  ref_classes <- sort(unique(ref))
  pred_classes <- sort(unique(lab))
  overlap <- table(factor(ref, levels = ref_classes), factor(lab, levels = pred_classes))
  overlap <- matrix(as.numeric(overlap), nrow = length(ref_classes))

  mapping <- rep(NA_real_, length(pred_classes))
  if (nrow(overlap) <= ncol(overlap)) {
    cols <- as.integer(clue::solve_LSAP(overlap, maximum = TRUE))
    mapping[cols] <- ref_classes
  } else {
    rows <- as.integer(clue::solve_LSAP(t(overlap), maximum = TRUE))
    mapping <- as.numeric(ref_classes[rows])
  }

  all_pred <- sort(unique(labels))
  full <- mapping[match(all_pred, pred_classes)]
  next_id <- max(ref_classes) + 1
  for (i in which(is.na(full))) {
    full[i] <- next_id
    next_id <- next_id + 1
  }
  aligned <- full[match(labels, all_pred)]
  if (is.integer(reference_labels)) as.integer(aligned) else aligned
}

n_cores <- function() {
  n <- parallel::detectCores(logical = TRUE)
  if (is.na(n) || n < 1L) {
    n <- 1L
  }
  limit <- Sys.getenv("_R_CHECK_LIMIT_CORES_")
  if (nzchar(limit) && !identical(tolower(limit), "false")) {
    n <- min(n, 2L)
  }
  as.integer(n)
}

# Splits n_jobs into resample workers (outer) and classifier threads per
# worker (inner), following joblib's convention for n_jobs: a positive
# count, -1 for every core, -2 for all but one, NULL for one worker. The
# worker count is capped at n_resamples.
resolve_core_budget <- function(n_jobs, n_resamples) {
  cores <- n_cores()
  if (is.null(n_jobs)) {
    outer <- 1L
  } else if (n_jobs == 0) {
    stop("n_jobs == 0 has no meaning; use 1 or a negative count", call. = FALSE)
  } else if (n_jobs < 0) {
    outer <- max(cores + 1L + as.integer(n_jobs), 1L)
  } else {
    outer <- as.integer(n_jobs)
  }
  outer <- max(1L, min(outer, as.integer(n_resamples)))
  c(outer = outer, inner = max(1L, cores %/% outer))
}

# A neighbor count chosen for the full data reaches further on a subsample.
# Scaling it by n_fit / n_full keeps the neighborhood the same size. The
# count comes from params or else from the estimator's default; it is
# rounded half to even, as Python's round() does, never set below 2 and
# never raised.
scale_neighbor_count <- function(estimator, params, n_fit, n_full) {
  if (n_fit >= n_full) {
    return(params)
  }
  base <- if ("n_neighbors" %in% names(params)) {
    params$n_neighbors
  } else if ("n_neighbors" %in% names(formals(estimator))) {
    formals(estimator)$n_neighbors
  } else {
    NULL
  }
  if (!is.numeric(base) || length(base) != 1L || is.na(base) || base != round(base)) {
    return(params)
  }
  scaled <- max(2, round(base * n_fit / n_full))
  params$n_neighbors <- as.integer(min(base, scaled))
  params
}

# Runs a clustering function on X. random_state is passed only when the
# function has that argument and the grid does not set it.
call_estimator <- function(estimator, X, params, random_state = NULL) {
  args <- c(list(X), params)
  if ("random_state" %in% names(formals(estimator)) && !"random_state" %in% names(params)) {
    args["random_state"] <- list(random_state)
  }
  labels <- do.call(estimator, args)
  if (is.factor(labels)) {
    labels <- as.integer(labels)
  }
  if (!is.numeric(labels) || length(labels) != nrow(X) || anyNA(labels) ||
      any(labels != round(labels))) {
    stop(sprintf(
      "A clustering function must return one integer label per row of X (%d).",
      nrow(X)
    ), call. = FALSE)
  }
  as.integer(labels)
}

# Builds the function that predicts held-out labels. The default is a
# ranger forest with the settings of Python's default classifier: n_trees
# trees, depth at most n_features, floor(sqrt(n_features)) candidate
# features per split. As in sklearn, it predicts the class with the largest
# mean leaf probability, not the majority vote of the trees. A custom
# classifier gets n_threads and random_state only when it has those
# arguments.
default_generalizability_classifier <- function(classifier, n_features, n_trees,
                                                random_state, n_threads) {
  check <- function(predicted, x_test) {
    if (length(predicted) != nrow(x_test)) {
      stop(sprintf(
        "A classifier must return one label per row of x_test (%d).",
        nrow(x_test)
      ), call. = FALSE)
    }
    as.integer(predicted)
  }
  if (is.null(classifier)) {
    return(function(x_train, y_train, x_test) {
      check(ranger_predict(
        x_train, y_train, x_test,
        n_trees = n_trees,
        mtry = max(1L, as.integer(floor(sqrt(n_features)))),
        max_depth = as.integer(n_features),
        seed = random_state,
        n_threads = n_threads
      ), x_test)
    })
  }
  arguments <- names(formals(classifier))
  function(x_train, y_train, x_test) {
    args <- list(x_train = x_train, y_train = y_train, x_test = x_test)
    if ("n_threads" %in% arguments) {
      args$n_threads <- n_threads
    }
    if ("random_state" %in% arguments) {
      args["random_state"] <- list(random_state)
    }
    check(do.call(classifier, args), x_test)
  }
}

ranger_predict <- function(x_train, y_train, x_test, n_trees, mtry, max_depth,
                           seed, n_threads) {
  classes <- sort(unique(y_train))
  if (length(classes) == 1L) {
    return(rep(classes, nrow(x_test)))
  }
  # ranger finds no covariates in a matrix without column names.
  feature_names <- paste0("x", seq_len(ncol(x_train)))
  colnames(x_train) <- feature_names
  colnames(x_test) <- feature_names
  # ranger treats seed 0 as "no seed" and draws from the R generator, so the
  # seed is shifted by one. predict() takes its own seed, used to break ties
  # in the vote.
  # probability = TRUE grows probability trees, whose leaves hold class
  # shares as in sklearn. ranger's default leaf size for them is 10, so
  # min.node.size = 1 matches sklearn's min_samples_leaf = 1.
  forest <- ranger::ranger(
    x = x_train,
    y = factor(y_train, levels = classes),
    num.trees = n_trees,
    mtry = mtry,
    max.depth = max_depth,
    min.node.size = 1L,
    probability = TRUE,
    seed = seed + 1L,
    num.threads = n_threads,
    verbose = FALSE
  )
  probabilities <- stats::predict(
    forest, data = x_test, seed = seed + 1L, num.threads = n_threads
  )$predictions
  largest_probability_class(probabilities)
}

# The class with the largest probability in each row. Ties go to the first
# class in sorted order, as np.argmax does over sklearn's sorted classes_.
largest_probability_class <- function(probabilities) {
  classes <- as.integer(colnames(probabilities))
  ordered <- order(classes)
  probabilities <- probabilities[, ordered, drop = FALSE]
  classes[ordered][max.col(probabilities, ties.method = "first")]
}

# The anchors of a run, sorted, or NULL when the exact path applies. With
# consensus_anchors = NULL there are min(n_samples, anchor_threshold)
# anchors, so the anchor count does not jump as n crosses the threshold. A
# value in (0, 1] is a share of the samples and a whole number above 1 a
# count. Python tells the two apart by type; R numbers are doubles.
resolve_anchors <- function(n_samples, consensus_anchors, anchor_threshold, random_state) {
  if (is.null(consensus_anchors)) {
    m <- min(n_samples, anchor_threshold)
  } else {
    valid <- is.numeric(consensus_anchors) && length(consensus_anchors) == 1L &&
      !is.na(consensus_anchors)
    if (valid && consensus_anchors > 1 && consensus_anchors == round(consensus_anchors)) {
      m <- consensus_anchors
    } else if (valid && consensus_anchors > 0 && consensus_anchors <= 1) {
      m <- round(consensus_anchors * n_samples)
    } else {
      stop(sprintf(
        "consensus_anchors given as a fraction must be in (0, 1], got %s.",
        format_repr(consensus_anchors)
      ), call. = FALSE)
    }
  }
  # The exact path comes first: a run this small is exact, not misconfigured.
  if (m >= n_samples) {
    return(NULL)
  }
  if (m < 2) {
    stop(sprintf(
      "The consensus anchor count must be at least 2, got %d. It comes from consensus_anchors=%s when that is set, and otherwise from min(n_samples, anchor_threshold=%s).",
      as.integer(m), format_repr(consensus_anchors), format_param_value(anchor_threshold)
    ), call. = FALSE)
  }
  sort(seeded(random_state, sample.int(n_samples, m), kind = "L'Ecuyer-CMRG"))
}

# One row per configuration, pipeline and sweep value of a randomized run.
# The scores are over the resamples that used the row's pipeline at the
# row's configuration only, and n_resamples counts them. Rows are sorted by
# the number in method_id (m10 after m9), then pipeline, then sweep value.
summarize_preprocessing_records <- function(pipeline_records, pipelines, sweep_param = "n_clusters") {
  rows <- list()
  for (record in pipeline_records) {
    runs <- record$runs
    for (label in unique(runs$pipeline)) {
      own <- runs[runs$pipeline == label, , drop = FALSE]
      spec <- pipelines[[label]]
      stability <- summarize_ari_scores(own$ari_stability)
      generalizability <- summarize_ari_scores(own$ari_generalizability)
      row <- list(
        method_id = record$method_id,
        method_label = record$method_label,
        pipeline = label,
        normalization = spec$normalization$label,
        dim_reduction = spec$dim_reduction$label
      )
      row[[sweep_param]] <- record$sweep_value
      rows[[length(rows) + 1L]] <- c(row, list(
        n_resamples = nrow(own),
        ari_stability = stability[["mean"]],
        ari_stability_se = stability[["se"]],
        ari_generalizability = generalizability[["mean"]],
        ari_generalizability_se = generalizability[["se"]],
        n_clusters_observed = mean(own$n_clusters),
        sweep_param = sweep_param,
        sweep_value = record$sweep_value,
        sweep_rank = record$sweep_rank
      ))
    }
  }
  summary <- do.call(rbind, lapply(rows, function(row) {
    data.frame(row, check.names = FALSE, stringsAsFactors = FALSE)
  }))
  order_by <- order(
    as.integer(substring(summary$method_id, 2L)),
    summary$pipeline,
    summary$sweep_value,
    method = "radix"
  )
  summary <- summary[order_by, , drop = FALSE]
  rownames(summary) <- NULL
  summary
}

# Single-cell input. SingleCellExperiment and Seurat objects keep cells as
# columns; CARVE clusters a matrix with cells as rows.
check_one_source <- function(assay, reduction) {
  if (!is.null(assay) && !is.null(reduction)) {
    stop(
      "Pass at most one of reduction= and assay=; they select the same thing from different places.",
      call. = FALSE
    )
  }
}

# A sparse matrix stays sparse, so as_data_matrix() can warn before it
# densifies a large one.
cells_by_features <- function(values) {
  if (inherits(values, "Matrix")) Matrix::t(values) else t(as.matrix(values))
}

select_dims <- function(X, n_dims) {
  if (is.null(n_dims)) {
    return(X)
  }
  if (!is.numeric(n_dims) || length(n_dims) != 1L || is.na(n_dims) || n_dims < 1 ||
      n_dims != round(n_dims)) {
    stop("n_dims must be a positive whole number.", call. = FALSE)
  }
  if (n_dims > ncol(X)) {
    stop(sprintf(
      "n_dims=%d exceeds the %d available components in the selected representation.",
      as.integer(n_dims), ncol(X)
    ), call. = FALSE)
  }
  X[, seq_len(n_dims), drop = FALSE]
}

# The first two columns of an embedding, for the scatter plots. where names
# the embedding in the error.
embedding_2d <- function(values, where) {
  values <- as.matrix(values)
  if (ncol(values) < 2L) {
    stop(sprintf(
      "%s has shape (%d, %d); a basis needs at least two columns.",
      where, nrow(values), ncol(values)
    ), call. = FALSE)
  }
  unname(values[, 1:2, drop = FALSE])
}
