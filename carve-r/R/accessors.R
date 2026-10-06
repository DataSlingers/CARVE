#' @include AllGenerics.R
NULL

# Selection, labels and accessors of a fitted CARVE object. Mirrors the
# query methods of api.CARVE.

select_row <- function(fit, measure, rule, not_two = FALSE, k = NULL, sweep_value = NULL) {
  results <- fit@estimator_results
  param <- sweep_param_name(results)
  if (!is.null(k) && !is.null(sweep_value)) {
    stop("Pass at most one of k= and sweep_value=.", call. = FALSE)
  }
  if (!is.null(k) && param != "n_clusters") {
    stop(sprintf(
      "This run swept '%s', not 'n_clusters'. Use sweep_value=... to pin a %s, and consensus_k=... to fix the number of clusters used to cut the consensus matrix.",
      param, param
    ), call. = FALSE)
  }
  pin <- if (!is.null(k)) k else sweep_value
  if (is.null(pin)) {
    row <- select_best_row_by_rule(results, measure, rule, not_two = not_two)
  } else {
    pinned <- results[which(isclose(as.numeric(results$sweep_value), as.numeric(pin))), , drop = FALSE]
    if (nrow(pinned) == 0L) {
      stop(sprintf(
        "No configurations found for %s=%s.",
        if (param == "n_clusters") "k" else param,
        format_param_value(pin)
      ), call. = FALSE)
    }
    row <- select_best_row_by_rule(pinned, measure, rule)
  }
  list(row = row, config_id = config_id_of(row), n_clusters = observed_k(row), pinned = !is.null(pin))
}

config_key <- function(fit, config_id) {
  key <- if (is.numeric(config_id) && length(config_id) == 1L) as.character(as.integer(config_id)) else NA_character_
  if (is.na(key) || !key %in% names(fit@consensus_matrices)) {
    stop(sprintf("config_id %s is not in estimator_results(fit).", format_repr(config_id)), call. = FALSE)
  }
  key
}

# Average-linkage clustering of 1 - consensus. The matrix is symmetrized
# and clipped to [0, 1], and pairs never drawn together count as 0.5.
cut_consensus <- function(consensus_matrix, n_clusters, estimator = NULL) {
  S <- 0.5 * (consensus_matrix + t(consensus_matrix))
  diag(S) <- 1
  S <- pmin(pmax(S, 0), 1)
  if (anyNA(S)) {
    S[is.na(S)] <- 0.5
    diag(S) <- 1
  }
  D <- 1 - S
  diag(D) <- 0
  if (is.null(estimator)) {
    tree <- stats::hclust(stats::as.dist(D), method = "average")
    return(as.integer(stats::cutree(tree, k = n_clusters)))
  }
  args <- list(D)
  if ("n_clusters" %in% names(formals(estimator))) {
    args$n_clusters <- n_clusters
  }
  labels <- do.call(estimator, args)
  if (length(labels) != nrow(D)) {
    stop(sprintf(
      "The estimator must return one label per sample (%d).",
      nrow(D)
    ), call. = FALSE)
  }
  as.integer(labels)
}

#' Select a configuration
#'
#' `get_k()` returns the number of clusters of the configuration a selection
#' rule picks, `get_sweep_value()` the value of the swept parameter there,
#' and `get_estimator()` the estimator with that configuration's parameters
#' filled in.
#'
#' `rule = "max"` takes the configuration with the highest value of
#' `measure`. `"1se"` takes the finest configuration, the one with the highest
#' `sweep_rank`, whose value is within one standard error of the highest.
#' `"quantile"` takes the finest configuration whose value lies between the
#' 5th and 95th percentiles of the resampled values at the best one. Only the
#' three ARI measures have standard errors and percentiles; for the others,
#' `"1se"` and `"quantile"` fall back to `"max"` with a warning.
#'
#' On an axis other than `n_clusters`, the number of clusters is the rounded
#' mean of the counts observed across resamples.
#'
#' @param fit A [CARVE-class] object from [carve()].
#' @param measure The criterion to select on: `"stability"` (or `"s"`),
#'   `"generalizability"` (`"g"`), `"average"` (the mean of the two),
#'   `"pac"`, `"gini"`, `"ce"` or `"accuracy"`, or the name of the matching
#'   column of [estimator_results()], such as `"ari_stability"`.
#' @param rule `"max"`, `"1se"` or `"quantile"`.
#' @param not_two Leave out configurations with two clusters.
#' @param ... Not used. An argument name these functions do not know is an
#'   error.
#' @return `get_k()` returns an integer and `get_sweep_value()` a number.
#'   `get_estimator()` returns a function `(X, random_state = NULL)` that
#'   clusters `X` with the selected configuration; its attributes
#'   `"estimator"` and `"params"` hold the estimator's name and parameters.
#' @seealso [get_labels()], [estimator_results()]
#' @examples
#' set.seed(1)
#' X <- rbind(matrix(rnorm(60, 0, 0.3), ncol = 2), matrix(rnorm(60, 3, 0.3), ncol = 2))
#' fit <- carve(X, n_clusters = 2:4, n_resamples = 10, random_state = 0,
#'              estimator_param_grids = list(estimator_grid(KMeans, n_clusters = 2:4)))
#' get_k(fit)
#' get_k(fit, measure = "generalizability", rule = "max")
#' selected <- get_estimator(fit)
#' attr(selected, "params")
#' @rdname get_k
#' @export
setMethod("get_k", "CARVE", function(fit, measure = "stability", rule = "1se",
                                     not_two = FALSE, ...) {
  check_dots(...)
  select_best_k(fit@estimator_results, measure, rule, not_two = not_two)
})

#' @rdname get_k
#' @export
setMethod("get_sweep_value", "CARVE", function(fit, measure = "stability", rule = "1se",
                                               not_two = FALSE, ...) {
  check_dots(...)
  as.numeric(select_row(fit, measure, rule, not_two = not_two)$row$sweep_value)
})

#' @rdname get_k
#' @export
setMethod("get_estimator", "CARVE", function(fit, measure = "stability", rule = "1se",
                                             not_two = FALSE, ...) {
  check_dots(...)
  row <- select_best_row_by_rule(fit@estimator_results, measure, rule, not_two = not_two)
  grid <- Find(function(g) identical(g$name, row$estimator[[1L]]), fit@estimator_param_grids)
  estimator <- grid$estimator
  arguments <- setdiff(names(formals(estimator))[-1L], c("random_state", "..."))
  params <- row_to_estimator_params(row, arguments)
  selected <- function(X, random_state = NULL) {
    call_estimator(estimator, as_data_matrix(X), params, random_state = random_state)
  }
  attr(selected, "estimator") <- grid$name
  attr(selected, "params") <- params
  selected
})

#' Consensus cluster labels
#'
#' `get_labels()` selects a configuration, as [get_k()] does, and cuts its
#' consensus matrix into clusters by average-linkage hierarchical clustering
#' of `1 - consensus`, at the configuration's number of clusters. Pairs of
#' samples never drawn together count as 0.5.
#'
#' When the reference labels have as many clusters as the cut, the clusters
#' are renamed to match them by maximum overlap; samples whose reference
#' label is -1 take no part in the matching. `get_labels()` keeps no state
#' between calls, so to keep cluster names stable across calls, pass earlier
#' labels as `reference_labels`.
#'
#' @inheritParams get_k
#' @param k Use a configuration with this number of clusters, the one `rule`
#'   picks among those with that `k`. Only for runs that sweep `n_clusters`.
#' @param sweep_value Use a configuration at this value of the swept
#'   parameter. Give at most one of `k` and `sweep_value`; with either,
#'   `not_two` is not used.
#' @param consensus_k Cut into this many clusters instead.
#' @param mode `"default"` and `"stability"` cut the stability consensus
#'   matrix; `"generalizability"` cuts the consensus of the classifier's
#'   held-out predictions.
#' @param estimator `NULL` for the average-linkage cut, or a function of the
#'   distance matrix `1 - consensus` that returns one label per sample. If it
#'   has an `n_clusters` argument, it gets the number of clusters.
#' @param reference_labels Labels to match, one per sample, coded as in
#'   [carve()]. `NULL` uses the labels given to [carve()], if any.
#' @return An integer vector of cluster labels, one per sample.
#' @seealso [get_k()], [consensus_matrix()]
#' @examples
#' set.seed(1)
#' X <- rbind(matrix(rnorm(60, 0, 0.3), ncol = 2), matrix(rnorm(60, 3, 0.3), ncol = 2))
#' fit <- carve(X, n_clusters = 2:4, n_resamples = 10, random_state = 0,
#'              estimator_param_grids = list(estimator_grid(KMeans, n_clusters = 2:4)))
#' labels <- get_labels(fit)
#' table(labels)
#' table(get_labels(fit, k = 3, reference_labels = labels))
#' @rdname get_labels
#' @export
setMethod("get_labels", "CARVE", function(fit, measure = "stability", rule = "1se", k = NULL,
                                          sweep_value = NULL, consensus_k = NULL,
                                          not_two = FALSE, mode = "default", estimator = NULL,
                                          reference_labels = NULL, ...) {
  check_dots(...)
  policy <- resolve_mode(mode)
  selected <- select_row(fit, measure, rule, not_two = not_two, k = k, sweep_value = sweep_value)
  # Off the k axis the cluster count is an outcome, so the consensus is cut
  # at the count observed.
  cut_k <- if (is.null(consensus_k)) selected$n_clusters else as.integer(consensus_k)
  if (cut_k < 2L) {
    stop(sprintf(
      "Cannot cut the consensus matrix at %d cluster(s). The selected configuration is degenerate; pass consensus_k= explicitly or exclude that end of the sweep.",
      cut_k
    ), call. = FALSE)
  }
  key <- as.character(selected$config_id)
  consensus <- if (policy$run_stability) {
    fit@consensus_matrices[[key]]
  } else {
    fit@consensus_generalizability_matrices[[key]]
  }
  if (is.null(consensus)) {
    stop(sprintf(
      "Consensus matrix not available for mode='%s'. This run likely used split-mode and skipped building that artifact.",
      policy$mode
    ), call. = FALSE)
  }
  labels <- cut_consensus(consensus, cut_k, estimator)
  reference <- if (is.null(reference_labels)) {
    fit@reference_labels
  } else {
    coerce_reference_labels(reference_labels, length(labels))
  }
  if (!is.null(reference) && count_clusters(reference) == length(unique(labels))) {
    labels <- align_cluster_labels(reference, labels, keep = reference >= 0)
  }
  as.integer(labels)
})

#' Contents of a CARVE fit
#'
#' `estimator_results()` returns the results table, one row per
#' configuration. `estimator_param_grids()` returns the estimator grids the
#' run evaluated, `sweep_spec()` the run's [SweepSpec-class] object, and
#' `input_data()` the data matrix.
#'
#' The columns of the results table:
#'
#' - `config_id` joins a row to its consensus matrices and per-sample
#'   scores. It counts from 0.
#' - `method_id` and `method_label` name the curve a row belongs to: an
#'   estimator with all its parameters except the swept one.
#' - `estimator` is followed by one column per estimator parameter, `NA`
#'   where an estimator does not have the parameter.
#' - `sweep_param`, `sweep_value` and `sweep_rank` place the row on the
#'   sweep axis; rank 0 is the coarsest.
#' - `n_clusters_observed` is the mean number of clusters over the
#'   resamples, with standard error `n_clusters_observed_se`, and
#'   `noise_fraction` the mean share of samples labeled noise.
#' - `ari_stability`, `ari_generalizability` and `ari_average` are means
#'   over the resamples. Each comes with `_se` (its standard error), `_upper`
#'   (the 95th percentile) and `_lower` (the 5th percentile).
#' - `consensus_pac_stability` is one minus the share of consensus values
#'   strictly between 0.05 and 0.95. `consensus_gini_stability` and
#'   `consensus_ce_stability` are the means of the per-sample stability
#'   scores, and `accuracy_generalizability` the mean held-out accuracy.
#'
#' @inheritParams get_k
#' @return `estimator_results()` returns a data frame,
#'   `estimator_param_grids()` a list of [estimator_grid()] objects,
#'   `sweep_spec()` a [SweepSpec-class] object and `input_data()` a matrix.
#' @seealso [consensus_matrix()]
#' @examples
#' set.seed(1)
#' X <- rbind(matrix(rnorm(60, 0, 0.3), ncol = 2), matrix(rnorm(60, 3, 0.3), ncol = 2))
#' fit <- carve(X, n_clusters = 2:4, n_resamples = 10, random_state = 0,
#'              estimator_param_grids = list(estimator_grid(KMeans, n_clusters = 2:4)))
#' estimator_results(fit)[, c("config_id", "n_clusters", "ari_stability", "ari_stability_se")]
#' sweep_spec(fit)
#' @rdname estimator_results
#' @export
setMethod("estimator_results", "CARVE", function(fit, ...) {
  check_dots(...)
  fit@estimator_results
})

#' @rdname estimator_results
#' @export
setMethod("estimator_param_grids", "CARVE", function(fit, ...) {
  check_dots(...)
  fit@estimator_param_grids
})

#' @rdname estimator_results
#' @export
setMethod("sweep_spec", "CARVE", function(fit, ...) {
  check_dots(...)
  fit@sweep
})

#' @rdname estimator_results
#' @export
setMethod("input_data", "CARVE", function(fit, ...) {
  check_dots(...)
  fit@input_data
})

#' Consensus matrices and per-sample scores
#'
#' `consensus_matrix()` returns a configuration's consensus matrix. Entry
#' `(i, j)` is the share of the subsamples drawing both samples in which the
#' two fell in the same cluster, and `NaN` when no subsample drew both.
#' `sample_scores()` returns a configuration's per-sample scores.
#'
#' @inheritParams get_k
#' @param config_id The configuration, a value of the `config_id` column of
#'   [estimator_results()].
#' @param type `"stability"` for the clusterings of the subsamples, or
#'   `"generalizability"` for the classifier's predictions on the held-out
#'   samples.
#' @param source `"gini"` or `"ce"` for per-sample stability, in its Gini or
#'   cross-entropy form; both run from 0 to 1, and 1 means every pair the
#'   sample was drawn with always agreed. `"accuracy"` is the share of the
#'   draws holding the sample out in which the classifier predicted its
#'   cluster.
#' @return `consensus_matrix()` returns an `n` by `n` matrix and
#'   `sample_scores()` a vector of length `n`. Stability scores are `NaN` for
#'   a sample never drawn with a partner; accuracy is 0 for a sample never
#'   held out.
#' @seealso [estimator_results()], [get_labels()]
#' @examples
#' set.seed(1)
#' X <- rbind(matrix(rnorm(60, 0, 0.3), ncol = 2), matrix(rnorm(60, 3, 0.3), ncol = 2))
#' fit <- carve(X, n_clusters = 2:4, n_resamples = 10, random_state = 0,
#'              estimator_param_grids = list(estimator_grid(KMeans, n_clusters = 2:4)))
#' M <- consensus_matrix(fit, config_id = 0)
#' dim(M)
#' summary(sample_scores(fit, config_id = 0, source = "gini"))
#' @rdname consensus_matrix
#' @export
setMethod("consensus_matrix", "CARVE", function(fit, config_id,
                                                type = c("stability", "generalizability"), ...) {
  check_dots(...)
  type <- match.arg(type)
  key <- config_key(fit, config_id)
  consensus <- if (type == "stability") {
    fit@consensus_matrices[[key]]
  } else {
    fit@consensus_generalizability_matrices[[key]]
  }
  if (is.null(consensus)) {
    stop(sprintf(
      "The %s consensus matrix is not available for this fit (mode = '%s').",
      type, fit@run_params$mode
    ), call. = FALSE)
  }
  consensus
})

#' @rdname consensus_matrix
#' @export
setMethod("sample_scores", "CARVE", function(fit, config_id, source = c("gini", "ce", "accuracy"),
                                             ...) {
  check_dots(...)
  if (identical(source, c("gini", "ce", "accuracy"))) {
    source <- "gini"
  }
  if (!is.character(source) || length(source) != 1L || !source %in% c("gini", "ce", "accuracy")) {
    stop("source must be one of: 'accuracy', 'gini', 'ce'.", call. = FALSE)
  }
  key <- config_key(fit, config_id)
  if (source == "gini") {
    if (is.null(fit@stability_gini_scores)) {
      stop("Gini stability scores are not available for this run.", call. = FALSE)
    }
    return(fit@stability_gini_scores[[key]])
  }
  if (source == "ce") {
    if (is.null(fit@stability_ce_scores)) {
      stop("CE stability scores are not available for this run.", call. = FALSE)
    }
    return(fit@stability_ce_scores[[key]])
  }
  scores <- fit@generalizability_scores[[key]]
  if (is.null(scores)) {
    stop("Generalizability scores are not available for this run.", call. = FALSE)
  }
  scores
})

#' @rdname CARVE-class
#' @param object A `CARVE` object.
#' @export
setMethod("show", "CARVE", function(object) {
  results <- object@estimator_results
  settings <- object@run_params
  sweep <- object@sweep
  cat("CARVE fit on ", nrow(object@input_data), " samples and ", ncol(object@input_data),
      " features\n", sep = "")
  cat("Sweep: ", sweep@param, " over ",
      paste(vapply(sweep@values, format_param_value, character(1)), collapse = ", "), "\n", sep = "")
  cat("Estimators: ", paste(unique(results$estimator), collapse = ", "), " (", nrow(results),
      " configurations)\n", sep = "")
  cat("Resamples: ", settings$n_resamples, " per configuration, subsample_ratio ",
      format(settings$subsample_ratio), "\n", sep = "")
  cat("Mode: ", settings$mode, "\n", sep = "")
  invisible(object)
})
