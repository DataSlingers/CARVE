#' @include AllGenerics.R
NULL

#' Validate clusterings by resampling
#'
#' `carve()` clusters many subsamples of the data with every configuration in
#' the estimator grids and scores each configuration on two criteria.
#' Stability is the adjusted Rand index (ARI) between the clusterings of two
#' overlapping subsamples, computed on the samples they share.
#' Generalizability is the ARI between the clustering of the held-out samples
#' and the labels a random forest predicts for them after training on the
#' first subsample and its clustering. Each configuration also gets consensus
#' matrices: for every pair of samples, the share of the subsamples drawing
#' both in which the two fell in the same cluster.
#'
#' Resample `b`, counting from 0, draws its first subsample with seed
#' `random_state + b` and its second with `random_state + b + n_resamples`,
#' and its estimator and classifier use `random_state + b`. The results are
#' therefore the same for every `n_jobs` and `BPPARAM`, and the session's
#' random number stream is left as it was.
#'
#' `n_jobs` is a core budget. The resamples are spread over `n_jobs` worker
#' processes, at most `n_resamples` of them, and each worker's random forest
#' gets the core count divided by the number of workers. With `n_jobs = 1`
#' one process runs every resample and the forest uses every core; with
#' `n_jobs = -1` there is one worker per core, each with a single-threaded
#' forest. Every worker holds its own copy of the data and of its forest, so
#' memory grows with the worker count.
#'
#' The fit keeps two `n` by `n` consensus matrices per configuration, stored
#' as doubles: `8 * n^2` bytes each, twice what the Python package needs for
#' the same run. With `verbose = 2` the header reports the total.
#'
#' @param x The data, one row per sample: a numeric matrix, a numeric data
#'   frame, a sparse matrix from the Matrix package, or a numeric vector.
#' @param ... Not used. An argument name `carve()` does not know is an error.
#' @param n_clusters Numbers of clusters to evaluate. A single number `K`
#'   means `2:K`. Custom grids set their own values.
#' @param n_resamples Resamples per configuration.
#' @param subsample_ratio Share of the samples drawn, without replacement,
#'   into each subsample. Must lie strictly between 0 and 1.
#' @param estimator_param_grids `"light"` (KMeans, Ward-linkage agglomerative
#'   and self-tuning spectral clustering), `"full"` (adds average, single and
#'   complete linkage and RBF spectral clustering), or a list of
#'   [estimator_grid()] specifications. All grids must sweep the same
#'   parameter over the same values.
#' @param classifier `NULL` for the default random forest, or a function
#'   `(x_train, y_train, x_test)` that returns one predicted label per row of
#'   `x_test`. If it also has `n_threads` or `random_state` arguments, CARVE
#'   fills them in.
#' @param n_trees Trees in the default random forest. Not used when
#'   `classifier` is given.
#' @param reference_labels Labels, one per sample, that [get_labels()]
#'   renames its clusters to match. Whole numbers without `NA` keep their
#'   values. Any other vector, including whole numbers with an `NA`, is
#'   coded 1, 2, ... in order of first appearance, and `NA` becomes -1.
#' @param mode `"default"` scores both criteria, `"stability"` skips
#'   generalizability and `"generalizability"` skips stability. The last two
#'   are experimental.
#' @param n_jobs Core budget; see Details. `-1` means every core and `-2` all
#'   but one.
#' @param BPPARAM Optional [BiocParallel::BiocParallelParam-class] backend for
#'   the resamples. Its worker count replaces the one `n_jobs` would set.
#' @param random_state Seed of the run. `NULL` means 0.
#' @param show_progress Show a progress bar over the configurations.
#' @param verbose 0 prints nothing, 1 prints a line per configuration, and 2
#'   also prints a header and a footer.
#'
#' @return A [CARVE-class] object.
#' @seealso [get_k()], [get_labels()], [estimator_results()]
#' @examples
#' set.seed(1)
#' X <- rbind(
#'   matrix(rnorm(60, mean = 0, sd = 0.3), ncol = 2),
#'   matrix(rnorm(60, mean = 3, sd = 0.3), ncol = 2)
#' )
#' fit <- carve(X, n_clusters = 2:4, n_resamples = 10, random_state = 0)
#' fit
#' estimator_results(fit)[, c("method_label", "n_clusters", "ari_stability")]
#' @rdname carve
#' @export
setMethod("carve", "ANY", function(x, n_clusters = 2:10, n_resamples = 100,
                                   subsample_ratio = 0.618, estimator_param_grids = "light",
                                   classifier = NULL, n_trees = 100, reference_labels = NULL,
                                   mode = "default", n_jobs = 1, BPPARAM = NULL,
                                   random_state = NULL, show_progress = FALSE, verbose = 0,
                                   ...) {
  check_dots(...)
  fit_carve(
    as_data_matrix(x),
    n_clusters = n_clusters,
    n_resamples = n_resamples,
    subsample_ratio = subsample_ratio,
    estimator_param_grids = estimator_param_grids,
    classifier = classifier,
    n_trees = n_trees,
    reference_labels = reference_labels,
    mode = mode,
    n_jobs = n_jobs,
    BPPARAM = BPPARAM,
    random_state = random_state,
    show_progress = show_progress,
    verbose = verbose
  )
})

fit_carve <- function(X, n_clusters, n_resamples, subsample_ratio, estimator_param_grids,
                      classifier, n_trees, reference_labels, mode, n_jobs, BPPARAM,
                      random_state, show_progress, verbose) {
  if (!is.numeric(subsample_ratio) || length(subsample_ratio) != 1L ||
      is.na(subsample_ratio) || subsample_ratio <= 0 || subsample_ratio >= 1) {
    stop(sprintf(
      "subsample_ratio must be in (0, 1), got %s. Each resample needs both a training and a held-out split.",
      format(subsample_ratio)
    ), call. = FALSE)
  }
  if (n_resamples < 1) {
    stop(sprintf("n_resamples must be at least 1, got %s.", format(n_resamples)), call. = FALSE)
  }
  if (!is.null(classifier) && !is.function(classifier)) {
    stop("classifier must be NULL or a function.", call. = FALSE)
  }
  if (is.null(classifier) && n_trees < 1) {
    stop(sprintf("n_trees must be at least 1, got %s.", format(n_trees)), call. = FALSE)
  }
  policy <- resolve_mode(mode)
  if (policy$mode != "default") {
    warning("Non-default mode is experimental and may break downstream functionality.", call. = FALSE)
  }
  if (!is.null(classifier) && n_trees != 100) {
    warning("n_trees is ignored when a custom classifier is provided.", call. = FALSE)
  }
  n_resamples <- as.integer(n_resamples)
  seed <- resolve_seed(random_state, n_resamples)
  reference <- coerce_reference_labels(reference_labels, nrow(X))

  # The sweep axis, and the grids along it. Custom grids decide the swept
  # parameter, so a grid sweeping resolution works without resolution=.
  if (is.character(estimator_param_grids) && length(estimator_param_grids) == 1L &&
      estimator_param_grids %in% c("light", "full")) {
    sweep <- resolve_sweep(n_clusters = n_clusters)
    grids <- default_estimator_grids(X, preset = estimator_param_grids, sweep = sweep)
  } else if (is.list(estimator_param_grids) && !is.object(estimator_param_grids) &&
             all(vapply(estimator_param_grids, inherits, logical(1), "carve_estimator_grid"))) {
    grids <- estimator_param_grids
    param <- infer_sweep_param(grids)
    values <- if (is.null(param)) NULL else grid_sweep_values(grids, param)
    sweep <- resolve_sweep(n_clusters = n_clusters, sweep = param, sweep_values = values)
    sweep@values <- validate_grids(grids, sweep)
  } else {
    stop(sprintf(
      "Unknown estimator_param_grids preset %s. Expected 'light', 'full', or a list of estimator_grid() specifications.",
      format_repr(estimator_param_grids)
    ), call. = FALSE)
  }

  print_run_header(X, sweep, n_resamples, subsample_ratio, grids, n_jobs, seed, verbose, policy$mode)
  run <- run_validation(
    X, grids,
    n_resamples = n_resamples,
    subsample_ratio = subsample_ratio,
    classifier = classifier,
    n_trees = as.integer(n_trees),
    n_jobs = n_jobs,
    BPPARAM = BPPARAM,
    random_state = seed,
    sweep = sweep,
    mode = policy$mode,
    show_progress = show_progress,
    verbose = verbose
  )

  results <- run$records
  # config_id is the join key between the table and the per-configuration
  # lists.
  keys <- as.character(results$config_id)
  gini <- NULL
  ce <- NULL
  if (policy$run_stability) {
    summaries <- run$summaries[keys]
    gini <- lapply(summaries, function(s) s$gini)
    ce <- lapply(summaries, function(s) s$ce)
    results$consensus_pac_stability <- unname(vapply(summaries, function(s) s$pac, numeric(1)))
    # A plain mean, as in Python: one sample never drawn with a partner
    # makes the configuration's mean NaN.
    results$consensus_gini_stability <- unname(vapply(gini, mean, numeric(1)))
    results$consensus_ce_stability <- unname(vapply(ce, mean, numeric(1)))
  } else {
    results$consensus_pac_stability <- NaN
    results$consensus_gini_stability <- NaN
    results$consensus_ce_stability <- NaN
  }
  results$accuracy_generalizability <- if (policy$run_generalizability) {
    unname(vapply(run$generalizability_scores[keys], mean, numeric(1)))
  } else {
    NaN
  }

  fit <- methods::new(
    "CARVE",
    input_data = X,
    reference_labels = reference,
    run_params = list(
      n_resamples = n_resamples,
      subsample_ratio = subsample_ratio,
      n_trees = as.integer(n_trees),
      n_jobs = n_jobs,
      mode = policy$mode,
      random_state = seed,
      classifier = classifier
    ),
    estimator_results = results,
    estimator_param_grids = grids,
    sweep = sweep,
    consensus_matrices = run$consensus_matrices,
    consensus_generalizability_matrices = run$consensus_generalizability_matrices,
    stability_gini_scores = gini,
    stability_ce_scores = ce,
    generalizability_scores = run$generalizability_scores
  )
  print_run_footer(results, verbose)
  fit
}

# NULL means 0, as in Python. Every derived seed, up to
# random_state + 2 * n_resamples, must fit in an R integer.
resolve_seed <- function(random_state, n_resamples) {
  if (is.null(random_state)) {
    return(0L)
  }
  if (!is.numeric(random_state) || length(random_state) != 1L || is.na(random_state) ||
      random_state != round(random_state) || random_state < 0) {
    stop("random_state must be NULL or a non-negative whole number.", call. = FALSE)
  }
  limit <- .Machine$integer.max - 2 * n_resamples
  if (random_state > limit) {
    stop(sprintf(
      "random_state must be at most %.0f so that every derived seed fits in an R integer.",
      limit
    ), call. = FALSE)
  }
  as.integer(random_state)
}

# Whole numbers without NA keep their values. Anything else, whole numbers
# with an NA included, is coded 1, 2, ... in order of first appearance, and
# NA becomes -1, a sample without a reference label. Python's pd.factorize
# codes the same cases, since a numpy integer array cannot hold a missing
# value.
coerce_reference_labels <- function(reference_labels, n_samples) {
  if (is.null(reference_labels)) {
    return(NULL)
  }
  if (length(reference_labels) != n_samples) {
    stop(sprintf(
      "reference_labels has %d entries but X has %d rows.",
      length(reference_labels), n_samples
    ), call. = FALSE)
  }
  if (is.numeric(reference_labels) && !anyNA(reference_labels) &&
      all(reference_labels == round(reference_labels))) {
    return(as.integer(reference_labels))
  }
  values <- as.character(reference_labels)
  codes <- match(values, unique(values[!is.na(values)]))
  codes[is.na(values)] <- -1L
  as.integer(codes)
}
