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
#' A run sweeps one parameter. By default it is `n_clusters`. Giving
#' `resolution` sweeps the resolution of [LeidenClustering()] and
#' [LouvainClustering()], and `sweep = "min_cluster_size"` with
#' `sweep_values` sweeps the minimum cluster size of [HDBSCAN()]. Off the
#' `n_clusters` axis the number of clusters is an outcome, recorded for each
#' configuration as its mean over the resamples.
#'
#' Resample `b`, counting from 0, draws its first subsample with seed
#' `random_state + b` and its second with `random_state + b + n_resamples`,
#' and its estimator and classifier use `random_state + b`. The results are
#' therefore the same for every `n_jobs` and `BPPARAM`, and the session's
#' random number stream is left as it was.
#'
#' The `n_neighbors` of [SpectralClustering()], [LeidenClustering()] and
#' [LouvainClustering()] is a count at the density of the full data. Each
#' subsample is clustered with the count scaled by its share of the samples,
#' rounded and at least 2, so that it covers the same neighborhood. The
#' neighbor settings of preprocessing transforms are not scaled.
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
#' Above `anchor_threshold` samples the run is anchored, because that memory
#' grows with the square of `n`: it draws `m` anchor samples once and keeps only
#' the `m` by `m` blocks over them. Per-sample stability is computed against
#' the anchors, so every sample still gets a score, and [get_labels()] cuts
#' the anchor block and labels the other samples with the classifier,
#' trained on the anchors. PAC then covers anchor pairs only and cannot be
#' compared with the PAC of an exact run. A warning says when a run is
#' anchored, and [consensus_anchors()] returns the anchors.
#'
#' With `randomize_preprocessing = TRUE`, each resample draws a pipeline: one
#' normalization option and one dimensionality reduction option, with one
#' value drawn for each of their arguments. Every pair of options gets
#' `n_resamples` divided by the number of pairs, rounded up or down, in a
#' seeded random order. The pipeline is fit separately on each of the
#' resample's subsamples, with seeds `random_state + b`,
#' `random_state + b + n_resamples` and `random_state + b + 2 * n_resamples`,
#' so an embedding that does not reproduce across fits lowers both criteria.
#' Each resample is embedded once, before the configurations run, and the
#' embeddings stay in memory during the fit. The classifier trains on the
#' raw features of the first subsample with the labels clustered from its
#' embedding, so a cluster that exists only in the embedding does not
#' generalize, and t-SNE, which cannot embed new samples, can take part.
#' [preprocessing_results()] splits the scores by pipeline.
#'
#' @param x The data: a numeric matrix with one row per sample, a numeric
#'   data frame, a sparse matrix from the Matrix package, a numeric vector,
#'   or a SingleCellExperiment or Seurat object, whose cells are its columns.
#' @param ... For a SingleCellExperiment, the arguments of the matrix
#'   method, from `n_clusters` on. For a Seurat object, `assay`, `reduction`
#'   and `n_dims`. For other input, not used: an argument name `carve()`
#'   does not know is an error.
#' @param n_clusters Numbers of clusters to evaluate. A single number `K`
#'   means `2:K`. Custom grids set their own values.
#' @param resolution Resolutions to evaluate with the graph estimators.
#'   Giving it makes `resolution` the swept parameter.
#' @param sweep Name of the swept parameter. Defaults to `"resolution"` when
#'   `resolution` is given and to `"n_clusters"` otherwise; custom grids set
#'   it themselves. Estimators that take a number of clusters and estimators
#'   that take a resolution cannot share a run.
#' @param sweep_values Values of `sweep` when it is neither `"n_clusters"`
#'   nor `"resolution"`, such as minimum cluster sizes for [HDBSCAN()].
#' @param finer_is_larger Whether larger values of `sweep` give more
#'   clusters. It is known for `n_clusters` and `resolution` (`TRUE`) and for
#'   `min_cluster_size` (`FALSE`), and required for any other parameter.
#' @param noise_policy What to do with the -1 labels of density-based
#'   methods such as [HDBSCAN()]. `"drop"` leaves those samples out of the
#'   resample, so the consensus matrix, the ARI and the classifier treat them
#'   as not drawn. `"as_cluster"` keeps -1 as an ordinary label, and
#'   `"singleton"` puts each of them in a cluster of its own.
#' @param n_resamples Resamples per configuration.
#' @param subsample_ratio Share of the samples drawn, without replacement,
#'   into each subsample. Must lie strictly between 0 and 1.
#' @param anchor_threshold Runs with at most this many samples build full
#'   consensus matrices; larger runs are anchored. See Details.
#' @param consensus_anchors Number of anchors, or a share of the samples in
#'   (0, 1]. `NULL` means `min(nrow(x), anchor_threshold)`. A setting that
#'   gives fewer anchors than samples anchors the run whatever its size; one
#'   that gives every sample, such as 1 or a count of at least `nrow(x)`,
#'   leaves the run exact. Lowering it shrinks the blocks, `8 * m^2` bytes
#'   each.
#' @param estimator_param_grids `"light"`, `"full"`, or a list of
#'   [estimator_grid()] specifications. On the `n_clusters` axis `"light"`
#'   runs KMeans, Ward-linkage agglomerative and self-tuning spectral
#'   clustering, and `"full"` adds average, single and complete linkage and
#'   RBF spectral clustering. On the `resolution` axis `"light"` runs Leiden
#'   and Louvain on 15-neighbor graphs, and `"full"` adds 10- and 30-neighbor
#'   graphs. On the `min_cluster_size` axis `"light"` runs HDBSCAN with
#'   excess-of-mass selection, and `"full"` adds leaf selection. All grids
#'   must sweep the same parameter over the same values.
#' @param normalization_options,dim_reduction_options Lists of
#'   [preprocessing_option()] objects for randomized preprocessing. `NULL`
#'   uses the defaults: identity, standardization and, for data without
#'   negative values, log1p; identity, PCA, t-SNE and, when uwot is
#'   installed, UMAP, over the values the smallest subsample supports. An
#'   empty `normalization_options` list also means the defaults, but an empty
#'   `dim_reduction_options` list is an error. Not used unless
#'   `randomize_preprocessing = TRUE`.
#' @param randomize_preprocessing Draw a preprocessing pipeline for each
#'   resample. See Details.
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
#' @param show_progress Show a progress bar over the configurations and,
#'   under randomized preprocessing, one before it over the embedding pass.
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
#'
#' graph_fit <- carve(X, resolution = c(0.25, 0.5, 1), n_resamples = 10, random_state = 0)
#' estimator_results(graph_fit)[, c("method_label", "resolution", "n_clusters_observed")]
#' @rdname carve
#' @export
setMethod("carve", "ANY", function(x, n_clusters = 2:10, resolution = NULL, sweep = NULL,
                                   sweep_values = NULL, finer_is_larger = NULL,
                                   noise_policy = "drop", n_resamples = 100,
                                   subsample_ratio = 0.618, anchor_threshold = 5000,
                                   consensus_anchors = NULL, estimator_param_grids = "light",
                                   normalization_options = NULL, dim_reduction_options = NULL,
                                   randomize_preprocessing = FALSE, classifier = NULL,
                                   n_trees = 100, reference_labels = NULL, mode = "default",
                                   n_jobs = 1, BPPARAM = NULL, random_state = NULL,
                                   show_progress = FALSE, verbose = 0, ...) {
  # A Seurat object is an S4 class of a suggested package, so no method can
  # name it; it arrives here, with assay, reduction and n_dims in ... .
  if (is_seurat(x)) {
    selection <- seurat_selection(...)
    data <- seurat_matrix(
      x,
      assay = selection$assay,
      reduction = selection$reduction,
      n_dims = selection$n_dims
    )
  } else {
    check_dots(...)
    data <- as_data_matrix(x)
  }
  fit_carve(
    data,
    n_clusters = n_clusters,
    resolution = resolution,
    sweep = sweep,
    sweep_values = sweep_values,
    finer_is_larger = finer_is_larger,
    noise_policy = noise_policy,
    n_resamples = n_resamples,
    subsample_ratio = subsample_ratio,
    anchor_threshold = anchor_threshold,
    consensus_anchors = consensus_anchors,
    estimator_param_grids = estimator_param_grids,
    normalization_options = normalization_options,
    dim_reduction_options = dim_reduction_options,
    randomize_preprocessing = randomize_preprocessing,
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
fit_carve <- function(X, n_clusters, resolution, sweep, sweep_values, finer_is_larger,
                      noise_policy, n_resamples, subsample_ratio, anchor_threshold,
                      consensus_anchors, estimator_param_grids, normalization_options,
                      dim_reduction_options, randomize_preprocessing, classifier, n_trees,
                      reference_labels, mode, n_jobs, BPPARAM, random_state, show_progress,
                      verbose) {
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
  # Python checks the noise policy only inside the first resample.
  check_noise_policy(noise_policy)
  if (!is.numeric(anchor_threshold) || length(anchor_threshold) != 1L || is.na(anchor_threshold) ||
      anchor_threshold < 0 || anchor_threshold != round(anchor_threshold)) {
    stop("anchor_threshold must be a non-negative whole number.", call. = FALSE)
  }
  if (!isTRUE(randomize_preprocessing) && !isFALSE(randomize_preprocessing)) {
    stop("randomize_preprocessing must be TRUE or FALSE.", call. = FALSE)
  }
  check_preprocessing_options(normalization_options, "normalization_options")
  check_preprocessing_options(dim_reduction_options, "dim_reduction_options")
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

  anchors <- resolve_anchors(nrow(X), consensus_anchors, anchor_threshold, seed)
  if (!is.null(anchors)) {
    # resolve_anchors() ignores anchor_threshold once consensus_anchors is
    # set, so the message names the setting that applied.
    reason <- if (is.null(consensus_anchors)) {
      sprintf("n=%d exceeds anchor_threshold=%s", nrow(X), format_param_value(anchor_threshold))
    } else {
      sprintf(
        "consensus_anchors=%s opts this run in regardless of anchor_threshold",
        format_repr(consensus_anchors)
      )
    }
    warning(sprintf(
      "%s, so CARVE is using anchored consensus over %d anchors. Per-sample scores and labels still cover every sample; consensus matrices and PAC are computed over the anchors.",
      reason, length(anchors)
    ), call. = FALSE)
  }

  # The sweep axis, and the grids along it. Custom grids decide the swept
  # parameter, so a grid sweeping resolution works without resolution=.
  is_preset <- is.character(estimator_param_grids) && length(estimator_param_grids) == 1L &&
    estimator_param_grids %in% c("light", "full")
  is_custom <- is.list(estimator_param_grids) && !is.object(estimator_param_grids) &&
    all(vapply(estimator_param_grids, inherits, logical(1), "carve_estimator_grid"))
  if (!is_preset && !is_custom) {
    stop(sprintf(
      "Unknown estimator_param_grids preset %s. Expected 'light', 'full', or a list of estimator_grid() specifications.",
      format_repr(estimator_param_grids)
    ), call. = FALSE)
  }
  swept <- sweep
  values <- sweep_values
  if (is_custom) {
    if (is.null(swept) && is.null(resolution)) {
      swept <- infer_sweep_param(estimator_param_grids)
    }
    if (is.null(values) && !is.null(swept)) {
      values <- grid_sweep_values(estimator_param_grids, swept)
    }
  }
  axis <- resolve_sweep(
    n_clusters = n_clusters,
    resolution = resolution,
    sweep = swept,
    sweep_values = values,
    finer_is_larger = finer_is_larger
  )
  if (is_custom) {
    grids <- estimator_param_grids
    axis@values <- validate_grids(grids, axis)
  } else {
    grids <- default_estimator_grids(X, preset = estimator_param_grids, sweep = axis)
  }

  # Only a randomized fit uses the preprocessing options. Resolving the
  # defaults otherwise would warn about log1p on any data with negative
  # values. An empty normalization list means the defaults, as in Python;
  # an empty reduction list is kept and rejected by the allocation.
  if (randomize_preprocessing) {
    norm_options <- if (length(normalization_options) > 0L) {
      normalization_options
    } else {
      default_normalization_options(X)
    }
    dr_options <- if (!is.null(dim_reduction_options)) {
      dim_reduction_options
    } else {
      default_dim_reduction_options(X, subsample_ratio)
    }
  } else {
    norm_options <- if (is.null(normalization_options)) list() else normalization_options
    dr_options <- if (is.null(dim_reduction_options)) list() else dim_reduction_options
  }

  print_run_header(
    X, axis, n_resamples, subsample_ratio, grids, n_jobs, seed, verbose, policy$mode,
    anchors = anchors,
    randomize_preprocessing = randomize_preprocessing,
    normalization_options = norm_options,
    dim_reduction_options = dr_options
  )
  run <- run_validation(
    X, grids,
    n_resamples = n_resamples,
    subsample_ratio = subsample_ratio,
    classifier = classifier,
    n_trees = as.integer(n_trees),
    n_jobs = n_jobs,
    BPPARAM = BPPARAM,
    random_state = seed,
    sweep = axis,
    noise_policy = noise_policy,
    mode = policy$mode,
    anchors = anchors,
    randomize_preprocessing = randomize_preprocessing,
    normalization_options = norm_options,
    dim_reduction_options = dr_options,
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

  preprocessing <- NULL
  if (randomize_preprocessing) {
    preprocessing <- summarize_preprocessing_records(run$pipeline_records, run$pipelines, axis@param)
  }
  # The classifier that extends an anchored cut in get_labels() is one fit
  # outside the resample loop, so it gets the run's whole budget.
  budget <- run_core_budget(n_jobs, n_resamples, BPPARAM)

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
      classifier = classifier,
      noise_policy = noise_policy,
      anchor_threshold = anchor_threshold,
      consensus_anchors = consensus_anchors,
      randomize_preprocessing = randomize_preprocessing,
      n_threads = as.integer(min(n_cores(), budget[["outer"]] * budget[["inner"]]))
    ),
    estimator_results = results,
    estimator_param_grids = grids,
    sweep = axis,
    consensus_matrices = run$consensus_matrices,
    consensus_generalizability_matrices = run$consensus_generalizability_matrices,
    stability_gini_scores = gini,
    stability_ce_scores = ce,
    generalizability_scores = run$generalizability_scores,
    consensus_anchors = anchors,
    preprocessing_results = preprocessing,
    preprocessing_pipelines = if (randomize_preprocessing) run$pipelines else NULL
  )
  print_run_footer(results, verbose)
  fit
}

check_preprocessing_options <- function(options, argument) {
  valid <- is.null(options) || (is.list(options) && !is.object(options) &&
    all(vapply(options, inherits, logical(1), "carve_preprocessing_option")))
  if (!valid) {
    stop(sprintf(
      "%s must be NULL or a list of preprocessing_option() specifications.",
      argument
    ), call. = FALSE)
  }
  invisible(options)
}

# NULL means 0, as in Python. Every derived seed, up to
# random_state + 3 * n_resamples - 1 (the held-out embedding of the last
# resample under randomized preprocessing), must fit in an R integer.
resolve_seed <- function(random_state, n_resamples) {
  if (is.null(random_state)) {
    return(0L)
  }
  if (!is.numeric(random_state) || length(random_state) != 1L || is.na(random_state) ||
      random_state != round(random_state) || random_state < 0) {
    stop("random_state must be NULL or a non-negative whole number.", call. = FALSE)
  }
  limit <- .Machine$integer.max - 3 * n_resamples
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
