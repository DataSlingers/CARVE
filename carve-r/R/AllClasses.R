# S4 classes of the package.

setClassUnion("carveListOrNULL", c("list", "NULL"))
setClassUnion("carveIntegerOrNULL", c("integer", "NULL"))
setClassUnion("carveDataFrameOrNULL", c("data.frame", "NULL"))

#' Sweep axis of a CARVE run
#'
#' Records the hyperparameter a [carve()] run swept, the values it tried and
#' whether larger values give more clusters. [sweep_spec()] returns it.
#'
#' @slot param Name of the swept parameter, such as `"n_clusters"`.
#' @slot values The values tried.
#' @slot finer_is_larger `TRUE` when larger values give more clusters.
#' @slot fixes_k `TRUE` when the parameter sets the number of clusters
#'   directly, which only `n_clusters` does.
#' @slot label Axis label for plots.
#' @export
setClass(
  "SweepSpec",
  slots = c(
    param = "character",
    values = "numeric",
    finer_is_larger = "logical",
    fixes_k = "logical",
    label = "character"
  )
)

#' Fitted CARVE object
#'
#' [carve()] returns an object of this class. Read it with the accessor
#' functions; the slots are listed here for reference.
#'
#' @slot input_data The data matrix of the fit, one row per sample.
#' @slot reference_labels Reference labels from `carve(reference_labels = )`,
#'   coded as integers, or `NULL`.
#' @slot run_params List of the run settings: `n_resamples`,
#'   `subsample_ratio`, `n_trees`, `n_jobs`, `mode`, `random_state` (the seed
#'   the run used), `classifier`, `noise_policy`, `anchor_threshold`,
#'   `consensus_anchors`, `randomize_preprocessing`, `estimator_param_grids`
#'   (`"light"`, `"full"` or `"custom"`), and `n_threads`, the threads of the
#'   classifier [get_labels()] trains on an anchored run.
#' @slot estimator_results Data frame with one row per configuration; see
#'   [estimator_results()].
#' @slot estimator_param_grids The estimator grids the run evaluated.
#' @slot sweep The run's [SweepSpec-class] object.
#' @slot consensus_matrices Stability consensus matrices, a list named by
#'   `config_id`. Entries are `NULL` when the run skipped stability. On an
#'   anchored run each is the block over the anchors.
#' @slot consensus_generalizability_matrices Consensus matrices of the
#'   classifier's held-out predictions, a list named by `config_id`. Entries
#'   are `NULL` when the run skipped generalizability.
#' @slot stability_gini_scores Per-sample Gini stability, a list named by
#'   `config_id`, or `NULL` when the run skipped stability.
#' @slot stability_ce_scores Per-sample cross-entropy stability, a list named
#'   by `config_id`, or `NULL` when the run skipped stability.
#' @slot generalizability_scores Per-sample held-out accuracy, a list named by
#'   `config_id`. Entries are `NULL` when the run skipped generalizability.
#' @slot consensus_anchors Sorted row indices of the anchors of an anchored
#'   run, or `NULL` for an exact run.
#' @slot preprocessing_results Scores split by preprocessing pipeline, or
#'   `NULL` for a run without randomized preprocessing.
#' @slot preprocessing_pipelines The pipelines of a randomized run, named by
#'   their labels, or `NULL`.
#' @seealso [carve()], [estimator_results()], [get_labels()]
#' @export
setClass(
  "CARVE",
  slots = c(
    input_data = "matrix",
    reference_labels = "carveIntegerOrNULL",
    run_params = "list",
    estimator_results = "data.frame",
    estimator_param_grids = "list",
    sweep = "SweepSpec",
    consensus_matrices = "list",
    consensus_generalizability_matrices = "list",
    stability_gini_scores = "carveListOrNULL",
    stability_ce_scores = "carveListOrNULL",
    generalizability_scores = "list",
    consensus_anchors = "carveIntegerOrNULL",
    preprocessing_results = "carveDataFrameOrNULL",
    preprocessing_pipelines = "carveListOrNULL"
  )
)

# config_id joins a results row to its matrices and scores. It must count
# from 0 in table order, and every container must be named by it.
setValidity("CARVE", function(object) {
  expected <- seq_len(nrow(object@estimator_results)) - 1L
  containers <- list(
    object@consensus_matrices,
    object@consensus_generalizability_matrices,
    object@generalizability_scores,
    object@stability_gini_scores,
    object@stability_ce_scores
  )
  named_by_id <- vapply(
    containers,
    function(x) is.null(x) || identical(as.character(names(x)), as.character(expected)),
    logical(1)
  )
  if (identical(as.integer(object@estimator_results$config_id), expected) && all(named_by_id)) {
    return(TRUE)
  }
  "config_id is misaligned with the per-configuration artifact containers. This is an internal CARVE error."
})
