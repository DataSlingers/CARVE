#' @include AllClasses.R
NULL

#' @rdname carve
#' @export
setGeneric("carve", function(x, ...) standardGeneric("carve"))

#' @rdname get_k
#' @export
setGeneric("get_k", function(fit, ...) standardGeneric("get_k"))

#' @rdname get_k
#' @export
setGeneric("get_sweep_value", function(fit, ...) standardGeneric("get_sweep_value"))

#' @rdname get_k
#' @export
setGeneric("get_estimator", function(fit, ...) standardGeneric("get_estimator"))

#' @rdname get_labels
#' @export
setGeneric("get_labels", function(fit, ...) standardGeneric("get_labels"))

#' @rdname estimator_results
#' @export
setGeneric("estimator_results", function(fit, ...) standardGeneric("estimator_results"))

#' @rdname estimator_results
#' @export
setGeneric("estimator_param_grids", function(fit, ...) standardGeneric("estimator_param_grids"))

#' @rdname estimator_results
#' @export
setGeneric("sweep_spec", function(fit, ...) standardGeneric("sweep_spec"))

#' @rdname estimator_results
#' @export
setGeneric("input_data", function(fit, ...) standardGeneric("input_data"))

#' @rdname consensus_matrix
#' @export
setGeneric("consensus_matrix", function(fit, ...) standardGeneric("consensus_matrix"))

#' @rdname consensus_matrix
#' @export
setGeneric("sample_scores", function(fit, ...) standardGeneric("sample_scores"))

#' @rdname consensus_matrix
#' @export
setGeneric("consensus_anchors", function(fit, ...) standardGeneric("consensus_anchors"))

#' @rdname preprocessing_results
#' @export
setGeneric("preprocessing_results", function(fit, ...) standardGeneric("preprocessing_results"))

#' @rdname preprocessing_results
#' @export
setGeneric("preprocessing_pipelines", function(fit, ...) standardGeneric("preprocessing_pipelines"))

#' @rdname plot_metric_over_n_clusters
#' @export
setGeneric("plot_metric_over_n_clusters", function(object, ...) {
  standardGeneric("plot_metric_over_n_clusters")
})

#' @rdname plot_metric_by_pipeline
#' @export
setGeneric("plot_metric_by_pipeline", function(object, ...) standardGeneric("plot_metric_by_pipeline"))

#' @rdname plot_n_clusters_over_sweep
#' @export
setGeneric("plot_n_clusters_over_sweep", function(object, ...) {
  standardGeneric("plot_n_clusters_over_sweep")
})

#' @rdname plot_consensus_matrix
#' @export
setGeneric("plot_consensus_matrix", function(object, ...) standardGeneric("plot_consensus_matrix"))
