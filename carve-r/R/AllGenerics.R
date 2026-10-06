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
