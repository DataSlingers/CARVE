#' CARVE: cluster analysis with resampling for validation and exploration
#'
#' CARVE chooses the number of clusters, and the clustering method, by
#' resampling. [carve()] clusters many subsamples of the data with every
#' candidate configuration and scores each one for stability and
#' generalizability. [get_k()] and [get_labels()] then pick a configuration
#' with a selection rule.
#'
#' @keywords internal
#' @importFrom methods new setClass setClassUnion setGeneric setMethod
#'   setValidity show validObject
"_PACKAGE"
