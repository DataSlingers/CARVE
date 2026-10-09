#' CARVE: cluster analysis with resampling for validation and exploration
#'
#' CARVE chooses the number of clusters, and the clustering method, by
#' resampling. [carve()] clusters many subsamples of the data with every
#' candidate configuration and scores each one for stability and
#' generalizability. [get_k()] and [get_labels()] then pick a configuration
#' with a selection rule.
#'
#' Start with `vignette("CARVE", package = "CARVE")`. The other vignettes
#' cover single-cell data (`"single-cell"`), the options of a run
#' (`"customizing"`) and the differences from the Python package
#' (`"python-users"`).
#'
#' @keywords internal
#' @importFrom methods new setClass setClassUnion setGeneric setMethod
#' @importFrom methods setValidity show validObject
#' @importFrom ggplot2 .data
"_PACKAGE"
