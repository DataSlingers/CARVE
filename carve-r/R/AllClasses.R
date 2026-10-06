# S4 classes of the package.

setClassUnion("carveListOrNULL", c("list", "NULL"))
setClassUnion("carveIntegerOrNULL", c("integer", "NULL"))

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
