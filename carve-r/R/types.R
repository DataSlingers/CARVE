#' Resolve a run-mode string into an execution policy
#'
#' Mirrors `carve._types.resolve_mode` in the Python package. The returned
#' list controls which branches of the resampling loop are executed.
#'
#' @param mode One of `"default"`, `"stability"`, or `"generalizability"`.
#'
#' @return A named list with class `"ModePolicy"` and fields:
#'   \describe{
#'     \item{mode}{The input mode string.}
#'     \item{run_stability}{Logical; whether to run the stability branch.}
#'     \item{run_generalizability}{Logical; whether to run the generalizability branch.}
#'     \item{compute_average_ari}{Logical; whether to compute mean(stability, generalizability) ARI.}
#'   }
#'
#' @export
resolve_mode <- function(mode = c("default", "stability", "generalizability")) {
  mode <- match.arg(mode)
  out <- switch(
    mode,
    "default" = list(
      mode = "default",
      run_stability = TRUE,
      run_generalizability = TRUE,
      compute_average_ari = TRUE
    ),
    "stability" = list(
      mode = "stability",
      run_stability = TRUE,
      run_generalizability = FALSE,
      compute_average_ari = FALSE
    ),
    "generalizability" = list(
      mode = "generalizability",
      run_stability = FALSE,
      run_generalizability = TRUE,
      compute_average_ari = FALSE
    )
  )
  class(out) <- "ModePolicy"
  out
}
