# Accuracy-based generalizability scoring. Mirrors _accuracy.py.


#' Per-sample generalizability scores
#'
#' Mirrors `_accuracy.compute_generalizability_scores`. For each sample,
#' accumulates prediction-correct counts across runs (after Hungarian
#' alignment to the true labels) and returns `correct / total`. Samples
#' that were never evaluated receive `0`.
#'
#' @param n_samples Integer; total number of samples in the original dataset.
#' @param runs List of runs. Each run is a list with components
#'   `sample_idx` (1-based integer indices), `labels` (true labels), and
#'   `predicted` (predicted labels).
#'
#' @return Numeric vector of length `n_samples`.
#'
#' @export
compute_generalizability_scores <- function(n_samples, runs) {
  correct <- integer(n_samples)
  total <- integer(n_samples)

  for (r in runs) {
    aligned <- align_cluster_labels(r$labels, r$predicted)
    correct_mask <- as.integer(r$labels == aligned)
    idx <- r$sample_idx
    correct[idx] <- correct[idx] + correct_mask
    total[idx] <- total[idx] + 1L
  }

  out <- numeric(n_samples)
  valid <- total > 0L
  out[valid] <- correct[valid] / total[valid]
  out
}
