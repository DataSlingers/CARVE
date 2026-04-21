#' Compute a consensus matrix from resampled clustering runs
#'
#' For each pair of samples `(i, j)`, the consensus value is the number of
#' runs in which `i` and `j` were assigned to the same cluster divided by the
#' number of runs in which both `i` and `j` appeared in the subsample. Pairs
#' that never co-occurred in any subsample are `NaN`.
#'
#' @param n_samples Integer, total number of samples in the original dataset.
#' @param runs A list of runs, each a two-element list with components
#'   `sample_idx` (1-based integer vector of indices into the original
#'   dataset) and `labels` (cluster labels for those indices).
#' @param return_counts Logical; if `TRUE`, also return the raw co-cluster
#'   and co-sample count matrices.
#'
#' @return If `return_counts = FALSE`, an `(n_samples, n_samples)` numeric
#'   matrix. Otherwise a list with components `consensus_matrix`,
#'   `co_cluster_counts`, and `co_sample_counts`.
#'
#' @export
compute_consensus_matrix <- function(n_samples, runs, return_counts = FALSE) {
  co_cluster <- matrix(0, n_samples, n_samples)
  co_sample <- matrix(0, n_samples, n_samples)

  for (run in runs) {
    idx <- run$sample_idx
    labs <- run$labels
    co_sample[idx, idx] <- co_sample[idx, idx] + 1

    for (lab in unique(labs)) {
      lidx <- idx[labs == lab]
      co_cluster[lidx, lidx] <- co_cluster[lidx, lidx] + 1
    }
  }

  consensus <- suppressWarnings(co_cluster / co_sample)
  consensus[co_sample == 0] <- NaN

  if (return_counts) {
    list(
      consensus_matrix = consensus,
      co_cluster_counts = co_cluster,
      co_sample_counts = co_sample
    )
  } else {
    consensus
  }
}


#' Reorder a consensus matrix by hierarchical clustering of distances
#'
#' Computes a dendrogram from the distances `1 - consensus` (with NaNs filled
#' for ordering only) using average linkage, and returns the matrix reordered
#' by dendrogram leaves.
#'
#' @param consensus_matrix A square numeric matrix with values in `[0, 1]` and
#'   `NaN` for never-co-sampled pairs.
#' @param fill_nan_for_order Value used to replace NaNs when computing the
#'   clustering order only. Default `0`.
#'
#' @return A list with components `reordered` (the reordered matrix) and
#'   `order` (the permutation applied).
#'
#' @export
reorder_consensus_matrix <- function(consensus_matrix, fill_nan_for_order = 0) {
  m <- consensus_matrix
  m[is.nan(m)] <- fill_nan_for_order
  d <- stats::as.dist(1 - m)
  h <- stats::hclust(d, method = "average")
  ord <- h$order
  list(
    reordered = consensus_matrix[ord, ord, drop = FALSE],
    order = ord
  )
}


#' Compute per-sample stability scores from a consensus matrix
#'
#' Returns Gini-based and cross-entropy-based stability scores per sample,
#' computed from off-diagonal consensus values. Both scores lie in `[0, 1]`;
#' larger is more stable.
#'
#' @param consensus_matrix A square numeric matrix with values in `[0, 1]`.
#'
#' @return A list with numeric vectors `stability_gini` and `stability_ce`,
#'   each of length `n_samples`.
#'
#' @export
stability_from_consensus <- function(consensus_matrix) {
  p <- consensus_matrix
  p <- as.matrix(p)
  storage.mode(p) <- "double"
  diag(p) <- NaN

  term <- p * (1 - p)

  clipped <- pmin(pmax(p, 1e-12), 1 - 1e-12)
  entropy <- -(clipped * log(clipped) + (1 - clipped) * log(1 - clipped))

  # nanmean across rows. rowMeans with na.rm handles NaN correctly.
  uncertainty_gini <- 2 * rowMeans(term, na.rm = TRUE)
  uncertainty_ce <- rowMeans(entropy, na.rm = TRUE)

  stability_gini <- 1 - pmin(pmax(2 * uncertainty_gini, 0), 1)
  stability_ce <- 1 - pmin(pmax(uncertainty_ce / log(2), 0), 1)

  list(stability_gini = stability_gini, stability_ce = stability_ce)
}


#' Compute a PAC-based stability score from a consensus matrix
#'
#' PAC (Proportion of Ambiguous Clustering) is the fraction of off-diagonal,
#' non-`NaN` consensus values that fall strictly in `(tau, 1 - tau)`. This
#' function returns `1 - PAC` so that larger values indicate greater stability.
#'
#' @param consensus_matrix A square numeric matrix with values in `[0, 1]`
#'   and `NaN` for never-co-sampled pairs.
#' @param tau Ambiguity threshold. Default `0.05`.
#'
#' @return A single numeric value in `[0, 1]`, or `NaN` if no off-diagonal
#'   pairs have a valid consensus value.
#'
#' @export
compute_consensus_pac <- function(consensus_matrix, tau = 0.05) {
  m <- as.matrix(consensus_matrix)
  storage.mode(m) <- "double"
  n <- nrow(m)
  mask <- !diag(TRUE, n)
  values <- m[mask]
  values <- values[!is.nan(values) & !is.na(values)]

  if (length(values) == 0L) {
    return(NaN)
  }

  ambiguous <- sum(values > tau & values < (1 - tau))
  pac <- ambiguous / length(values)
  1 - pac
}


#' Compute stability metrics for multiple consensus matrices
#'
#' Convenience wrapper that applies [stability_from_consensus()] and
#' [compute_consensus_pac()] to each consensus matrix in a list.
#'
#' @param consensus_matrices A list of consensus matrices.
#'
#' @return A list with components:
#'   \describe{
#'     \item{gini_list}{List of per-sample Gini stability vectors.}
#'     \item{ce_list}{List of per-sample cross-entropy stability vectors.}
#'     \item{pac_list}{Numeric vector of PAC-based (`1 - PAC`) scores.}
#'   }
#'
#' @export
compute_consensus_metrics <- function(consensus_matrices) {
  gini_list <- vector("list", length(consensus_matrices))
  ce_list <- vector("list", length(consensus_matrices))
  pac_list <- numeric(length(consensus_matrices))

  for (i in seq_along(consensus_matrices)) {
    s <- stability_from_consensus(consensus_matrices[[i]])
    gini_list[[i]] <- s$stability_gini
    ce_list[[i]] <- s$stability_ce
    pac_list[i] <- compute_consensus_pac(consensus_matrices[[i]])
  }

  list(gini_list = gini_list, ce_list = ce_list, pac_list = pac_list)
}
