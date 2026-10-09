# Consensus matrices and the stability scores derived from them. Mirrors
# _consensus.py, the exact path and the anchored one.

# Entry (i, j) is the share of the runs that drew both i and j in which they
# landed in the same cluster; NaN when no run drew both. S marks which runs
# drew each sample and B which run-cluster each sample fell in, so the two
# counts are tcrossprod(S) and tcrossprod(B).
compute_consensus_matrix <- function(n_samples, runs) {
  n_labels <- vapply(runs, function(r) length(unique(r$labels)), integer(1))
  S <- matrix(0, n_samples, length(runs))
  B <- matrix(0, n_samples, sum(n_labels))
  col <- 0L
  for (r in seq_along(runs)) {
    indices <- runs[[r]]$indices
    labels <- runs[[r]]$labels
    S[indices, r] <- 1
    for (label in unique(labels)) {
      col <- col + 1L
      B[indices[labels == label], col] <- 1
    }
  }
  co_sampled <- tcrossprod(S)
  co_clustered <- tcrossprod(B)
  consensus <- co_clustered / co_sampled
  consensus[co_sampled == 0] <- NaN
  consensus
}

# Per-sample stability from the off-diagonal consensus values of each row.
stability_from_consensus <- function(consensus_matrix) {
  p <- consensus_matrix
  diag(p) <- NaN
  row_stability(p)
}

# Gini uncertainty p(1 - p) and binary entropy averaged over each row's
# entries that are not NaN, and rescaled so that 1 means every pair always
# agreed. A row without such an entry scores NaN.
row_stability <- function(p) {
  gini_term <- p * (1 - p)
  clipped <- pmin(pmax(p, 1e-12), 1 - 1e-12)
  entropy <- -(clipped * log(clipped) + (1 - clipped) * log(1 - clipped))
  uncertainty_gini <- 2 * rowMeans(gini_term, na.rm = TRUE)
  uncertainty_ce <- rowMeans(entropy, na.rm = TRUE)
  list(
    gini = 1 - pmin(pmax(2 * uncertainty_gini, 0), 1),
    ce = 1 - pmin(pmax(uncertainty_ce / log(2), 0), 1)
  )
}

# 1 minus the proportion of ambiguous consensus values, those strictly
# between tau and 1 - tau, over the off-diagonal pairs drawn together.
compute_consensus_pac <- function(consensus_matrix, tau = 0.05) {
  off_diagonal <- row(consensus_matrix) != col(consensus_matrix)
  values <- consensus_matrix[off_diagonal]
  values <- values[!is.na(values)]
  if (length(values) == 0L) {
    return(NaN)
  }
  1 - mean(values > tau & values < 1 - tau)
}

# Sparse indicator factors of the runs: S[i, r] is 1 when run r drew sample
# i, and B[i, c] is 1 when sample i fell in run-cluster c. The consensus
# matrix is tcrossprod(B) / tcrossprod(S); the anchored functions use only
# the rows they need.
run_indicators <- function(n_samples, runs) {
  indices <- lapply(runs, function(r) r$indices)
  codes <- lapply(runs, function(r) match(r$labels, unique(r$labels)))
  n_clusters <- vapply(codes, function(code) length(unique(code)), integer(1))
  offsets <- cumsum(c(0L, n_clusters))[seq_along(runs)]
  # With no runs, possible when every resample was all noise, unlist()
  # returns NULL. Matrix 1.7.5 accepts i = NULL; as.integer() passes
  # integer(0) instead, so the call does not rely on that.
  rows <- as.integer(unlist(indices))
  S <- Matrix::sparseMatrix(
    i = rows,
    j = rep(seq_along(runs), lengths(indices)),
    x = 1,
    dims = c(n_samples, length(runs))
  )
  B <- Matrix::sparseMatrix(
    i = rows,
    j = as.integer(unlist(Map(`+`, codes, offsets))),
    x = 1,
    dims = c(n_samples, sum(n_clusters))
  )
  list(S = S, B = B)
}

# The consensus matrix restricted to anchor-by-anchor pairs. Anchoring
# changes which pairs are computed, not any pair's value.
consensus_anchor_block <- function(n_samples, runs, anchors) {
  factors <- run_indicators(n_samples, runs)
  Sa <- as.matrix(factors$S[anchors, , drop = FALSE])
  Ba <- as.matrix(factors$B[anchors, , drop = FALSE])
  co_sampled <- tcrossprod(Sa)
  block <- tcrossprod(Ba) / co_sampled
  block[co_sampled == 0] <- NaN
  unname(block)
}

# Rows per chunk of stability_from_runs_anchored(). Each chunk holds a few
# chunk-by-m double matrices at once; about 2^23 entries each keeps them
# near 64 MB whatever the anchor count.
default_anchor_chunk_size <- function(m) {
  max(256L, min(8192L, as.integer(2^23 %/% max(m, 1))))
}

# Per-sample Gini and cross-entropy stability against the anchors: the row
# statistic of stability_from_consensus() over the m anchor partners instead
# of all n - 1. Rows are processed in chunks, so the n by m slab never
# exists in full, and the result does not depend on the chunk size.
stability_from_runs_anchored <- function(n_samples, runs, anchors, chunk_size = NULL) {
  factors <- run_indicators(n_samples, runs)
  Sa_t <- t(as.matrix(factors$S[anchors, , drop = FALSE]))
  Ba_t <- t(as.matrix(factors$B[anchors, , drop = FALSE]))
  position <- integer(n_samples)
  position[anchors] <- seq_along(anchors)
  if (is.null(chunk_size)) {
    chunk_size <- default_anchor_chunk_size(length(anchors))
  }
  gini <- numeric(n_samples)
  ce <- numeric(n_samples)
  for (start in seq.int(1L, n_samples, by = chunk_size)) {
    rows <- start:min(start + chunk_size - 1L, n_samples)
    co_sampled <- as.matrix(factors$S[rows, , drop = FALSE] %*% Sa_t)
    probs <- as.matrix(factors$B[rows, , drop = FALSE] %*% Ba_t) / co_sampled
    probs[co_sampled == 0] <- NaN
    # An anchor does not count its own pair, as the diagonal is left out of
    # stability_from_consensus().
    own <- position[rows]
    self <- which(own > 0L)
    probs[cbind(self, own[self])] <- NaN
    scores <- row_stability(probs)
    gini[rows] <- scores$gini
    ce[rows] <- scores$ce
  }
  list(gini = gini, ce = ce)
}
