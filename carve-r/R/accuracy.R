# Per-sample held-out accuracy. Mirrors _accuracy.py.

# For every held-out sample, the share of the runs that held it out in which
# the classifier, after its labels were aligned to the held-out clustering,
# predicted the sample's cluster. A sample never held out scores 0.
compute_generalizability_scores <- function(n_samples, runs) {
  correct <- numeric(n_samples)
  total <- numeric(n_samples)
  for (run in runs) {
    aligned <- align_cluster_labels(run$true, run$predicted)
    correct[run$indices] <- correct[run$indices] + (run$true == aligned)
    total[run$indices] <- total[run$indices] + 1
  }
  out <- numeric(n_samples)
  held_out <- total > 0
  out[held_out] <- correct[held_out] / total[held_out]
  out
}
