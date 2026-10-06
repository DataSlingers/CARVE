# The resampling loop. Mirrors _runner.py: a serial loop over
# configurations with the resamples inside it, run in this process or
# through BiocParallel. Every resample derives its own seeds from
# random_state, so the results do not depend on the backend.

# The only place the subsample seeds are derived.
resample_indices <- function(n_samples, seed, n_resamples, subsample_ratio,
                             random_state, run_stability) {
  first <- split_subsample_indices(n_samples, subsample_ratio, random_state + seed)
  second <- NULL
  if (run_stability) {
    second <- split_subsample_indices(n_samples, subsample_ratio, random_state + seed + n_resamples)$train
  }
  list(train = first$train, test = first$test, stability = second)
}

validation_iter <- function(X, estimator, estimator_name, params, subsample_ratio,
                            n_resamples, seed, classifier = NULL, n_trees = 100L,
                            sweep_param = "n_clusters", mode = "default",
                            random_state = 0L, classifier_threads = 1L) {
  policy <- resolve_mode(mode)
  n <- nrow(X)
  idx <- resample_indices(n, seed, n_resamples, subsample_ratio, random_state, policy$run_stability)

  # Each subsample is clustered on its own, with the neighbor count scaled
  # to its own rows.
  fit_labels <- function(rows) {
    scaled <- scale_neighbor_count(estimator, params, n_fit = length(rows), n_full = n)
    call_estimator(estimator, X[rows, , drop = FALSE], scaled, random_state = random_state + seed)
  }
  labels_train <- fit_labels(idx$train)
  labels_test <- if (policy$run_generalizability) fit_labels(idx$test) else NULL
  labels_stability <- if (policy$run_stability) fit_labels(idx$stability) else NULL

  k_train <- count_clusters(labels_train)
  k_test <- if (policy$run_generalizability) count_clusters(labels_test) else 0L
  k_stability <- if (policy$run_stability) count_clusters(labels_stability) else 0L

  if (sweep_param == "n_clusters") {
    expected <- as.integer(params$n_clusters)
    if (k_train != expected) {
      warning(sprintf("labels_1 has %d clusters, expected %d", k_train, expected), call. = FALSE)
    }
    if (policy$run_generalizability && k_test != expected) {
      warning(sprintf("labels_test has %d clusters, expected %d", k_test, expected), call. = FALSE)
    }
    if (policy$run_stability && k_stability != expected) {
      warning(sprintf("labels_2 has %d clusters, expected %d", k_stability, expected), call. = FALSE)
    }
  } else if (k_train < 2L) {
    warning(sprintf(
      "%s with %s produced %d cluster(s) on a subsample; stability and generalizability are degenerate at this point on the sweep axis.",
      estimator_name, format_params(params), k_train
    ), call. = FALSE)
  }

  # Stability: ARI on the samples the two subsamples share.
  ari_stability <- NaN
  if (policy$run_stability) {
    shared <- intersect(idx$train, idx$stability)
    ari_stability <- adjusted_rand_index(
      labels_train[match(shared, idx$train)],
      labels_stability[match(shared, idx$stability)]
    )
  }

  # Generalizability: the classifier learns the first subsample's clusters
  # from its raw features and predicts the held-out samples.
  labels_predicted <- NULL
  ari_generalizability <- NaN
  if (policy$run_generalizability) {
    predict_labels <- default_generalizability_classifier(
      classifier,
      n_features = ncol(X),
      n_trees = n_trees,
      random_state = random_state + seed,
      n_threads = classifier_threads
    )
    labels_predicted <- predict_labels(
      X[idx$train, , drop = FALSE],
      labels_train,
      X[idx$test, , drop = FALSE]
    )
    ari_generalizability <- adjusted_rand_index(labels_test, labels_predicted)
  }

  list(
    ari_stability = ari_stability,
    ari_generalizability = ari_generalizability,
    labels_train = labels_train,
    labels_test = labels_test,
    labels_predicted = labels_predicted,
    labels_stability = labels_stability,
    train_indices = idx$train,
    test_indices = idx$test,
    stability_indices = idx$stability,
    n_clusters_train = k_train,
    n_clusters_test = k_test,
    n_clusters_stability = k_stability,
    noise_fraction = mean(labels_train < 0)
  )
}
