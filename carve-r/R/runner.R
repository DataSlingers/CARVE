# Core validation runner. Mirrors _runner.py.
#
# Parallelization: `n_jobs > 1` dispatches the inner resample loop via
# `furrr::future_map` under a scoped `future::plan(multisession)`. Every
# resample seeds its own RNG as `random_state + b` (where `b` is the
# resample index), so results are reproducible *across* `n_jobs` values,
# not just within a fixed `n_jobs`. Progress reporting piggybacks on
# `progressr::progressor`, which works transparently over futures.


#' Run the CARVE validation loop
#'
#' For each `(estimator, param-set)` configuration, draw `n_resamples`
#' paired subsamples, cluster each, align labels across subsamples, and
#' accumulate stability ARI, generalizability ARI (via random-forest
#' prediction), consensus matrices, and per-sample generalizability
#' scores.
#'
#' Mirrors `_runner.run_validation`.
#'
#' @param X Numeric matrix of shape `(n_samples, n_features)`.
#' @param estimator_grids List of grid specs (see [default_estimator_grids()]).
#' @param n_resamples Number of resamples per configuration.
#' @param subsample_ratio Proportion of samples in each subsample.
#' @param normalization_options,dim_reduction_options Preprocessor spec
#'   lists; only used when `randomize_preprocessing = TRUE`.
#' @param classifier Currently ignored; a random-forest via `ranger` is used.
#' @param n_trees Integer; number of trees in the random forest.
#' @param randomize_preprocessing Logical; sample preprocessing per resample.
#' @param n_jobs Integer; when `> 1`, the resample loop runs over
#'   `furrr::future_map` with a scoped `future::plan(multisession,
#'   workers = min(n_jobs, n_resamples))`. Per-resample seeds make the
#'   result reproducible across `n_jobs` values.
#' @param random_state Integer seed, or `NULL`.
#' @param show_progress Logical; when `TRUE`, a `progressr` progress bar
#'   is emitted per resample. Register a handler (e.g.
#'   `progressr::handlers("txtprogressbar")`) to see the bar.
#' @param mode One of `"default"`, `"stability"`, `"generalizability"`.
#' @param verbose Integer verbosity level (0–2).
#'
#' @return A list with elements
#'   * `estimator_results` — data frame, one row per configuration.
#'   * `pipeline_records` — list of per-config resample pipeline records
#'     when `randomize_preprocessing = TRUE`; otherwise empty.
#'   * `consensus_matrices`, `consensus_generalizability_matrices` — lists
#'     of per-config consensus matrices (each a dense matrix).
#'   * `generalizability_scores` — list of per-config numeric vectors.
#'
#' @export
run_validation <- function(X,
                           estimator_grids,
                           n_resamples,
                           subsample_ratio,
                           normalization_options = default_normalization_options(),
                           dim_reduction_options = default_dim_reduction_options(X, subsample_ratio),
                           classifier = NULL,
                           n_trees = 100L,
                           randomize_preprocessing = FALSE,
                           n_jobs = 1L,
                           random_state = NULL,
                           show_progress = FALSE,
                           mode = "default",
                           verbose = 0L) {
  X <- ensure_2d_matrix(X)
  storage.mode(X) <- "double"
  policy <- resolve_mode(mode)
  n <- nrow(X)
  n_jobs <- max(1L, as.integer(n_jobs))

  print_run_header(
    X, n_clusters = "<per grid>",
    n_resamples = n_resamples, subsample_ratio = subsample_ratio,
    estimator_grids = estimator_grids, n_jobs = n_jobs,
    randomize_preprocessing = randomize_preprocessing,
    random_state = random_state, verbose = verbose
  )

  # Flatten grids to a list of (type, params) configs, mirroring Python's
  # ParameterGrid.
  configs <- unlist(
    lapply(estimator_grids, function(g) {
      lapply(expand_grid_spec(g$grid), function(p) {
        list(type = g$type, params = p)
      })
    }),
    recursive = FALSE
  )
  total_configs <- length(configs)

  estimator_records <- vector("list", total_configs)
  pipeline_records <- list()
  consensus_matrices <- vector("list", total_configs)
  consensus_gen_matrices <- vector("list", total_configs)
  generalizability_scores <- vector("list", total_configs)

  # Scope a parallel plan for the duration of this call, only if needed.
  # `min(n_jobs, n_resamples)` avoids idle workers when n_resamples is small.
  if (n_jobs > 1L) {
    workers <- min(n_jobs, as.integer(n_resamples))
    old_plan <- future::plan(future::multisession, workers = workers)
    on.exit(future::plan(old_plan), add = TRUE)
  }

  # Local-bind the internal helper so `future` captures it as a global.
  # Under `devtools::load_all`, the CARVE namespace is not installed in
  # parallel workers, so relying on `packages = "CARVE"` alone would
  # raise `could not find function "validation_iter"`. The explicit
  # binding makes this work in dev, test, and installed contexts alike.
  .validation_iter <- validation_iter

  run_one_config <- function(cfg, progress_fn) {
    iter_fn <- function(b) {
      res <- .validation_iter(
        X = X, type = cfg$type, params = cfg$params,
        subsample_ratio = subsample_ratio, n_resamples = n_resamples,
        seed = b, normalization_options = normalization_options,
        dim_reduction_options = dim_reduction_options,
        classifier = classifier, n_trees = n_trees,
        randomize_preprocessing = randomize_preprocessing,
        mode = mode, random_state = random_state
      )
      if (!is.null(progress_fn)) progress_fn()
      res
    }
    if (n_jobs > 1L) {
      furrr::future_map(
        seq_len(n_resamples), iter_fn,
        .options = furrr::furrr_options(seed = TRUE, packages = "CARVE")
      )
    } else {
      lapply(seq_len(n_resamples), iter_fn)
    }
  }

  # progressr::with_progress activates registered handlers only when
  # show_progress is TRUE. progressor() returns a no-op-ish function
  # outside with_progress, so the inner code path is identical either way.
  core_loop <- function(progress_fn) {
    for (config_idx in seq_len(total_configs)) {
      cfg <- configs[[config_idx]]
      results <- run_one_config(cfg, progress_fn)

    aris_stab <- vapply(results, function(r) r$ari_stability, numeric(1L))
    aris_gen <- vapply(results, function(r) r$ari_generalizability, numeric(1L))
    aris_avg <- if (policy$compute_average_ari) {
      (aris_stab + aris_gen) / 2
    } else {
      rep(NaN, n_resamples)
    }

    M <- if (policy$run_stability) {
      compute_consensus_matrix(
        n_samples = n,
        runs = lapply(results, function(r) {
          list(sample_idx = r$train_indices, labels = r$labels_train)
        })
      )
    } else NULL

    M_g <- if (policy$run_generalizability) {
      compute_consensus_matrix(
        n_samples = n,
        runs = lapply(results, function(r) {
          list(sample_idx = r$test_indices, labels = r$labels_predicted)
        })
      )
    } else NULL

    E <- if (policy$run_generalizability) {
      compute_generalizability_scores(
        n_samples = n,
        runs = lapply(results, function(r) {
          list(sample_idx = r$test_indices,
               labels = r$labels_test,
               predicted = r$labels_predicted)
        })
      )
    } else NULL

    consensus_matrices[[config_idx]] <<- M
    consensus_gen_matrices[[config_idx]] <<- M_g
    generalizability_scores[[config_idx]] <<- E

    stab <- summarize_ari_scores(aris_stab)
    gen <- summarize_ari_scores(aris_gen)
    avg <- summarize_ari_scores(aris_avg)

    record <- c(
      list(estimator = .estimator_display_name(cfg$type)),
      cfg$params,
      list(
        ari_stability = unname(stab["mean"]),
        ari_stability_se = unname(stab["se"]),
        ari_stability_upper = unname(stab["q95"]),
        ari_stability_lower = unname(stab["q05"]),
        ari_generalizability = unname(gen["mean"]),
        ari_generalizability_se = unname(gen["se"]),
        ari_generalizability_upper = unname(gen["q95"]),
        ari_generalizability_lower = unname(gen["q05"]),
        ari_average = unname(avg["mean"]),
        ari_average_se = unname(avg["se"]),
        ari_average_upper = unname(avg["q95"]),
        ari_average_lower = unname(avg["q05"])
      )
    )
    estimator_records[[config_idx]] <<- record

    if (randomize_preprocessing) {
      pipeline_records[[length(pipeline_records) + 1L]] <<- list(
        estimator = .estimator_display_name(cfg$type),
        params = cfg$params,
        results = results
      )
    }

    log_config_progress(config_idx, total_configs, cfg$type, cfg$params,
                        record, verbose)
    }
  }

  if (isTRUE(show_progress)) {
    progressr::with_progress({
      prog <- progressr::progressor(steps = total_configs * n_resamples)
      core_loop(prog)
    })
  } else {
    core_loop(NULL)
  }

  estimator_df <- records_to_dataframe(estimator_records)
  print_run_footer(estimator_df, verbose)

  list(
    estimator_results = estimator_df,
    pipeline_records = pipeline_records,
    consensus_matrices = consensus_matrices,
    consensus_generalizability_matrices = consensus_gen_matrices,
    generalizability_scores = generalizability_scores
  )
}


#' Run a single resampling iteration
#'
#' Mirrors `_runner.validation_iter`. Exposed for testing.
#'
#' @inheritParams run_validation
#' @param type Estimator backend name (e.g. `"kmeans"`).
#' @param params Named list of estimator parameters.
#' @param seed Per-resample seed offset.
#'
#' @return A list with labels, indices, ARI metrics, and preprocessing
#'   metadata for this resample.
#'
#' @export
validation_iter <- function(X, type, params, subsample_ratio, n_resamples,
                            seed, normalization_options,
                            dim_reduction_options,
                            classifier = NULL, n_trees = 100L,
                            randomize_preprocessing = FALSE,
                            mode = "default", random_state = NULL) {
  policy <- resolve_mode(mode)
  n_samples <- nrow(X)
  rs0 <- if (is.null(random_state)) 0L else as.integer(random_state)

  split1 <- split_subsample_indices(
    n_samples, subsample_ratio = subsample_ratio, random_state = rs0 + seed
  )
  P_1_idx <- split1$train
  P_test_idx <- split1$test

  P_2_idx <- NULL
  if (policy$run_stability) {
    split2 <- split_subsample_indices(
      n_samples, subsample_ratio = subsample_ratio,
      random_state = rs0 + seed + n_resamples
    )
    P_2_idx <- split2$train
  }

  pp <- build_preprocessing_pipeline(
    randomize = randomize_preprocessing,
    normalization_options = normalization_options,
    dim_reduction_options = dim_reduction_options,
    seed = rs0 + seed
  )
  X_preprocessed <- pp$pipeline(X)

  X_1 <- X_preprocessed[P_1_idx, , drop = FALSE]
  X_test <- if (policy$run_generalizability) {
    X_preprocessed[P_test_idx, , drop = FALSE]
  } else NULL
  X_2 <- if (policy$run_stability) {
    X_preprocessed[P_2_idx, , drop = FALSE]
  } else NULL

  spec <- c(list(type = type), params)
  labels_1 <- cluster_labels(X_1, spec, random_state = rs0 + seed)
  labels_test <- if (policy$run_generalizability) {
    cluster_labels(X_test, spec, random_state = rs0 + seed)
  } else NULL
  labels_2 <- if (policy$run_stability) {
    cluster_labels(X_2, spec, random_state = rs0 + seed)
  } else NULL

  n_clusters <- params$n_clusters
  if (!is.null(n_clusters)) {
    if (length(unique(labels_1)) != n_clusters) {
      warning(sprintf("labels_1 has %d clusters, expected %d",
                      length(unique(labels_1)), n_clusters),
              call. = FALSE)
    }
    if (policy$run_generalizability &&
        length(unique(labels_test)) != n_clusters) {
      warning(sprintf("labels_test has %d clusters, expected %d",
                      length(unique(labels_test)), n_clusters),
              call. = FALSE)
    }
    if (policy$run_stability &&
        length(unique(labels_2)) != n_clusters) {
      warning(sprintf("labels_2 has %d clusters, expected %d",
                      length(unique(labels_2)), n_clusters),
              call. = FALSE)
    }
  }

  ari_stab <- compute_stability_ari(policy, P_1_idx, P_2_idx,
                                    labels_1, labels_2)

  gen <- compute_generalizability_ari(
    policy = policy, X_1 = X_1, X_test = X_test,
    labels_1 = labels_1, labels_test = labels_test,
    n_trees = n_trees, seed = rs0 + seed
  )

  list(
    ari_stability = ari_stab,
    ari_generalizability = gen$ari,
    labels_train = labels_1,
    labels_test = labels_test,
    labels_predicted = gen$labels_pred,
    labels_stability = labels_2,
    train_indices = P_1_idx,
    test_indices = P_test_idx,
    stability_indices = P_2_idx,
    normalization_params = pp$normalization_params,
    dim_reduction_params = pp$dim_reduction_params,
    normalization_name = pp$normalization_name,
    dim_reduction_name = pp$dim_reduction_name
  )
}


# Stability ARI on the overlap of two subsamples.
compute_stability_ari <- function(policy, P_1_idx, P_2_idx, labels_1, labels_2) {
  if (!policy$run_stability) return(NaN)
  # Intersect and retrieve per-subsample positional indices.
  shared <- intersect(P_1_idx, P_2_idx)
  if (length(shared) == 0L) return(NaN)
  i_1 <- match(shared, P_1_idx)
  i_2 <- match(shared, P_2_idx)
  adjusted_rand_index(labels_1[i_1], labels_2[i_2])
}


# Generalizability ARI via random-forest prediction.
compute_generalizability_ari <- function(policy, X_1, X_test, labels_1,
                                         labels_test, n_trees, seed) {
  if (!policy$run_generalizability) {
    return(list(labels_pred = NULL, ari = NaN))
  }
  p <- ncol(X_1)
  # ranger requires column names on x.
  X_1_named <- X_1
  X_test_named <- X_test
  if (is.null(colnames(X_1_named))) {
    cn <- paste0("V", seq_len(p))
    colnames(X_1_named) <- cn
    colnames(X_test_named) <- cn
  }
  rf <- .with_seed(seed, {
    ranger::ranger(
      x = X_1_named,
      y = factor(labels_1),
      num.trees = n_trees,
      mtry = max(1L, as.integer(sqrt(p))),
      max.depth = p,
      probability = FALSE,
      classification = TRUE,
      num.threads = 1L,
      seed = seed,
      verbose = FALSE
    )
  })
  pred <- stats::predict(rf, data = X_test_named)$predictions
  labels_pred <- as.integer(as.character(pred))
  ari <- adjusted_rand_index(labels_test, labels_pred)
  list(labels_pred = labels_pred, ari = ari)
}


# Display name for a record's `estimator` column. Mirrors Python's
# `est_class.__name__` (e.g. `"KMeans"`, `"AgglomerativeClustering"`,
# `"SpectralClusteringCARVE"`).
.estimator_display_name <- function(type) {
  switch(type,
    kmeans = "KMeans",
    agglomerative = "AgglomerativeClustering",
    spectral = "SpectralClusteringCARVE",
    type
  )
}


# Convert a list of heterogeneous record lists into a data frame. Columns
# are the union of keys across records; missing fields become NA.
records_to_dataframe <- function(records) {
  if (length(records) == 0L) return(data.frame())
  all_cols <- unique(unlist(lapply(records, names)))
  mat <- lapply(all_cols, function(col) {
    vals <- lapply(records, function(r) r[[col]] %||% NA)
    # Unbox: if all values are length-1 scalars, use a simple atomic vector.
    unlist(vals, use.names = FALSE)
  })
  names(mat) <- all_cols
  as.data.frame(mat, stringsAsFactors = FALSE)
}
