# The resampling loop. Mirrors _runner.py: a serial loop over
# configurations with the resamples inside it, run in this process or
# through BiocParallel. Every resample derives its own seeds from
# random_state, so the results do not depend on the backend.

# The only place the subsample seeds are derived, so the embedding pass and
# the configuration loop draw the same subsamples.
resample_indices <- function(n_samples, seed, n_resamples, subsample_ratio,
                             random_state, run_stability) {
  first <- split_subsample_indices(n_samples, subsample_ratio, random_state + seed)
  second <- NULL
  if (run_stability) {
    second <- split_subsample_indices(n_samples, subsample_ratio, random_state + seed + n_resamples)$train
  }
  list(train = first$train, test = first$test, stability = second)
}

# Python's warning when the noise policy leaves a subsample empty.
ALL_NOISE_WARNING <- "All points in a subsample were labelled as noise and dropped (noise_policy='drop'); %s ARI is undefined for this resample. Consider relaxing the clustering parameters or switching to noise_policy='as_cluster'."

validation_iter <- function(X, estimator, estimator_name, params, subsample_ratio,
                            n_resamples, seed, classifier = NULL, n_trees = 100L,
                            sweep_param = "n_clusters", noise_policy = "drop",
                            mode = "default", random_state = 0L, classifier_threads = 1L,
                            embeddings = NULL) {
  policy <- resolve_mode(mode)
  n <- nrow(X)
  idx <- resample_indices(n, seed, n_resamples, subsample_ratio, random_state, policy$run_stability)

  # What the estimator clusters: the raw subsamples, or under randomized
  # preprocessing this resample's embeddings, one independent fit per
  # subsample.
  pipeline <- NULL
  if (is.null(embeddings)) {
    X_1 <- X[idx$train, , drop = FALSE]
    X_test <- if (policy$run_generalizability) X[idx$test, , drop = FALSE] else NULL
    X_2 <- if (policy$run_stability) X[idx$stability, , drop = FALSE] else NULL
  } else {
    check_embeddings(policy, embeddings, idx)
    pipeline <- embeddings$spec$label
    X_1 <- embeddings$X_1
    X_test <- embeddings$X_test
    X_2 <- embeddings$X_2
  }

  # Each set is clustered on its own, with the neighbor count scaled to its
  # own rows.
  fit_labels <- function(X_fit) {
    scaled <- scale_neighbor_count(estimator, params, n_fit = nrow(X_fit), n_full = n)
    call_estimator(estimator, X_fit, scaled, random_state = random_state + seed)
  }
  labels_train <- fit_labels(X_1)
  labels_test <- if (policy$run_generalizability) fit_labels(X_test) else NULL
  labels_stability <- if (policy$run_stability) fit_labels(X_2) else NULL

  # Noise labels. Under "drop" the index vectors shrink with the labels. The
  # classifier reads raw features through these indices and the embeddings
  # are not used again, so nothing else needs re-slicing.
  train <- apply_noise_policy(idx$train, labels_train, noise_policy)
  idx$train <- train$indices
  labels_train <- train$labels
  if (policy$run_generalizability) {
    test <- apply_noise_policy(idx$test, labels_test, noise_policy)
    idx$test <- test$indices
    labels_test <- test$labels
  }
  if (policy$run_stability) {
    second <- apply_noise_policy(idx$stability, labels_stability, noise_policy)
    idx$stability <- second$indices
    labels_stability <- second$labels
  }

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
    if (length(idx$train) == 0L || length(idx$stability) == 0L) {
      warning(sprintf(ALL_NOISE_WARNING, "stability"), call. = FALSE)
    } else {
      shared <- intersect(idx$train, idx$stability)
      ari_stability <- adjusted_rand_index(
        labels_train[match(shared, idx$train)],
        labels_stability[match(shared, idx$stability)]
      )
    }
  }

  # Generalizability: the classifier learns the first subsample's clusters
  # from its raw features and predicts the held-out samples. It reads raw
  # features also under randomized preprocessing: a cluster generalizes only
  # if it can be learned from the data, and a transform such as t-SNE cannot
  # embed new samples anyway.
  labels_predicted <- NULL
  ari_generalizability <- NaN
  if (policy$run_generalizability) {
    if (length(idx$train) == 0L || length(idx$test) == 0L) {
      warning(sprintf(ALL_NOISE_WARNING, "generalizability"), call. = FALSE)
    } else {
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
    pipeline = pipeline,
    n_clusters_train = k_train,
    n_clusters_test = k_test,
    n_clusters_stability = k_stability,
    noise_fraction = train$noise_fraction
  )
}

# The embedding pass and validation_iter() derive the same indices from
# resample_indices(); a row count that differs means that has broken.
check_embeddings <- function(policy, embeddings, idx) {
  expected <- list(X_1 = idx$train)
  if (policy$run_stability) {
    expected$X_2 <- idx$stability
  }
  if (policy$run_generalizability) {
    expected$X_test <- idx$test
  }
  for (name in names(expected)) {
    rows <- if (is.null(embeddings[[name]])) "None" else nrow(embeddings[[name]])
    if (!identical(rows, length(expected[[name]]))) {
      stop(sprintf(
        "Precomputed embedding %s has %s rows but this resample's subsample has %d. This is an internal CARVE error.",
        name, rows, length(expected[[name]])
      ), call. = FALSE)
    }
  }
  invisible(TRUE)
}

# Fits resample `seed`'s pipeline on each subsample its mode uses: the first
# with random_state + seed, the second with random_state + seed +
# n_resamples and the held-out set with random_state + seed + 2 *
# n_resamples, so no two fits share a seed.
embed_resample <- function(X, spec, seed, n_resamples, subsample_ratio, random_state,
                           mode = "default") {
  policy <- resolve_mode(mode)
  idx <- resample_indices(nrow(X), seed, n_resamples, subsample_ratio, random_state, policy$run_stability)
  base <- random_state + seed
  embed <- function(rows, offset) {
    pipeline_from_spec(spec, base + offset)(X[rows, , drop = FALSE])
  }
  list(
    spec = spec,
    X_1 = embed(idx$train, 0L),
    X_2 = if (policy$run_stability) embed(idx$stability, n_resamples) else NULL,
    X_test = if (policy$run_generalizability) embed(idx$test, 2L * n_resamples) else NULL
  )
}

# One resample of one configuration, described by its task: the resample
# index b and, under randomized preprocessing, that resample's embeddings.
# It is a namespace function, so a worker receives these arguments and
# nothing else; a closure made inside run_validation would carry that frame,
# with every consensus matrix computed so far, to every worker. The data
# argument is not called X, which would clash with the X of lapply and
# bplapply.
# The resample runs under the seed random_state + b, so an estimator or
# classifier that draws without a random_state argument gives the same
# results on every backend and leaves the caller's stream alone. The
# built-in functions reseed themselves inside it.
run_resample <- function(task, data, estimator, estimator_name, params, subsample_ratio,
                         n_resamples, classifier, n_trees, sweep_param, noise_policy, mode,
                         random_state, classifier_threads) {
  seeded(random_state + task$b, collect_warnings(validation_iter(
    data, estimator, estimator_name, params,
    subsample_ratio = subsample_ratio,
    n_resamples = n_resamples,
    seed = task$b,
    classifier = classifier,
    n_trees = n_trees,
    sweep_param = sweep_param,
    noise_policy = noise_policy,
    mode = mode,
    random_state = random_state,
    classifier_threads = classifier_threads,
    embeddings = task$embeddings
  )))
}

# Workers over resamples (outer) and classifier threads per worker (inner).
# A BPPARAM brings its own worker count; as with n_jobs, at most
# n_resamples of them have a resample, so the threads are shared among
# those.
run_core_budget <- function(n_jobs, n_resamples, BPPARAM = NULL) {
  if (is.null(BPPARAM)) {
    return(resolve_core_budget(n_jobs, n_resamples))
  }
  workers <- as.integer(BiocParallel::bpnworkers(BPPARAM))
  busy <- max(1L, min(workers, as.integer(n_resamples)))
  c(outer = workers, inner = max(1L, n_cores() %/% busy))
}

# The BiocParallel backend for the resamples, or NULL to run them in this
# process.
resample_bpparam <- function(outer, BPPARAM = NULL) {
  if (!is.null(BPPARAM)) {
    return(BPPARAM)
  }
  if (outer == 1L) {
    return(NULL)
  }
  if (.Platform$OS.type == "windows") {
    BiocParallel::SnowParam(workers = outer)
  } else {
    BiocParallel::MulticoreParam(workers = outer)
  }
}

resample_backend <- function(BPPARAM = NULL) {
  if (is.null(BPPARAM)) {
    return(function(X, FUN, ...) lapply(X, FUN, ...))
  }
  function(X, FUN, ...) BiocParallel::bplapply(X, FUN, ..., BPPARAM = BPPARAM)
}

summary_columns <- function(prefix, scores) {
  s <- summarize_ari_scores(scores)
  stats::setNames(as.list(unname(s)), paste0(prefix, c("", "_se", "_upper", "_lower")))
}

# One row per record. Columns appear in first-seen order and are NA where a
# record lacks them (another estimator's parameters), as in pandas'
# DataFrame.from_records. A parameter whose values are not scalars becomes a
# list column.
bind_records <- function(records) {
  columns <- unique(unlist(lapply(records, names)))
  out <- lapply(columns, function(column) {
    values <- lapply(records, function(record) {
      value <- record[[column]]
      if (is.null(value)) NA else value
    })
    scalar <- all(lengths(values) == 1L) && all(vapply(values, is.atomic, logical(1)))
    if (scalar) unlist(values) else I(values)
  })
  names(out) <- columns
  as.data.frame(out, stringsAsFactors = FALSE, check.names = FALSE)
}

run_validation <- function(X, estimator_grids, n_resamples, subsample_ratio, classifier = NULL,
                           n_trees = 100L, n_jobs = 1L, BPPARAM = NULL, random_state = 0L,
                           sweep, noise_policy = "drop", mode = "default", anchors = NULL,
                           show_progress = FALSE, verbose = 0L) {
  policy <- resolve_mode(mode)
  n <- nrow(X)
  budget <- run_core_budget(n_jobs, n_resamples, BPPARAM)
  # The backend starts once for the whole run. Left to bplapply, a backend
  # that is not running starts and stops for every configuration, and a
  # SnowParam launches new R processes each time. A backend the caller
  # already started stays up.
  BPPARAM <- resample_bpparam(budget[["outer"]], BPPARAM)
  if (!is.null(BPPARAM) && !BiocParallel::bpisup(BPPARAM)) {
    BiocParallel::bpstart(BPPARAM)
    on.exit(BiocParallel::bpstop(BPPARAM), add = TRUE)
  }
  apply_resamples <- resample_backend(BPPARAM)

  # One task per resample; a worker receives its own tasks only.
  tasks <- lapply(seq_len(n_resamples) - 1L, function(b) list(b = b, embeddings = NULL))

  configs <- unlist(lapply(estimator_grids, function(g) {
    lapply(expand_param_grid(g$grid), function(params) list(grid = g, params = params))
  }), recursive = FALSE)
  total <- length(configs)
  ids <- as.character(seq_len(total) - 1L)
  assign_method <- method_id_assigner(sweep@param)

  records <- vector("list", total)
  consensus <- stats::setNames(vector("list", total), ids)
  consensus_generalizability <- stats::setNames(vector("list", total), ids)
  generalizability_scores <- stats::setNames(vector("list", total), ids)
  summaries <- stats::setNames(vector("list", total), ids)

  bar <- NULL
  if (show_progress) {
    bar <- utils::txtProgressBar(min = 0, max = total, style = 3, file = stderr())
    on.exit(close(bar), add = TRUE)
  }

  for (i in seq_len(total)) {
    g <- configs[[i]]$grid
    params <- configs[[i]]$params
    outputs <- apply_resamples(
      tasks, run_resample,
      data = X,
      estimator = g$estimator,
      estimator_name = g$name,
      params = params,
      subsample_ratio = subsample_ratio,
      n_resamples = n_resamples,
      classifier = classifier,
      n_trees = n_trees,
      sweep_param = sweep@param,
      noise_policy = noise_policy,
      mode = mode,
      random_state = random_state,
      classifier_threads = budget[["inner"]]
    )
    for (text in unique(unlist(lapply(outputs, function(o) o$warnings)))) {
      warning(text, call. = FALSE)
    }
    results <- lapply(outputs, function(o) o$value)

    # A resample whose subsample was all noise under "drop" has nothing to
    # aggregate; it already warned.
    stability_runs <- Filter(function(r) length(r$labels_train) > 0L, results)
    held_out_runs <- Filter(
      function(r) length(r$labels_predicted) > 0L && length(r$labels_test) > 0L,
      results
    )
    stability_pairs <- lapply(stability_runs, function(r) {
      list(indices = r$train_indices, labels = r$labels_train)
    })
    held_out_pairs <- lapply(held_out_runs, function(r) {
      list(indices = r$test_indices, labels = r$labels_predicted)
    })
    if (policy$run_stability) {
      # Anchored, the stored matrix is the anchor block, and PAC comes from
      # it; the per-sample scores still cover every sample.
      if (is.null(anchors)) {
        M <- compute_consensus_matrix(n, stability_pairs)
        scores <- stability_from_consensus(M)
      } else {
        M <- consensus_anchor_block(n, stability_pairs, anchors)
        scores <- stability_from_runs_anchored(n, stability_pairs, anchors)
      }
      consensus[i] <- list(M)
      summaries[i] <- list(list(gini = scores$gini, ce = scores$ce, pac = compute_consensus_pac(M)))
    }
    if (policy$run_generalizability) {
      consensus_generalizability[i] <- list(if (is.null(anchors)) {
        compute_consensus_matrix(n, held_out_pairs)
      } else {
        consensus_anchor_block(n, held_out_pairs, anchors)
      })
      generalizability_scores[i] <- list(compute_generalizability_scores(n, lapply(held_out_runs, function(r) {
        list(indices = r$test_indices, true = r$labels_test, predicted = r$labels_predicted)
      })))
    }

    aris_stability <- vapply(results, function(r) r$ari_stability, numeric(1))
    aris_generalizability <- vapply(results, function(r) r$ari_generalizability, numeric(1))
    aris_average <- if (policy$compute_average_ari) {
      (aris_stability + aris_generalizability) / 2
    } else {
      rep(NaN, length(results))
    }
    k_observed <- vapply(results, function(r) as.numeric(r$n_clusters_train), numeric(1))
    noise <- vapply(results, function(r) r$noise_fraction, numeric(1))
    sweep_value <- params[[sweep@param]]
    method <- assign_method(g$name, params)

    record <- c(
      list(config_id = i - 1L, method_id = method[[1L]], method_label = method[[2L]], estimator = g$name),
      params,
      list(
        sweep_param = sweep@param,
        sweep_value = sweep_value,
        sweep_rank = sweep_rank_of(sweep, sweep_value),
        n_clusters_observed = mean(k_observed),
        n_clusters_observed_se = if (length(k_observed) > 1L) {
          stats::sd(k_observed) / sqrt(length(k_observed))
        } else {
          NaN
        },
        noise_fraction = mean(noise)
      ),
      summary_columns("ari_stability", aris_stability),
      summary_columns("ari_generalizability", aris_generalizability),
      summary_columns("ari_average", aris_average)
    )
    records[[i]] <- record
    log_config_progress(i, total, g$name, params, record, sweep@param, verbose)
    if (!is.null(bar)) {
      utils::setTxtProgressBar(bar, i)
    }
  }

  list(
    records = bind_records(records),
    consensus_matrices = consensus,
    consensus_generalizability_matrices = consensus_generalizability,
    generalizability_scores = generalizability_scores,
    summaries = if (policy$run_stability) summaries else NULL
  )
}
