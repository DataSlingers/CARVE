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

resample_backend <- function(outer, BPPARAM = NULL) {
  if (is.null(BPPARAM) && outer == 1L) {
    return(function(X, FUN) lapply(X, FUN))
  }
  if (is.null(BPPARAM)) {
    BPPARAM <- if (.Platform$OS.type == "windows") {
      BiocParallel::SnowParam(workers = outer)
    } else {
      BiocParallel::MulticoreParam(workers = outer)
    }
  }
  function(X, FUN) BiocParallel::bplapply(X, FUN, BPPARAM = BPPARAM)
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
                           sweep, mode = "default", show_progress = FALSE, verbose = 0L) {
  policy <- resolve_mode(mode)
  n <- nrow(X)
  # Workers over resamples (outer), and threads for each worker's forest
  # (inner). A BPPARAM brings its own worker count.
  budget <- resolve_core_budget(n_jobs, n_resamples)
  if (!is.null(BPPARAM)) {
    workers <- as.integer(BiocParallel::bpnworkers(BPPARAM))
    budget <- c(outer = workers, inner = max(1L, n_cores() %/% workers))
  }
  apply_resamples <- resample_backend(budget[["outer"]], BPPARAM)

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
    outputs <- apply_resamples(seq_len(n_resamples) - 1L, function(b) {
      collect_warnings(validation_iter(
        X, g$estimator, g$name, params,
        subsample_ratio = subsample_ratio,
        n_resamples = n_resamples,
        seed = b,
        classifier = classifier,
        n_trees = n_trees,
        sweep_param = sweep@param,
        mode = mode,
        random_state = random_state,
        classifier_threads = budget[["inner"]]
      ))
    })
    for (text in unique(unlist(lapply(outputs, function(o) o$warnings)))) {
      warning(text, call. = FALSE)
    }
    results <- lapply(outputs, function(o) o$value)

    stability_runs <- Filter(function(r) length(r$labels_train) > 0L, results)
    held_out_runs <- Filter(
      function(r) length(r$labels_predicted) > 0L && length(r$labels_test) > 0L,
      results
    )
    if (policy$run_stability) {
      M <- compute_consensus_matrix(n, lapply(stability_runs, function(r) {
        list(indices = r$train_indices, labels = r$labels_train)
      }))
      scores <- stability_from_consensus(M)
      consensus[i] <- list(M)
      summaries[i] <- list(list(gini = scores$gini, ce = scores$ce, pac = compute_consensus_pac(M)))
    }
    if (policy$run_generalizability) {
      consensus_generalizability[i] <- list(compute_consensus_matrix(n, lapply(held_out_runs, function(r) {
        list(indices = r$test_indices, labels = r$labels_predicted)
      })))
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
