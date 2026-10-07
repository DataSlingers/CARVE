# Compares the R package with the Python package on the same data and
# writes data-raw/parity-report.md. Run from code/carve-r:
#
#   Rscript data-raw/parity_check.R
#
# It loads the package from source with devtools, and needs reticulate, the
# dbscan and uwot packages, and the Python environment in ../.venv, where
# carve is installed. The two languages draw different subsamples, so the
# results agree within tolerances, not exactly.

devtools::load_all(".", quiet = TRUE)
# normalizePath() on the interpreter would follow the symlink out of the
# virtual environment, so resolve the environment directory instead.
reticulate::use_python(file.path(normalizePath("../.venv"), "bin", "python"), required = TRUE)
datasets <- reticulate::import("sklearn.datasets")
preprocessing <- reticulate::import("sklearn.preprocessing")
decomposition <- reticulate::import("sklearn.decomposition")
np <- reticulate::import("numpy", convert = FALSE)
pycarve <- reticulate::import("carve")

n_resamples <- 100L
ks <- 2:6

# Largest allowed absolute difference between the R and the Python value of
# one configuration.
tolerance <- c(
  ari_stability = 0.05,
  ari_generalizability = 0.05,
  ari_average = 0.05,
  consensus_pac_stability = 0.03,
  consensus_gini_stability = 0.03,
  consensus_ce_stability = 0.03,
  accuracy_generalizability = 0.03,
  n_clusters_observed = 0.5,
  noise_fraction = 0.02
)

# reticulate converts numeric pandas columns to R vectors but leaves string
# columns as Python objects, so those are rebuilt with tolist().
py_table <- function(frame, columns) {
  out <- frame[, columns]
  for (column in columns) {
    if (!is.atomic(out[[column]])) {
      out[[column]] <- unlist(out[[column]]$tolist())
    }
  }
  out
}

largest_difference <- function(a, b) {
  difference <- abs(a - b)
  if (all(is.na(difference))) NA_real_ else max(difference, na.rm = TRUE)
}

compare_tables <- function(r_table, python_table, keys, columns) {
  joined <- merge(
    r_table[, c(keys, columns)],
    python_table[, c(keys, columns)],
    by = keys,
    suffixes = c("_r", "_py")
  )
  list(
    rows = nrow(joined),
    expected = nrow(r_table),
    differences = vapply(columns, function(column) {
      largest_difference(joined[[paste0(column, "_r")]], joined[[paste0(column, "_py")]])
    }, numeric(1))
  )
}

# The selected k on the k axis, the selected sweep value elsewhere.
compare_selections <- function(r_fit, py_fit, by_value) {
  out <- expand.grid(
    measure = c("stability", "generalizability", "average"),
    rule = c("max", "1se"),
    stringsAsFactors = FALSE
  )
  out$r <- mapply(function(m, r) {
    if (by_value) get_sweep_value(r_fit, measure = m, rule = r) else get_k(r_fit, measure = m, rule = r)
  }, out$measure, out$rule)
  out$py <- mapply(function(m, r) {
    as.numeric(if (by_value) py_fit$get_sweep_value(measure = m, rule = r) else py_fit$get_k(measure = m, rule = r))
  }, out$measure, out$rule)
  out
}

compare_fits <- function(name, r_fit, py_fit, keys, by_value = FALSE, note = NULL,
                         pipelines = FALSE) {
  columns <- names(tolerance)
  out <- list(
    name = name,
    note = note,
    results = compare_tables(estimator_results(r_fit), py_table(py_fit$estimator_results_, c(keys, columns)), keys, columns),
    selections = compare_selections(r_fit, py_fit, by_value),
    label_ari = adjusted_rand_index(get_labels(r_fit), as.integer(py_fit$get_labels()))
  )
  if (pipelines) {
    pipeline_keys <- c("method_label", "pipeline", setdiff(keys, "method_label"))
    pipeline_columns <- c("n_resamples", "ari_stability", "ari_generalizability")
    out$pipelines <- compare_tables(
      preprocessing_results(r_fit),
      py_table(py_fit$preprocessing_results_, c(pipeline_keys, pipeline_columns)),
      pipeline_keys,
      pipeline_columns
    )
  }
  out
}

blobs <- function(n, std, seed, features = 2L, centers = 3L) {
  datasets$make_blobs(
    n_samples = as.integer(n), n_features = as.integer(features), centers = as.integer(centers),
    cluster_std = std, random_state = as.integer(seed)
  )[[1L]]
}

k_case <- function(name, X) {
  r_fit <- carve(X, n_clusters = ks, n_resamples = n_resamples, random_state = 0)
  py_fit <- pycarve$CARVE(n_clusters = np$array(ks), n_resamples = n_resamples, random_state = 0L)$fit(X)
  compare_fits(name, r_fit, py_fit, c("method_label", "n_clusters"))
}

resolution_case <- function() {
  X <- blobs(600, 2.0, 4, features = 10L, centers = 5L)
  resolutions <- c(0.1, 0.25, 0.5, 1, 2)
  r_fit <- carve(X, resolution = resolutions, n_resamples = n_resamples, random_state = 0)
  py_fit <- pycarve$CARVE(resolution = np$array(resolutions), n_resamples = n_resamples, random_state = 0L)$fit(X)
  compare_fits(
    "resolution sweep, 10-dimensional blobs", r_fit, py_fit, c("method_label", "resolution"),
    by_value = TRUE,
    note = "Leiden runs in igraph in R and in leidenalg in Python. Both optimize modularity at each resolution."
  )
}

hdbscan_case <- function() {
  X <- blobs(500, 1.0, 7, centers = 4L)
  sizes <- c(5L, 10L, 20L, 40L)
  r_fit <- carve(X, sweep = "min_cluster_size", sweep_values = sizes, n_resamples = n_resamples, random_state = 0)
  py_fit <- pycarve$CARVE(sweep = "min_cluster_size", sweep_values = np$array(sizes),
                          n_resamples = n_resamples, random_state = 0L)$fit(X)
  compare_fits(
    "min_cluster_size sweep, HDBSCAN", r_fit, py_fit, c("method_label", "min_cluster_size"),
    by_value = TRUE,
    note = "dbscan and scikit-learn can merge tied distances in a different order, so HDBSCAN can select different clusters on some subsamples (see the HDBSCAN help page)."
  )
}

anchored_case <- function(X) {
  # The anchored warning is expected.
  r_fit <- suppressWarnings(carve(X, n_clusters = ks, n_resamples = n_resamples, anchor_threshold = 200,
                                  random_state = 0))
  py_fit <- pycarve$CARVE(n_clusters = np$array(ks), n_resamples = n_resamples, anchor_threshold = 200L,
                          random_state = 0L)$fit(X)
  compare_fits(
    "easy blobs, anchored over 200 of 500 samples", r_fit, py_fit, c("method_label", "n_clusters"),
    note = "The two languages draw different anchors. PAC covers anchor pairs only, so it can differ more than in the exact case."
  )
}

randomized_case <- function() {
  X <- blobs(300, 0.8, 3)
  r_fit <- carve(
    X, n_clusters = 2:4, n_resamples = n_resamples, random_state = 0, randomize_preprocessing = TRUE,
    normalization_options = list(preprocessing_option(Identity), preprocessing_option(StandardScaler)),
    dim_reduction_options = list(preprocessing_option(Identity), preprocessing_option(PCA, n_components = 1L))
  )
  identity <- reticulate::tuple(preprocessing$FunctionTransformer, reticulate::dict())
  py_fit <- pycarve$CARVE(
    n_clusters = np$array(2:4), n_resamples = n_resamples, random_state = 0L,
    normalization_options = list(identity, reticulate::tuple(preprocessing$StandardScaler, reticulate::dict())),
    dim_reduction_options = list(identity, reticulate::tuple(decomposition$PCA, reticulate::dict(n_components = list(1L))))
  )$fit(X, randomize_preprocessing = TRUE)
  compare_fits(
    "easy blobs, randomized preprocessing", r_fit, py_fit, c("method_label", "n_clusters"),
    pipelines = TRUE,
    note = "Each of the four pipelines gets 25 of the 100 resamples in both languages, so n_resamples must agree exactly."
  )
}

format_table <- function(comparison, limits) {
  over <- !is.na(comparison$differences) & comparison$differences > limits
  shown <- ifelse(is.na(comparison$differences), "NA", sprintf("%.3f", comparison$differences))
  c(
    sprintf("Rows compared: %d of %d.", comparison$rows, comparison$expected),
    "",
    "| Column | Largest difference | Tolerance |",
    "|---|---|---|",
    sprintf("| %s | %s%s | %s |", names(limits), shown, ifelse(over, " (over)", ""), format(limits))
  )
}

format_case <- function(result) {
  s <- result$selections
  value <- function(x) vapply(x, format_param_value, character(1))
  lines <- c(
    paste("##", result$name),
    "",
    if (!is.null(result$note)) c(result$note, ""),
    format_table(result$results, tolerance),
    "",
    "| Measure | Rule | R | Python |",
    "|---|---|---|---|",
    sprintf("| %s | %s | %s | %s |", s$measure, s$rule, value(s$r), value(s$py)),
    "",
    sprintf(
      "ARI between the R and Python labels at the default selection (stability, 1se): %.3f",
      result$label_ari
    ),
    ""
  )
  if (!is.null(result$pipelines)) {
    lines <- c(
      lines,
      "Rows of preprocessing_results():",
      "",
      format_table(result$pipelines, c(n_resamples = 0, ari_stability = 0.05, ari_generalizability = 0.05)),
      ""
    )
  }
  lines
}

easy <- blobs(500, 0.8, 3)
results <- list(
  k_case("easy blobs", easy),
  k_case("hard blobs", blobs(500, 2.3, 3)),
  k_case("circles", datasets$make_circles(n_samples = 400L, factor = 0.5, noise = 0.05, random_state = 0L)[[1L]]),
  resolution_case(),
  hdbscan_case(),
  anchored_case(easy),
  randomized_case()
)
report <- c(
  "# R and Python parity report",
  "",
  sprintf(
    "Generated on %s with CARVE %s in R and carve %s in Python, %d resamples per configuration and random_state 0. The k cases use the light preset over k = %d to %d.",
    format(Sys.Date()), as.character(utils::packageVersion("CARVE")), pycarve$`__version__`,
    n_resamples, min(ks), max(ks)
  ),
  "",
  unlist(lapply(results, format_case))
)
writeLines(report, "data-raw/parity-report.md")
writeLines(report)
