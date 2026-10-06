# Compares the R package with the Python package on the same data and
# writes data-raw/parity-report.md. Run from code/carve-r:
#
#   Rscript data-raw/parity_check.R
#
# It loads the package from source with devtools, and needs reticulate and
# the Python environment in ../.venv, where carve is installed. The two
# languages draw different subsamples, so the results agree within
# tolerances, not exactly.

devtools::load_all(".", quiet = TRUE)
# normalizePath() on the interpreter would follow the symlink out of the
# virtual environment, so resolve the environment directory instead.
reticulate::use_python(file.path(normalizePath("../.venv"), "bin", "python"), required = TRUE)
datasets <- reticulate::import("sklearn.datasets")
np <- reticulate::import("numpy", convert = FALSE)
pycarve <- reticulate::import("carve")

ks <- 2:6
n_resamples <- 100L

# Largest allowed absolute difference between the R and the Python value
# of one configuration.
tolerance <- c(
  ari_stability = 0.05,
  ari_generalizability = 0.05,
  ari_average = 0.05,
  consensus_pac_stability = 0.03,
  consensus_gini_stability = 0.03,
  consensus_ce_stability = 0.03,
  accuracy_generalizability = 0.03
)

cases <- list(
  "easy blobs" = datasets$make_blobs(
    n_samples = 500L, n_features = 2L, centers = 3L, cluster_std = 0.8, random_state = 3L
  ),
  "hard blobs" = datasets$make_blobs(
    n_samples = 500L, n_features = 2L, centers = 3L, cluster_std = 2.3, random_state = 3L
  ),
  "circles" = datasets$make_circles(n_samples = 400L, factor = 0.5, noise = 0.05, random_state = 0L)
)

compare_case <- function(name, data) {
  X <- data[[1L]]
  r_fit <- carve(X, n_clusters = ks, n_resamples = n_resamples, random_state = 0)
  py_fit <- pycarve$CARVE(n_clusters = np$array(ks), n_resamples = n_resamples, random_state = 0L)$fit(X)
  columns <- c("method_label", "n_clusters", names(tolerance))
  # reticulate leaves pandas string columns as Python objects, so the label
  # column is rebuilt as an R character vector.
  py_results <- py_fit$estimator_results_[, columns]
  py_results$method_label <- unlist(py_fit$estimator_results_$method_label$tolist())
  joined <- merge(
    estimator_results(r_fit)[, columns],
    py_results,
    by = c("method_label", "n_clusters"),
    suffixes = c("_r", "_py")
  )
  differences <- vapply(names(tolerance), function(column) {
    max(abs(joined[[paste0(column, "_r")]] - joined[[paste0(column, "_py")]]))
  }, numeric(1))
  selections <- expand.grid(
    measure = c("stability", "generalizability", "average"),
    rule = c("max", "1se"),
    stringsAsFactors = FALSE
  )
  selections$k_r <- mapply(function(m, r) get_k(r_fit, measure = m, rule = r), selections$measure, selections$rule)
  selections$k_py <- mapply(
    function(m, r) as.integer(py_fit$get_k(measure = m, rule = r)),
    selections$measure, selections$rule
  )
  list(
    name = name,
    rows = nrow(joined),
    differences = differences,
    selections = selections,
    label_ari = adjusted_rand_index(get_labels(r_fit), as.integer(py_fit$get_labels()))
  )
}

format_case <- function(result) {
  over <- result$differences > tolerance
  c(
    paste("##", result$name),
    "",
    sprintf("Configurations compared: %d of %d.", result$rows, 3L * length(ks)),
    "",
    "| Column | Largest difference | Tolerance |",
    "|---|---|---|",
    sprintf(
      "| %s | %.3f%s | %.2f |",
      names(tolerance), result$differences, ifelse(over, " (over)", ""), tolerance
    ),
    "",
    "| Measure | Rule | k in R | k in Python |",
    "|---|---|---|---|",
    sprintf(
      "| %s | %s | %d | %d |",
      result$selections$measure, result$selections$rule,
      result$selections$k_r, result$selections$k_py
    ),
    "",
    sprintf(
      "ARI between the R and Python labels at the default selection (stability, 1se): %.3f",
      result$label_ari
    ),
    ""
  )
}

results <- Map(compare_case, names(cases), cases)
report <- c(
  "# R and Python parity report",
  "",
  sprintf(
    "Generated on %s with CARVE %s in R and carve %s in Python: the light preset, k from %d to %d, %d resamples, random_state 0.",
    format(Sys.Date()), as.character(utils::packageVersion("CARVE")), pycarve$`__version__`,
    min(ks), max(ks), n_resamples
  ),
  "",
  unlist(lapply(results, format_case))
)
writeLines(report, "data-raw/parity-report.md")
writeLines(report)
