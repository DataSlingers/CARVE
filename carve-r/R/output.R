# Console output helpers. Mirrors _output.py.
#
# Verbosity levels follow the Python convention:
#   0 = silent
#   1 = per-configuration progress
#   2 = header, per-config, footer


#' @noRd
print_run_header <- function(X, n_clusters, n_resamples, subsample_ratio,
                             estimator_grids, n_jobs, randomize_preprocessing,
                             random_state, verbose) {
  if (verbose < 2L) return(invisible(NULL))

  total_configs <- sum(vapply(
    estimator_grids,
    function(g) length(expand_grid_spec(g$grid)),
    integer(1L)
  ))
  line <- strrep("=", 60L)

  message("[CARVE] ", line)
  message("[CARVE] Validation Settings:")
  message("[CARVE] n_samples          : ", nrow(X))
  message("[CARVE] n_features         : ", ncol(X))
  message("[CARVE] n_clusters         : ",
          paste(n_clusters, collapse = ", "))
  message("[CARVE] n_resamples        : ", n_resamples)
  message("[CARVE] subsample_ratio    : ", subsample_ratio)
  message("[CARVE] n_jobs             : ", n_jobs)
  message("[CARVE] total configs      : ", total_configs)
  message("[CARVE] randomize_preproc  : ", randomize_preprocessing)
  message("[CARVE] random_state       : ",
          if (is.null(random_state)) "NULL" else random_state)
  message("[CARVE] ", line)
  message("")
  message("[CARVE] Starting validation ...")
  message("")
}


#' @noRd
print_run_footer <- function(estimator_df, verbose) {
  if (verbose < 2L) return(invisible(NULL))
  message("")
  message("[CARVE] finished. evaluated ", nrow(estimator_df),
          " estimator configurations.")
}


#' @noRd
log_config_progress <- function(config_idx, total_configs, type, params,
                                record, verbose = 0L) {
  if (verbose <= 0L) return(invisible(NULL))
  n_clusters <- params$n_clusters %||% "?"
  msg <- sprintf(
    "[CARVE] [%d/%d] est=%s n_clusters=%s | ARI_stab=%.3f\u00b1%.3f  ARI_gen=%.3f\u00b1%.3f",
    config_idx, total_configs, type, as.character(n_clusters),
    record$ari_stability %||% NaN, record$ari_stability_se %||% NaN,
    record$ari_generalizability %||% NaN, record$ari_generalizability_se %||% NaN
  )
  message(msg)
}
