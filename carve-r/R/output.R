# Console output of a run. Mirrors _output.py, through message() so that
# suppressMessages() silences it.

print_run_header <- function(X, sweep, n_resamples, subsample_ratio, estimator_grids,
                             n_jobs, random_state, verbose, mode = "default") {
  if (verbose < 2) {
    return(invisible(NULL))
  }
  total <- sum(vapply(estimator_grids, function(g) length(expand_param_grid(g$grid)), integer(1)))
  policy <- resolve_mode(mode)
  n_matrices <- total * (policy$run_stability + policy$run_generalizability)
  memory_gb <- n_matrices * as.numeric(nrow(X))^2 * 8 / 1e9
  line <- strrep("=", 60)
  message(paste(c(
    paste("[CARVE]", line),
    "[CARVE] Validation Settings:",
    sprintf("[CARVE] n_samples          : %d", nrow(X)),
    sprintf("[CARVE] n_features         : %d", ncol(X)),
    sprintf("[CARVE] sweep parameter    : %s", sweep@param),
    sprintf(
      "[CARVE] %-19s: [%s]",
      sweep@param,
      paste(vapply(sweep@values, format_param_value, character(1)), collapse = " ")
    ),
    sprintf("[CARVE] n_resamples        : %d", as.integer(n_resamples)),
    sprintf("[CARVE] subsample_ratio    : %s", format(subsample_ratio)),
    sprintf("[CARVE] n_jobs             : %s", if (is.null(n_jobs)) "NULL" else format(n_jobs)),
    sprintf("[CARVE] total configs      : %d", total),
    sprintf("[CARVE] consensus memory   : %.2f GB", memory_gb),
    sprintf("[CARVE] random_state       : %d", as.integer(random_state)),
    paste("[CARVE]", line),
    "",
    "[CARVE] Starting validation ...",
    ""
  ), collapse = "\n"))
}

print_run_footer <- function(results, verbose) {
  if (verbose < 2) {
    return(invisible(NULL))
  }
  message(sprintf("\n[CARVE] finished. evaluated %d estimator configurations.", nrow(results)))
}

log_config_progress <- function(config_idx, total_configs, estimator_name, params, record,
                                sweep_param, verbose) {
  if (verbose <= 0) {
    return(invisible(NULL))
  }
  value <- params[[sweep_param]]
  message(sprintf(
    "[CARVE] [%d/%d] est=%s %s=%s | ARI_stab=%.3f\u00b1%.3f  ARI_gen=%.3f\u00b1%.3f  ",
    as.integer(config_idx),
    as.integer(total_configs),
    estimator_name,
    sweep_param,
    if (is.null(value)) "?" else format(value),
    record$ari_stability,
    record$ari_stability_se,
    record$ari_generalizability,
    record$ari_generalizability_se
  ))
}
