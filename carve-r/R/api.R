# Public CARVE R6 class. Mirrors api.py.
#
# R6 chosen over S4 / reference classes because mutation semantics
# (`self$attr_ <- ...`) are a 1:1 match for Python's `self.attr_ = ...`.


#' CARVE estimator
#'
#' Stability- and generalizability-based validator for clustering
#' methods. Mirrors the Python [carve][https://pypi.org/project/carve/]
#' class field-for-field. See the package vignette for worked examples.
#'
#' @section Lifecycle:
#' 1. `CARVE$new(...)` — construct with hyperparameters.
#' 2. `$fit(X, ...)` — run resampled validation over the estimator grid.
#' 3. `$get_k()`, `$get_labels()`, `$get_estimator()` — query results.
#' 4. `$save(path)` / `CARVE$load(path)` — persist / restore.
#'
#' @section Selection arguments (`measure` / `rule` / `not_two`):
#' See [select_best_k()] for the full vocabulary. Common measures:
#' `"stability"`, `"generalizability"`, `"average"`, `"pac"`, `"gini"`,
#' `"ce"`, `"accuracy"`. Common rules: `"max"`, `"1se"`, `"quantile"`.
#'
#' @examples
#' set.seed(1)
#' X <- rbind(
#'   matrix(rnorm(120, 0, 0.4), ncol = 2),
#'   matrix(rnorm(120, 5, 0.4), ncol = 2),
#'   matrix(rnorm(120, c(2.5, 4), 0.4), ncol = 2)
#' )
#' carve <- CARVE$new(n_clusters = 2:4, n_resamples = 10, random_state = 1L,
#'                    estimator_param_grids = list(
#'                      list(type = "kmeans", grid = list(n_clusters = 2:4))
#'                    ))
#' carve$fit(X)
#' carve$get_k(measure = "stability", rule = "max")
#'
#' @name CARVE-class
#' @aliases CARVE
#' @docType class
#' @export
CARVE <- R6::R6Class(
  "CARVE",
  public = list(
    #' @field n_clusters Integer scalar or vector of candidate k values.
    n_clusters = NULL,
    #' @field n_resamples Number of resampling iterations per configuration.
    n_resamples = NULL,
    #' @field subsample_ratio Proportion of samples per subsample.
    subsample_ratio = NULL,
    #' @field estimator_param_grids `"light"`, `"full"`, or a user list.
    estimator_param_grids = NULL,
    #' @field normalization_options Optional list of normalization specs.
    normalization_options = NULL,
    #' @field dim_reduction_options Optional list of DR specs.
    dim_reduction_options = NULL,
    #' @field classifier Currently ignored; ranger RF is always used.
    classifier = NULL,
    #' @field n_trees Number of trees for the default random forest.
    n_trees = NULL,
    #' @field reference_labels Optional reference labels for alignment.
    reference_labels = NULL,
    #' @field n_jobs Number of parallel workers for the resample loop.
    n_jobs = NULL,
    #' @field random_state Integer seed, or `NULL`.
    random_state = NULL,
    #' @field verbose Integer verbosity level (0-2).
    verbose = NULL,

    #' @field X_ The input matrix passed to `$fit()` (cleared on `save(include_data=FALSE)`).
    X_ = NULL,
    #' @field estimator_results_ Data frame of per-configuration metrics.
    estimator_results_ = NULL,
    #' @field estimator_param_grids_ Resolved grids used in the fit.
    estimator_param_grids_ = NULL,
    #' @field preprocessing_results_ Reserved (randomize_preprocessing summary).
    preprocessing_results_ = NULL,
    #' @field consensus_matrices_ List of stability consensus matrices.
    consensus_matrices_ = NULL,
    #' @field consensus_generalizability_matrices_ List of generalizability consensus matrices.
    consensus_generalizability_matrices_ = NULL,
    #' @field stability_gini_scores_ Matrix (n_configs x n_samples) of Gini scores.
    stability_gini_scores_ = NULL,
    #' @field stability_ce_scores_ Matrix (n_configs x n_samples) of CE scores.
    stability_ce_scores_ = NULL,
    #' @field generalizability_scores_ List of per-config generalizability vectors.
    generalizability_scores_ = NULL,

    #' @description Construct a new CARVE estimator.
    #' @param n_clusters Integer scalar K (evaluates `2:K`) or integer vector.
    #' @param n_resamples Number of resamples per configuration.
    #' @param subsample_ratio Proportion of samples per subsample, in (0, 1).
    #' @param estimator_param_grids `"light"`, `"full"`, or a list of grid specs.
    #' @param normalization_options,dim_reduction_options Optional preprocessor specs.
    #' @param classifier Currently ignored; reserved for user-specified classifier.
    #' @param n_trees Number of trees in the default random forest.
    #' @param reference_labels Optional reference labels used for alignment.
    #' @param n_jobs Integer >= 1. When > 1, the resample loop runs in
    #'   parallel via [furrr::future_map()] under a scoped
    #'   [future::multisession()] plan; sequential when `n_jobs = 1L`.
    #' @param random_state Integer seed or `NULL`.
    #' @param verbose Integer verbosity (0 silent, 1 per-config, 2 +header/footer).
    initialize = function(n_clusters = 2:10,
                          n_resamples = 100L,
                          subsample_ratio = 0.8,
                          estimator_param_grids = "light",
                          normalization_options = NULL,
                          dim_reduction_options = NULL,
                          classifier = NULL,
                          n_trees = 100L,
                          reference_labels = NULL,
                          n_jobs = 1L,
                          random_state = NULL,
                          verbose = 0L) {
      self$n_clusters <- n_clusters
      self$n_resamples <- as.integer(n_resamples)
      self$subsample_ratio <- as.numeric(subsample_ratio)
      self$estimator_param_grids <- estimator_param_grids
      self$normalization_options <- normalization_options
      self$dim_reduction_options <- dim_reduction_options
      self$classifier <- classifier
      self$n_trees <- as.integer(n_trees)
      self$reference_labels <- reference_labels
      self$n_jobs <- as.integer(n_jobs)
      self$random_state <- random_state
      self$verbose <- as.integer(verbose)
    },

    #' @description Run CARVE validation on `X`.
    #' @param X Numeric matrix.
    #' @param reference_labels Optional reference labels (overrides field).
    #' @param randomize_preprocessing Logical; sample preprocessing per resample.
    #' @param show_progress If `TRUE`, emit a [progressr::progressor()]
    #'   sized `n_configs * n_resamples`. Wrap the call in
    #'   [progressr::with_progress()] to render a handler.
    #' @param mode `"default"`, `"stability"`, or `"generalizability"`.
    #' @param random_state Integer seed (overrides `$random_state` for this call).
    #' @return `self`, invisibly.
    fit = function(X,
                   reference_labels = NULL,
                   randomize_preprocessing = FALSE,
                   show_progress = FALSE,
                   mode = "default",
                   random_state = NULL) {
      policy <- resolve_mode(mode)
      if (!identical(policy$mode, "default")) {
        warning("Non-default mode is experimental and may break ",
                "downstream functionality.", call. = FALSE)
      }
      if (!is.null(self$classifier) && !identical(self$n_trees, 100L)) {
        warning("n_trees is ignored when a custom classifier is provided.",
                call. = FALSE)
      }

      X <- ensure_2d_matrix(X)
      storage.mode(X) <- "double"
      self$X_ <- X

      if (!is.null(reference_labels)) {
        ref <- reference_labels
        if (!is.integer(ref)) ref <- as.integer(factor(ref))
        self$reference_labels <- ref
      }

      # Resolve estimator grids.
      if (is.character(self$estimator_param_grids)) {
        preset <- match.arg(self$estimator_param_grids, c("light", "full"))
        n_clusters_arr <- coerce_n_clusters(self$n_clusters)
        grids <- default_estimator_grids(X, n_clusters_arr, preset = preset)
      } else {
        grids <- self$estimator_param_grids
        # Verify n_clusters consistency across user grids.
        n_clusters_arr <- grids[[1]]$grid$n_clusters
        for (g in grids[-1L]) {
          if (!identical(g$grid$n_clusters, n_clusters_arr)) {
            stop("All estimator parameter grids must contain the same ",
                 "n_clusters values.", call. = FALSE)
          }
        }
      }
      self$estimator_param_grids_ <- grids

      norm_options <- self$normalization_options %||% default_normalization_options()
      dr_options <- self$dim_reduction_options %||%
        default_dim_reduction_options(X, self$subsample_ratio)

      rs <- random_state %||% self$random_state

      out <- run_validation(
        X = X,
        estimator_grids = grids,
        n_resamples = self$n_resamples,
        subsample_ratio = self$subsample_ratio,
        normalization_options = norm_options,
        dim_reduction_options = dr_options,
        classifier = self$classifier,
        n_trees = self$n_trees,
        randomize_preprocessing = randomize_preprocessing,
        n_jobs = self$n_jobs,
        random_state = rs,
        show_progress = show_progress,
        mode = policy$mode,
        verbose = self$verbose
      )

      self$estimator_results_ <- out$estimator_results
      self$consensus_matrices_ <- out$consensus_matrices
      self$consensus_generalizability_matrices_ <-
        out$consensus_generalizability_matrices
      self$generalizability_scores_ <- out$generalizability_scores

      n_rows <- nrow(self$estimator_results_)

      # Stability-derived columns.
      if (policy$run_stability && !is.null(self$consensus_matrices_)) {
        cm <- compute_consensus_metrics(self$consensus_matrices_)
        self$stability_gini_scores_ <- do.call(rbind, cm$gini_list)
        self$stability_ce_scores_ <- do.call(rbind, cm$ce_list)
        self$estimator_results_$consensus_pac_stability <- cm$pac_list
        self$estimator_results_$consensus_gini_stability <-
          rowMeans(self$stability_gini_scores_, na.rm = TRUE)
        self$estimator_results_$consensus_ce_stability <-
          rowMeans(self$stability_ce_scores_, na.rm = TRUE)
      } else {
        self$stability_gini_scores_ <- NULL
        self$stability_ce_scores_ <- NULL
        self$estimator_results_$consensus_pac_stability <- rep(NaN, n_rows)
        self$estimator_results_$consensus_gini_stability <- rep(NaN, n_rows)
        self$estimator_results_$consensus_ce_stability <- rep(NaN, n_rows)
      }

      # Generalizability-derived column.
      if (policy$run_generalizability &&
          !is.null(self$generalizability_scores_)) {
        gen_mat <- do.call(rbind, self$generalizability_scores_)
        self$estimator_results_$accuracy_generalizability <-
          rowMeans(gen_mat, na.rm = TRUE)
      } else {
        self$estimator_results_$accuracy_generalizability <- rep(NaN, n_rows)
      }

      invisible(self)
    },

    #' @description Return the best number of clusters.
    #' @param measure,rule,not_two See [select_best_k()].
    get_k = function(measure = "generalizability",
                     rule = "1se",
                     not_two = FALSE) {
      private$require_fitted()
      select_best_k(self$estimator_results_, measure = measure,
                    rule = rule, not_two = not_two)
    },

    #' @description Return the best estimator spec.
    #' @param measure,rule,not_two See [select_best_estimator()].
    get_estimator = function(measure = "generalizability",
                             rule = "1se",
                             not_two = FALSE) {
      private$require_fitted()
      select_best_estimator(self$estimator_results_, measure = measure,
                            rule = rule, not_two = not_two)
    },

    #' @description Return clustering labels from the selected consensus matrix.
    #' @param measure,rule,not_two Selection knobs.
    #' @param k Optional fixed cluster count.
    #' @param mode `"default"` (stability matrix) or `"generalizability"`.
    get_labels = function(measure = "generalizability",
                          rule = "1se",
                          k = NULL,
                          not_two = FALSE,
                          mode = "default") {
      private$require_fitted()
      mode <- match.arg(mode, c("default", "generalizability"))
      policy <- resolve_mode(mode)
      df <- self$estimator_results_

      if (is.null(k)) {
        row <- select_best_row_by_rule(df, measure = measure, rule = rule,
                                       not_two = not_two)
        best_idx <- as.integer(rownames(row))
        k <- as.integer(row$n_clusters)
      } else {
        k <- as.integer(k)
        df_k <- df[df$n_clusters == k, , drop = FALSE]
        if (nrow(df_k) == 0L) {
          stop(sprintf("No configurations found for k=%d.", k), call. = FALSE)
        }
        row <- select_best_row_by_rule(df_k, measure = measure, rule = rule)
        best_idx <- as.integer(rownames(row))
      }

      # Retrieve the appropriate consensus matrix.
      M_raw <- if (policy$run_stability && !is.null(self$consensus_matrices_)) {
        self$consensus_matrices_[[best_idx]]
      } else if (policy$run_generalizability &&
                 !is.null(self$consensus_generalizability_matrices_)) {
        self$consensus_generalizability_matrices_[[best_idx]]
      } else {
        stop("mode must be 'default' or 'generalizability'.", call. = FALSE)
      }
      if (is.null(M_raw)) {
        stop(sprintf(
          "Consensus matrix not available for mode=%s. This run likely ",
          sQuote(mode)), call. = FALSE)
      }

      # Symmetrize, set diagonal to 1, clip to [0, 1], and treat NaN as 0.5
      # (matches Python's nan_to_num behavior).
      M <- as.matrix(M_raw)
      S <- 0.5 * (M + t(M))
      diag(S) <- 1
      S <- pmin(pmax(S, 0), 1)
      if (any(is.nan(S))) {
        S[is.nan(S)] <- 0.5
        diag(S) <- 1
      }

      # Convert to distance and cluster via average linkage with
      # precomputed distances (matches sklearn's AgglomerativeClustering).
      D <- 1 - S
      diag(D) <- 0
      hc <- stats::hclust(stats::as.dist(D), method = "average")
      labels <- as.integer(stats::cutree(hc, k = k))

      # Reference-label alignment (same logic as Python).
      cur_k <- length(unique(labels))
      ref <- self$reference_labels
      ref_k <- if (is.null(ref)) NULL else length(unique(ref))
      if (is.null(ref) || !identical(ref_k, cur_k)) {
        self$reference_labels <- labels
      } else {
        labels <- align_cluster_labels(ref, labels)
      }
      as.integer(labels)
    },

    #' @description Persist a fitted CARVE instance via [saveRDS()].
    #' @param path Destination file path (conventional extension `.carve`).
    #' @param include_data If `FALSE` (default), clear `X_` before saving.
    save = function(path, include_data = FALSE) {
      private$require_fitted()
      dir.create(dirname(path), showWarnings = FALSE, recursive = TRUE)
      if (include_data) {
        saveRDS(self$clone(deep = TRUE), file = path)
      } else {
        clone_ <- self$clone(deep = TRUE)
        clone_$X_ <- NULL
        saveRDS(clone_, file = path)
      }
      invisible(path)
    },

    #' @description Plot a metric across `n_clusters`, one line per
    #'   estimator configuration, with an SE ribbon and a dashed vertical
    #'   line at the selected k. Returns a [ggplot2::ggplot] object.
    #' @param measure,rule,not_two Selection knobs (see [select_best_k()]).
    #' @param title,xlabel,ylabel Optional plot labels.
    #' @param legend Whether to draw the estimator legend.
    #' @param palette Discrete palette name. Currently only `"Accent"` is
    #'   honored; other values fall back to a qualitative HCL palette.
    plot_metric_over_n_clusters = function(measure = "generalizability",
                                            rule = "1se",
                                            not_two = FALSE,
                                            title = NULL,
                                            xlabel = NULL,
                                            ylabel = NULL,
                                            legend = TRUE,
                                            palette = "Accent") {
      private$require_fitted()
      .plot_metric_over_n_clusters_df(
        self$estimator_results_, measure = measure, rule = rule,
        not_two = not_two, title = title, xlabel = xlabel, ylabel = ylabel,
        legend = legend, palette = palette
      )
    },

    #' @description Plot the consensus matrix of the selected configuration
    #'   with a top cluster-color band. Returns a [patchwork::patchwork] object.
    #' @param measure,rule,not_two Selection knobs.
    #' @param mode `"default"` / `"stability"` (consensus_matrices_) or
    #'   `"generalizability"` (consensus_generalizability_matrices_).
    #' @param k Optional fixed cluster count.
    #' @param cmap Continuous colormap. `"viridis"` or anything else
    #'   (which falls back to white→black).
    #' @param palette Discrete palette for the cluster band.
    #' @param colorbar Whether to draw the consensus colorbar.
    #' @param colorbar_label,title Optional plot labels.
    plot_consensus_matrix = function(measure = "generalizability",
                                      rule = "1se",
                                      not_two = FALSE,
                                      mode = c("default", "stability",
                                               "generalizability"),
                                      k = NULL,
                                      cmap = "viridis",
                                      palette = "Accent",
                                      colorbar = TRUE,
                                      colorbar_label = "Consensus",
                                      title = NULL) {
      private$require_fitted()
      mode <- match.arg(mode)
      if (mode %in% c("default", "stability")) {
        matrices <- self$consensus_matrices_
        labels_mode <- "default"
      } else {
        matrices <- self$consensus_generalizability_matrices_
        labels_mode <- "generalizability"
      }
      if (is.null(matrices)) {
        stop(sprintf("Consensus matrices for mode=%s are not available.",
                     sQuote(mode)), call. = FALSE)
      }

      df <- self$estimator_results_
      if (is.null(k)) {
        row <- select_best_row_by_rule(df, measure = measure, rule = rule,
                                       not_two = not_two)
      } else {
        df_k <- df[df$n_clusters == as.integer(k), , drop = FALSE]
        if (nrow(df_k) == 0L) {
          stop(sprintf("No configurations found for k=%d.", as.integer(k)),
               call. = FALSE)
        }
        row <- select_best_row_by_rule(df_k, measure = measure, rule = rule)
      }
      best_idx <- as.integer(rownames(row))
      selected_k <- as.integer(row$n_clusters)
      M <- matrices[[best_idx]]
      if (is.null(M)) {
        stop(sprintf("Selected consensus matrix not available for mode=%s.",
                     sQuote(mode)), call. = FALSE)
      }
      labels <- self$get_labels(measure = measure, rule = rule,
                                k = selected_k, mode = labels_mode)
      .plot_consensus_matrix_mat(M, labels, cmap = cmap, palette = palette,
                                 colorbar = colorbar,
                                 colorbar_label = colorbar_label,
                                 title = title)
    },

    #' @description Per-cluster boxplot of `source` scores
    #'   (`"gini"` / `"ce"` / `"accuracy"`). Returns a [ggplot2::ggplot] object.
    #' @param source Score source.
    #' @param measure,rule,not_two,mode,k Selection knobs.
    #' @param order Optional explicit cluster ordering.
    #' @param palette Discrete palette name.
    #' @param showfliers Show boxplot outliers.
    #' @param width Box width.
    #' @param title,xlabel,ylabel Optional labels.
    #' @param annotation `TRUE` to auto-generate, a string for verbatim,
    #'   `FALSE` to suppress.
    #' @param rotation Tick label rotation in degrees.
    plot_cluster_boxplot = function(source = c("gini", "ce", "accuracy"),
                                     measure = "generalizability",
                                     rule = "1se",
                                     not_two = FALSE,
                                     mode = c("default", "stability",
                                              "generalizability"),
                                     k = NULL,
                                     order = NULL,
                                     palette = "Accent",
                                     showfliers = FALSE,
                                     width = 0.75,
                                     title = NULL,
                                     xlabel = "Cluster",
                                     ylabel = NULL,
                                     annotation = TRUE,
                                     rotation = NULL) {
      private$require_fitted()
      source <- match.arg(source)
      mode <- match.arg(mode)
      sel <- private$select_score_and_labels(source, measure, rule, not_two,
                                              mode, k)
      if (is.null(ylabel)) ylabel <- sel$default_ylabel
      ann_text <- private$resolve_annotation(annotation, measure, rule, k,
                                              sel$row, sel$selected_k)
      .plot_cluster_boxplot_scores(
        sel$scores, sel$labels, order = order, palette = palette,
        showfliers = showfliers, width = width, title = title,
        xlabel = xlabel, ylabel = ylabel, annotation = ann_text,
        rotation = rotation
      )
    },

    #' @description Per-cluster violin plot. See `$plot_cluster_boxplot()`
    #'   for the shared selection arguments. Returns a [ggplot2::ggplot] object.
    #' @param source,measure,rule,not_two,mode,k Selection knobs.
    #' @param order,palette,title,xlabel,ylabel,annotation,rotation As in
    #'   `$plot_cluster_boxplot()`.
    #' @param density_norm `"width"`, `"area"`, or `"count"`.
    #' @param stripplot Overlay individual points.
    #' @param jitter Strip-plot jitter width (logical or numeric).
    #' @param size,alpha Strip-plot marker size and alpha.
    #' @param inner Inner annotation: `"box"`, `"quartile"`, or `"none"`.
    plot_cluster_violin = function(source = c("gini", "ce", "accuracy"),
                                    measure = "generalizability",
                                    rule = "1se",
                                    not_two = FALSE,
                                    mode = c("default", "stability",
                                             "generalizability"),
                                    k = NULL,
                                    order = NULL,
                                    palette = "Accent",
                                    density_norm = c("width", "area", "count"),
                                    stripplot = TRUE,
                                    jitter = TRUE,
                                    size = 1.0,
                                    alpha = 0.22,
                                    inner = c("box", "quartile", "none"),
                                    title = NULL,
                                    xlabel = "Cluster",
                                    ylabel = NULL,
                                    annotation = TRUE,
                                    rotation = NULL) {
      private$require_fitted()
      source <- match.arg(source)
      mode <- match.arg(mode)
      density_norm <- match.arg(density_norm)
      inner <- match.arg(inner)
      sel <- private$select_score_and_labels(source, measure, rule, not_two,
                                              mode, k)
      if (is.null(ylabel)) ylabel <- sel$default_ylabel
      ann_text <- private$resolve_annotation(annotation, measure, rule, k,
                                              sel$row, sel$selected_k)
      .plot_cluster_violin_scores(
        sel$scores, sel$labels, order = order, palette = palette,
        density_norm = density_norm, stripplot = stripplot, jitter = jitter,
        size = size, alpha = alpha, inner = inner, title = title,
        xlabel = xlabel, ylabel = ylabel, annotation = ann_text,
        rotation = rotation
      )
    },

    #' @description 2D scatter with score-encoded opacity and point size.
    #'   PCA falls back to 2 components when `embedding` is omitted and
    #'   `X` has more than 2 columns. Returns a [ggplot2::ggplot] object.
    #' @param source,measure,rule,not_two,mode,k Selection knobs.
    #' @param X Optional alternative data matrix (defaults to `self$X_`).
    #' @param embedding Optional pre-computed 2D embedding.
    #' @param palette Discrete palette name.
    #' @param alpha_range,size_range Two-vectors `c(high_score, low_score)`.
    #' @param sort_order Plot transparent points first.
    #' @param legend Show the cluster legend.
    #' @param annotation,annotation_style See `$plot_cluster_boxplot()`.
    #'   `annotation_style="legend"` appends to the cluster legend title;
    #'   `"box"` places a free-floating label.
    #' @param title,xlabel,ylabel Optional labels.
    #' @param frameon Draw axis spines.
    plot_cluster_scatter = function(source = c("gini", "ce", "accuracy"),
                                     measure = "generalizability",
                                     rule = "1se",
                                     not_two = FALSE,
                                     mode = c("default", "stability",
                                              "generalizability"),
                                     k = NULL,
                                     X = NULL,
                                     embedding = NULL,
                                     palette = "Accent",
                                     alpha_range = c(0.45, 0.9),
                                     size_range = c(1.5, 5.0),
                                     sort_order = TRUE,
                                     legend = TRUE,
                                     annotation = TRUE,
                                     annotation_style = c("legend", "box"),
                                     title = NULL,
                                     xlabel = "Component 1",
                                     ylabel = "Component 2",
                                     frameon = FALSE) {
      private$require_fitted()
      source <- match.arg(source)
      mode <- match.arg(mode)
      annotation_style <- match.arg(annotation_style)
      sel <- private$select_score_and_labels(source, measure, rule, not_two,
                                              mode, k)
      data <- if (is.null(X)) self$X_ else ensure_2d_matrix(X)
      if (is.null(data)) {
        stop("Raw data are not available on this instance. ",
             "Pass X=... explicitly (e.g., after loading a model saved ",
             "with include_data=FALSE).", call. = FALSE)
      }
      tight <- identical(annotation_style, "legend")
      ann_text <- private$resolve_annotation(annotation, measure, rule, k,
                                              sel$row, sel$selected_k,
                                              tight_layout = tight)
      .plot_cluster_scatter_xy(
        data, sel$labels, sel$scores, embedding = embedding,
        palette = palette, alpha_range = alpha_range, size_range = size_range,
        sort_order = sort_order, legend = legend, annotation = ann_text,
        annotation_style = annotation_style, title = title,
        scores_name = sel$scores_name, xlabel = xlabel, ylabel = ylabel,
        frameon = frameon
      )
    },

    #' @description Diagnostic 2D scatter: shape encodes cluster, color
    #'   encodes per-sample score. Returns a [ggplot2::ggplot] object.
    #' @param source,measure,rule,not_two,mode,k Selection knobs.
    #' @param X,embedding Data matrix and optional 2D embedding.
    #' @param cmap Continuous colormap name. `"Greens_r"` (default) /
    #'   `"Greens"` / fallback `"viridis"`.
    #' @param alpha_encoding Reinforce score encoding via alpha.
    #' @param alpha_range Two-vector `c(high_score, low_score)`.
    #' @param marker_size Point size.
    #' @param markers Integer vector of `pch` codes; cycled if too few.
    #' @param sort_order Stable points first; unstable on top.
    #' @param legend,colorbar Whether to draw shape legend / colorbar.
    #' @param colorbar_label Colorbar label (defaults to `scores_name`).
    #' @param annotation,annotation_style As in `$plot_cluster_scatter()`.
    #' @param title,xlabel,ylabel,frameon See `$plot_cluster_scatter()`.
    plot_diagnostic_scatter = function(source = c("gini", "ce", "accuracy"),
                                        measure = "generalizability",
                                        rule = "1se",
                                        not_two = FALSE,
                                        mode = c("default", "stability",
                                                 "generalizability"),
                                        k = NULL,
                                        X = NULL,
                                        embedding = NULL,
                                        cmap = "Greens_r",
                                        alpha_encoding = TRUE,
                                        alpha_range = c(0.3, 1.0),
                                        marker_size = 2.0,
                                        markers = NULL,
                                        sort_order = TRUE,
                                        legend = TRUE,
                                        colorbar = TRUE,
                                        colorbar_label = NULL,
                                        annotation = TRUE,
                                        annotation_style = c("legend", "box"),
                                        title = NULL,
                                        xlabel = "Component 1",
                                        ylabel = "Component 2",
                                        frameon = FALSE) {
      private$require_fitted()
      source <- match.arg(source)
      mode <- match.arg(mode)
      annotation_style <- match.arg(annotation_style)
      sel <- private$select_score_and_labels(source, measure, rule, not_two,
                                              mode, k)
      data <- if (is.null(X)) self$X_ else ensure_2d_matrix(X)
      if (is.null(data)) {
        stop("Raw data are not available on this instance. ",
             "Pass X=... explicitly (e.g., after loading a model saved ",
             "with include_data=FALSE).", call. = FALSE)
      }
      tight <- identical(annotation_style, "legend")
      ann_text <- private$resolve_annotation(annotation, measure, rule, k,
                                              sel$row, sel$selected_k,
                                              tight_layout = tight)
      .plot_diagnostic_scatter_xy(
        data, sel$labels, sel$scores, embedding = embedding,
        cmap = cmap, alpha_encoding = alpha_encoding,
        alpha_range = alpha_range, marker_size = marker_size,
        markers = markers, sort_order = sort_order, legend = legend,
        colorbar = colorbar, colorbar_label = colorbar_label,
        annotation = ann_text, annotation_style = annotation_style,
        title = title, scores_name = sel$scores_name,
        xlabel = xlabel, ylabel = ylabel, frameon = frameon
      )
    }
  ),

  private = list(
    require_fitted = function() {
      if (is.null(self$estimator_results_)) {
        stop("CARVE instance has not been fitted yet. Call $fit(X) first.",
             call. = FALSE)
      }
    },

    # Resolve the (best_idx, scores, labels, default_ylabel, scores_name)
    # tuple shared by the four cluster-level plotting methods.
    select_score_and_labels = function(source, measure, rule, not_two,
                                       mode, k) {
      df <- self$estimator_results_
      if (is.null(k)) {
        row <- select_best_row_by_rule(df, measure = measure, rule = rule,
                                       not_two = not_two)
      } else {
        df_k <- df[df$n_clusters == as.integer(k), , drop = FALSE]
        if (nrow(df_k) == 0L) {
          stop(sprintf("No configurations found for k=%d.", as.integer(k)),
               call. = FALSE)
        }
        row <- select_best_row_by_rule(df_k, measure = measure, rule = rule)
      }
      best_idx <- as.integer(rownames(row))
      selected_k <- as.integer(row$n_clusters)

      if (identical(source, "gini")) {
        if (is.null(self$stability_gini_scores_)) {
          stop("Gini stability scores are not available for this run.",
               call. = FALSE)
        }
        scores <- as.numeric(self$stability_gini_scores_[best_idx, ])
        default_ylabel <- "Cluster Stability (Gini)"
        scores_name <- "Gini Stability"
      } else if (identical(source, "ce")) {
        if (is.null(self$stability_ce_scores_)) {
          stop("CE stability scores are not available for this run.",
               call. = FALSE)
        }
        scores <- as.numeric(self$stability_ce_scores_[best_idx, ])
        default_ylabel <- "Cluster Stability (CE)"
        scores_name <- "CE Stability"
      } else {
        if (is.null(self$generalizability_scores_)) {
          stop("Generalizability scores are not available for this run.",
               call. = FALSE)
        }
        scores <- as.numeric(self$generalizability_scores_[[best_idx]])
        default_ylabel <- "Cluster Generalizability"
        scores_name <- "Generalizability"
      }

      labels_mode <- if (mode %in% c("default", "stability")) {
        "default"
      } else "generalizability"
      labels <- self$get_labels(measure = measure, rule = rule,
                                k = selected_k, mode = labels_mode)

      list(row = row, best_idx = best_idx, selected_k = selected_k,
           scores = scores, labels = labels,
           default_ylabel = default_ylabel, scores_name = scores_name)
    },

    # Mirror Python's `annotation: bool | str = True` resolution.
    resolve_annotation = function(annotation, measure, rule, k, row,
                                  selected_k, tight_layout = FALSE) {
      if (isTRUE(annotation)) {
        .get_annotation(measure = measure, rule = rule, k = k,
                        results_df = self$estimator_results_,
                        row = row, selected_k = selected_k,
                        tight_layout = tight_layout)
      } else if (is.character(annotation)) {
        annotation
      } else {
        NULL
      }
    }
  )
)


# Attach a "static" loader to the class generator so calls read like
# `CARVE$load(path)`. R6 generators are environments and allow this
# assignment; instances do not inherit from it.
CARVE$load <- function(path) {
  if (!file.exists(path)) {
    stop("No such file: ", sQuote(path), call. = FALSE)
  }
  obj <- readRDS(path)
  if (!inherits(obj, "CARVE")) {
    stop("Expected a CARVE instance; got ", sQuote(class(obj)[1L]), ".",
         call. = FALSE)
  }
  obj
}
