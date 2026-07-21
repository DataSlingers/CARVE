# Plotting primitives. Mirrors _plotting.py.
#
# Each function below is a pure, R6-free helper that takes raw inputs
# (data frames / matrices / labels) and returns a ggplot (or patchwork)
# object. The public R6 methods on the CARVE class (in api.R) select
# the best configuration, extract the right score / matrix, and forward
# to these helpers.


# --- Palettes ----------------------------------------------------------------

.discrete_palette <- function(n, name = "Tableau 10") {
  full <- tryCatch(
    grDevices::palette.colors(palette = name),
    error = function(e) {
      stop(sprintf(
        "Unknown palette '%s'. Available palettes: %s",
        name, paste(grDevices::palette.pals(), collapse = ", ")
      ), call. = FALSE)
    }
  )
  unname(full[((seq_len(n) - 1L) %% length(full)) + 1L])
}


# --- Label and annotation helpers -------------------------------------------

# Which columns of estimator_results_ are metric columns (to exclude
# from the human-readable estimator label)?
.metric_cols <- function(col_names) {
  patterns <- c("ari_", "consensus_", "accuracy_", "_se", "_upper", "_lower")
  keep <- vapply(col_names, function(n) {
    any(vapply(patterns, function(p) grepl(p, n, fixed = TRUE), logical(1L)))
  }, logical(1L))
  col_names[keep]
}


# Build "estimator, param=value, ..." like _build_estimator_label in Python.
.build_estimator_label <- function(row, exclude_cols, tight_layout = FALSE) {
  parts <- list(as.character(row[["estimator"]]))
  for (col in setdiff(names(row), c(exclude_cols, "estimator"))) {
    val <- row[[col]]
    if (is.null(val) || length(val) == 0L) next
    if (is.na(val)) next
    if (is.character(val) && identical(val, "_NA_")) next
    if (is.numeric(val)) {
      if (isTRUE(val == as.integer(val))) {
        parts[[length(parts) + 1L]] <- sprintf("%s=%d", col, as.integer(val))
      } else {
        parts[[length(parts) + 1L]] <- sprintf("%s=%.3g", col, val)
      }
    } else {
      parts[[length(parts) + 1L]] <- sprintf("%s=%s", col, as.character(val))
    }
  }
  paste(unlist(parts), collapse = if (tight_layout) "\n" else ", ")
}


.wrap_label <- function(label, width = 40L) {
  paste(strwrap(label, width = width), collapse = "\n")
}


# Two-line annotation describing the selected model. Mirrors _get_annotation.
.get_annotation <- function(measure, rule, k, results_df, row, selected_k,
                            tight_layout = FALSE) {
  metric_cols <- .metric_cols(names(results_df))
  exclude_cols <- unique(c(metric_cols, "estimator", "n_clusters", "index"))
  model_label <- .build_estimator_label(row, exclude_cols,
                                        tight_layout = tight_layout)
  rule_str <- if (identical(rule, "1se")) "1-SE" else tools::toTitleCase(rule)
  measure_str <- tools::toTitleCase(gsub("_", " ", measure))
  k_suffix <- if (is.null(k)) {
    sprintf("(k = %d)", selected_k)
  } else {
    sprintf("(k = %d, fixed)", selected_k)
  }
  txt <- sprintf("%s %s\n%s, %s rule", model_label, k_suffix,
                 measure_str, rule_str)
  if (tight_layout) txt <- paste0(txt, "\n")
  txt
}


# Stable-sorted per-cluster score groups; mirrors _prepare_cluster_score_groups.
.prepare_cluster_score_groups <- function(scores, labels, order = NULL) {
  scores <- as.numeric(scores)
  labels <- as.vector(labels)
  if (length(scores) != length(labels)) {
    stop("scores and labels must have matching length.", call. = FALSE)
  }
  mask <- is.finite(scores)
  scores <- scores[mask]
  labels <- labels[mask]
  if (length(scores) == 0L) {
    stop("No finite scores available for plotting.", call. = FALSE)
  }

  if (is.null(order)) {
    uniq <- unique(labels)
    ord_vals <- tryCatch(
      uniq[order(suppressWarnings(as.numeric(uniq)))],
      warning = function(w) uniq[order(as.character(uniq))],
      error = function(e) uniq[order(as.character(uniq))]
    )
  } else {
    ord_vals <- order
  }

  groups <- list()
  kept <- vector(mode = class(ord_vals)[1L], length = 0L)
  for (lab in ord_vals) {
    vals <- scores[labels == lab]
    if (length(vals) > 0L) {
      groups[[length(groups) + 1L]] <- vals
      kept <- c(kept, lab)
    }
  }
  if (length(groups) == 0L) {
    stop("No cluster values found for the provided order.", call. = FALSE)
  }
  list(groups = groups, order = kept)
}


# Unique labels sorted numerically when possible, lexically otherwise.
.sort_unique_labels <- function(labels) {
  uniq <- unique(labels)
  tryCatch(
    uniq[order(suppressWarnings(as.numeric(uniq)))],
    warning = function(w) uniq[order(as.character(uniq))],
    error = function(e) uniq[order(as.character(uniq))]
  )
}


# 2D PCA embedding for scatter fallbacks.
.pca_2d <- function(X) {
  pc <- stats::prcomp(X, center = TRUE, scale. = FALSE, rank. = 2L)
  pc$x[, 1:2, drop = FALSE]
}


# --- 1) Metric over n_clusters ----------------------------------------------

.plot_metric_over_n_clusters_df <- function(results_df,
                                            measure = "generalizability",
                                            rule = "1se",
                                            not_two = FALSE,
                                            title = NULL,
                                            xlabel = NULL,
                                            ylabel = NULL,
                                            legend = TRUE,
                                            palette = "Tableau 10") {
  if (is.null(results_df) || nrow(results_df) == 0L) {
    stop("Results DataFrame is empty.", call. = FALSE)
  }
  if (!measure %in% names(MEASURE_MAP)) {
    stop(sprintf("Measure '%s' not found. Valid options: %s",
                 measure, paste(names(MEASURE_MAP), collapse = ", ")),
         call. = FALSE)
  }
  measure_col <- unname(MEASURE_MAP[[measure]])
  se_col <- paste0(measure_col, "_se")
  if (!measure_col %in% names(results_df)) {
    stop(sprintf("Metric column '%s' not found in results_df.", measure_col),
         call. = FALSE)
  }
  has_se <- se_col %in% names(results_df)

  metric_cols <- .metric_cols(names(results_df))
  exclude_cols <- unique(c(metric_cols, "estimator", "n_clusters", "index"))
  group_cols <- setdiff(names(results_df), c(exclude_cols, "n_clusters"))

  df <- results_df
  if (length(group_cols) > 0L) {
    key_parts <- lapply(group_cols, function(cn) {
      ifelse(is.na(df[[cn]]), "_NA_", as.character(df[[cn]]))
    })
    df$.group_key <- do.call(paste,
                             c(list(as.character(df$estimator)),
                               key_parts, list(sep = "|")))
  } else {
    df$.group_key <- as.character(df$estimator)
  }

  unique_keys <- unique(df$.group_key)
  label_map <- vapply(unique_keys, function(kk) {
    first_row <- df[df$.group_key == kk, , drop = FALSE][1L, , drop = FALSE]
    .build_estimator_label(first_row, c(exclude_cols, ".group_key"))
  }, character(1L))
  label_map <- vapply(label_map, .wrap_label, character(1L), width = 40L)
  df$.group_label <- factor(label_map[df$.group_key], levels = unname(label_map))

  df$.metric <- df[[measure_col]]
  df$.se <- if (has_se) df[[se_col]] else 0

  colors <- .discrete_palette(length(unique_keys), palette)

  p <- ggplot2::ggplot(
    df,
    ggplot2::aes(x = .data$n_clusters, y = .data$.metric,
                 color = .data$.group_label, group = .data$.group_label)
  )
  if (has_se) {
    p <- p + ggplot2::geom_errorbar(
      ggplot2::aes(ymin = .data$.metric - .data$.se,
                   ymax = .data$.metric + .data$.se),
      width = 0.15, linewidth = 0.55, alpha = 0.8
    )
  }
  p <- p +
    ggplot2::geom_line(linewidth = 0.9, alpha = 0.85) +
    ggplot2::geom_point(size = 2.3, alpha = 0.9) +
    ggplot2::scale_color_manual(values = colors, name = "Estimators")

  # Vertical dashed line at selected k.
  best_k <- tryCatch(
    select_best_k(results_df, measure = measure, rule = rule,
                  not_two = not_two),
    error = function(e) NULL
  )
  if (!is.null(best_k)) {
    rule_str <- if (identical(rule, "1se")) "1-SE" else tools::toTitleCase(rule)
    ks <- sort(unique(df$n_clusters))
    hjust <- if (best_k > stats::median(ks)) 1.05 else -0.05
    p <- p +
      ggplot2::geom_vline(xintercept = best_k, linetype = "dashed",
                          color = "gray40", linewidth = 0.7, alpha = 0.7) +
      ggplot2::annotate("text", x = best_k, y = Inf,
                        label = sprintf("Selected k (%s): %d", rule_str, best_k),
                        vjust = 1.4, hjust = hjust,
                        color = "gray30", size = 3.2)
  }

  if (is.null(xlabel)) xlabel <- "Number of Clusters (k)"
  if (is.null(ylabel)) {
    ylabel <- tools::toTitleCase(gsub("_", " ", measure_col))
    ylabel <- gsub("Ari", "ARI", ylabel, fixed = TRUE)
  }
  p <- p +
    ggplot2::labs(x = xlabel, y = ylabel, title = title) +
    ggplot2::theme_minimal(base_size = 11) +
    ggplot2::theme(panel.grid.minor = ggplot2::element_blank())

  ks <- sort(unique(df$n_clusters))
  if (length(ks) <= 20L) {
    p <- p + ggplot2::scale_x_continuous(breaks = ks)
  }
  if (!isTRUE(legend)) p <- p + ggplot2::guides(color = "none")

  p
}


# --- 2) Consensus matrix ----------------------------------------------------

.plot_consensus_matrix_mat <- function(consensus_matrix, labels,
                                       cmap = "viridis", palette = "Tableau 10",
                                       colorbar = TRUE,
                                       colorbar_label = "Consensus",
                                       title = NULL) {
  M <- as.matrix(consensus_matrix)
  labels <- as.vector(labels)
  if (nrow(M) != ncol(M)) {
    stop("consensus_matrix must be a square 2D array.", call. = FALSE)
  }
  if (length(labels) != nrow(M)) {
    stop("consensus_matrix and labels must have matching first dimension.",
         call. = FALSE)
  }

  S <- 0.5 * (M + t(M))
  S[is.nan(S)] <- 0.5
  S <- pmin(pmax(S, 0), 1)
  diag(S) <- 1

  ord <- order(labels)  # stable by default in R
  S_ord <- S[ord, ord]
  labels_ord <- labels[ord]
  n <- length(labels_ord)

  df <- data.frame(
    i = rep(seq_len(n), times = n),
    j = rep(seq_len(n), each = n),
    v = as.vector(S_ord)
  )

  p_main <- ggplot2::ggplot(df,
      ggplot2::aes(x = .data$j, y = .data$i, fill = .data$v)) +
    ggplot2::geom_raster() +
    ggplot2::coord_fixed(expand = FALSE) +
    ggplot2::scale_y_reverse(expand = c(0, 0))

  if (identical(cmap, "viridis")) {
    p_main <- p_main + ggplot2::scale_fill_viridis_c(
      limits = c(0, 1), name = colorbar_label
    )
  } else {
    p_main <- p_main + ggplot2::scale_fill_gradient(
      low = "white", high = "black",
      limits = c(0, 1), name = colorbar_label
    )
  }
  p_main <- p_main +
    ggplot2::labs(x = "Samples (ordered by cluster)", y = NULL) +
    ggplot2::theme_minimal(base_size = 11) +
    ggplot2::theme(
      axis.text = ggplot2::element_blank(),
      axis.ticks = ggplot2::element_blank(),
      panel.grid = ggplot2::element_blank(),
      plot.margin = ggplot2::margin(t = 0, r = 5.5, b = 5.5, l = 5.5, unit = "pt")
    )
  if (!isTRUE(colorbar)) p_main <- p_main + ggplot2::guides(fill = "none")

  uniq <- unique(labels_ord)
  colors <- .discrete_palette(length(uniq), palette)
  names(colors) <- as.character(uniq)
  band_df <- data.frame(
    j = seq_len(n),
    y = 1L,
    cluster = factor(as.character(labels_ord), levels = as.character(uniq))
  )
  p_band <- ggplot2::ggplot(band_df,
      ggplot2::aes(x = .data$j, y = .data$y, fill = .data$cluster)) +
    ggplot2::geom_raster() +
    ggplot2::scale_fill_manual(values = colors) +
    ggplot2::coord_fixed(ratio = 0.04 * n, expand = FALSE) +
    ggplot2::guides(fill = "none") +
    ggplot2::theme_void() +
    ggplot2::theme(plot.margin = ggplot2::margin(0, 0, 0, 0))

  boundaries <- which(labels_ord[-length(labels_ord)] != labels_ord[-1L]) + 0.5
  for (b in boundaries) {
    p_main <- p_main +
      ggplot2::geom_hline(yintercept = b, color = "white",
                          linewidth = 0.4, alpha = 0.8) +
      ggplot2::geom_vline(xintercept = b, color = "white",
                          linewidth = 0.4, alpha = 0.8)
    p_band <- p_band +
      ggplot2::geom_vline(xintercept = b, color = "white",
                          linewidth = 0.5, alpha = 0.9)
  }

  combined <- patchwork::wrap_plots(p_band, p_main, ncol = 1L,
                                    heights = c(0.04, 1))
  if (!is.null(title)) {
    combined <- combined + patchwork::plot_annotation(title = title)
  }
  combined
}


# --- 3) Cluster boxplot -----------------------------------------------------

.plot_cluster_boxplot_scores <- function(scores, labels, order = NULL,
                                         palette = "Tableau 10",
                                         showfliers = FALSE, width = 0.75,
                                         title = NULL, xlabel = "Cluster",
                                         ylabel = "Uncertainty",
                                         annotation = NULL, rotation = NULL,
                                         ylim = c(-0.02, 1.02)) {
  prep <- .prepare_cluster_score_groups(scores, labels, order = order)
  df <- do.call(rbind, Map(function(lab, vals) {
    data.frame(cluster = rep(as.character(lab), length(vals)),
               value = vals, stringsAsFactors = FALSE)
  }, prep$order, prep$groups))
  df$cluster <- factor(df$cluster, levels = as.character(prep$order))

  colors <- .discrete_palette(length(prep$order), palette)

  p <- ggplot2::ggplot(df,
      ggplot2::aes(x = .data$cluster, y = .data$value, fill = .data$cluster)) +
    ggplot2::geom_boxplot(
      width = width,
      outlier.shape = if (isTRUE(showfliers)) 19 else NA,
      alpha = 0.8, linewidth = 0.5
    ) +
    ggplot2::scale_fill_manual(values = colors, guide = "none") +
    ggplot2::labs(x = xlabel, y = ylabel, title = title) +
    ggplot2::theme_minimal(base_size = 11) +
    ggplot2::theme(panel.grid.major.x = ggplot2::element_blank())

  if (!is.null(ylim)) {
    p <- p + ggplot2::coord_cartesian(ylim = ylim)
  }
  if (!is.null(rotation)) {
    p <- p + ggplot2::theme(
      axis.text.x = ggplot2::element_text(angle = rotation, hjust = 1)
    )
  }
  if (!is.null(annotation)) {
    p <- p + ggplot2::labs(subtitle = annotation) +
      ggplot2::theme(plot.subtitle = ggplot2::element_text(
        size = 9, color = "gray30"
      ))
  }
  p
}


# --- 4) Cluster violin ------------------------------------------------------

.plot_cluster_violin_scores <- function(scores, labels, order = NULL,
                                        palette = "Tableau 10",
                                        density_norm = c("width", "area", "count"),
                                        stripplot = TRUE, jitter = TRUE,
                                        size = 1.0, alpha = 0.22,
                                        inner = c("box", "quartile", "none"),
                                        title = NULL, xlabel = "Cluster",
                                        ylabel = "Uncertainty",
                                        annotation = NULL, rotation = NULL,
                                        ylim = c(-0.02, 1.02)) {
  density_norm <- match.arg(density_norm)
  inner <- match.arg(inner)

  prep <- .prepare_cluster_score_groups(scores, labels, order = order)
  df <- do.call(rbind, Map(function(lab, vals) {
    data.frame(cluster = rep(as.character(lab), length(vals)),
               value = vals, stringsAsFactors = FALSE)
  }, prep$order, prep$groups))
  df$cluster <- factor(df$cluster, levels = as.character(prep$order))
  colors <- .discrete_palette(length(prep$order), palette)

  scale_arg <- switch(density_norm, width = "width",
                      area = "area", count = "count")

  p <- ggplot2::ggplot(df,
      ggplot2::aes(x = .data$cluster, y = .data$value, fill = .data$cluster)) +
    ggplot2::geom_violin(scale = scale_arg, trim = TRUE, alpha = 0.8,
                         linewidth = 0.6, color = "black") +
    ggplot2::scale_fill_manual(values = colors, guide = "none")

  if (identical(inner, "box")) {
    p <- p + ggplot2::geom_boxplot(width = 0.15, outlier.shape = NA,
                                   fill = "white", color = "black",
                                   linewidth = 0.5)
  } else if (identical(inner, "quartile")) {
    p <- p + ggplot2::stat_summary(fun = stats::median, geom = "crossbar",
                                   width = 0.25, fatten = 2, color = "black")
  }

  if (isTRUE(stripplot)) {
    jitter_width <- if (isTRUE(jitter)) 0.11
                    else if (isFALSE(jitter)) 0
                    else as.numeric(jitter)
    p <- p + ggplot2::geom_jitter(width = jitter_width, height = 0,
                                  size = size, alpha = alpha, color = "black")
  }

  p <- p +
    ggplot2::labs(x = xlabel, y = ylabel, title = title) +
    ggplot2::theme_minimal(base_size = 11) +
    ggplot2::theme(panel.grid.major.x = ggplot2::element_blank())

  if (!is.null(ylim)) p <- p + ggplot2::coord_cartesian(ylim = ylim)
  if (!is.null(rotation)) {
    p <- p + ggplot2::theme(
      axis.text.x = ggplot2::element_text(angle = rotation, hjust = 1)
    )
  }
  if (!is.null(annotation)) {
    p <- p + ggplot2::labs(subtitle = annotation) +
      ggplot2::theme(plot.subtitle = ggplot2::element_text(
        size = 9, color = "gray30"
      ))
  }
  p
}


# --- 5) Cluster scatter (alpha + size by score) -----------------------------

.plot_cluster_scatter_xy <- function(X, labels, scores, embedding = NULL,
                                     palette = "Tableau 10",
                                     alpha_range = c(0.45, 0.9),
                                     size_range = c(1.5, 5.0),
                                     sort_order = TRUE,
                                     legend = TRUE,
                                     annotation = NULL,
                                     annotation_style = c("legend", "box"),
                                     title = NULL,
                                     scores_name = "Score",
                                     xlabel = "Component 1",
                                     ylabel = "Component 2",
                                     frameon = FALSE) {
  annotation_style <- match.arg(annotation_style)
  X <- as.matrix(X)
  labels <- as.vector(labels)
  scores <- as.numeric(scores)

  if (nrow(X) != length(labels) || nrow(X) != length(scores)) {
    stop("X, labels, and scores must have matching n_samples.", call. = FALSE)
  }

  if (is.null(embedding)) {
    if (ncol(X) > 2L) {
      coords <- .pca_2d(X)
    } else if (ncol(X) == 2L) {
      coords <- X[, 1:2, drop = FALSE]
    } else {
      coords <- cbind(X[, 1], rep(0, nrow(X)))
    }
  } else {
    coords <- as.matrix(embedding)[, 1:2, drop = FALSE]
  }

  valid <- is.finite(scores)
  if (!any(valid)) stop("No finite scores available for plotting.", call. = FALSE)
  lo_score <- min(scores[valid])
  hi_score <- max(scores[valid])

  size_hi <- size_range[1L]
  size_lo <- size_range[2L]
  alpha_hi <- alpha_range[1L]
  alpha_lo <- alpha_range[2L]

  if (isTRUE(all.equal(lo_score, hi_score))) {
    size_vals <- rep(mean(size_range), length(scores))
    alpha_vals <- rep(mean(alpha_range), length(scores))
  } else {
    norm <- (scores - lo_score) / (hi_score - lo_score)
    size_vals <- size_lo + norm * (size_hi - size_lo)
    alpha_vals <- alpha_lo + norm * (alpha_hi - alpha_lo)
  }

  uniq <- .sort_unique_labels(labels)
  colors <- .discrete_palette(length(uniq), palette)
  names(colors) <- as.character(uniq)

  cluster_mean <- vapply(uniq, function(lab) {
    v <- scores[(labels == lab) & valid]
    if (length(v) == 0L) NaN else mean(v)
  }, numeric(1L))

  df <- data.frame(
    x = coords[, 1L], y = coords[, 2L],
    cluster = factor(as.character(labels), levels = as.character(uniq)),
    alpha = pmin(pmax(alpha_vals, 0), 1),
    size = size_vals
  )
  if (isTRUE(sort_order)) df <- df[order(df$alpha), , drop = FALSE]

  legend_labels <- setNames(
    sprintf("%s (Mean = %.2f)", as.character(uniq), cluster_mean),
    as.character(uniq)
  )
  base_title <- if (!is.null(scores_name) && nzchar(scores_name)) {
    sprintf("Cluster, %s", scores_name)
  } else {
    "Cluster"
  }
  color_title <- if (!is.null(annotation) &&
                     identical(annotation_style, "legend")) {
    paste0(annotation, "\n", base_title)
  } else {
    base_title
  }

  p <- ggplot2::ggplot(df,
      ggplot2::aes(x = .data$x, y = .data$y, color = .data$cluster,
                   alpha = .data$alpha, size = .data$size)) +
    ggplot2::geom_point(stroke = 0.2) +
    ggplot2::scale_color_manual(values = colors, labels = legend_labels,
                                name = color_title) +
    ggplot2::scale_alpha_identity(guide = "none") +
    ggplot2::scale_size_identity(guide = "none") +
    ggplot2::labs(x = xlabel, y = ylabel, title = title) +
    ggplot2::theme_minimal(base_size = 11)

  if (!isTRUE(frameon)) {
    p <- p + ggplot2::theme(
      panel.border = ggplot2::element_blank(),
      panel.grid = ggplot2::element_blank(),
      axis.line = ggplot2::element_blank(),
      axis.text = ggplot2::element_blank(),
      axis.ticks = ggplot2::element_blank()
    )
  }
  if (!isTRUE(legend)) p <- p + ggplot2::guides(color = "none")

  if (!is.null(annotation) && identical(annotation_style, "box")) {
    p <- p + ggplot2::labs(subtitle = annotation) +
      ggplot2::theme(plot.subtitle = ggplot2::element_text(
        size = 9, color = "gray30"
      ))
  }

  p
}


# --- 6) Diagnostic scatter (shape by cluster, color by score) ---------------

.plot_diagnostic_scatter_xy <- function(X, labels, scores, embedding = NULL,
                                        cmap = "Greens_r",
                                        alpha_encoding = TRUE,
                                        alpha_range = c(0.3, 1.0),
                                        marker_size = 2.0,
                                        markers = NULL,
                                        sort_order = TRUE,
                                        legend = TRUE,
                                        colorbar = TRUE,
                                        colorbar_label = NULL,
                                        annotation = NULL,
                                        annotation_style = c("legend", "box"),
                                        title = NULL,
                                        scores_name = "Score",
                                        xlabel = "Component 1",
                                        ylabel = "Component 2",
                                        frameon = FALSE) {
  annotation_style <- match.arg(annotation_style)
  X <- as.matrix(X)
  labels <- as.vector(labels)
  scores <- as.numeric(scores)

  if (nrow(X) != length(labels) || nrow(X) != length(scores)) {
    stop("X, labels, and scores must have matching n_samples.", call. = FALSE)
  }

  if (is.null(embedding)) {
    if (ncol(X) > 2L) {
      coords <- .pca_2d(X)
    } else if (ncol(X) == 2L) {
      coords <- X[, 1:2, drop = FALSE]
    } else {
      coords <- cbind(X[, 1], rep(0, nrow(X)))
    }
  } else {
    coords <- as.matrix(embedding)[, 1:2, drop = FALSE]
  }

  valid <- is.finite(scores)
  if (!any(valid)) stop("No finite scores available for plotting.", call. = FALSE)
  lo_score <- min(scores[valid])
  hi_score <- max(scores[valid])

  # ggplot plotting characters (pch codes) as shape defaults; 14 of them.
  if (is.null(markers)) {
    markers <- c(21, 22, 24, 23, 25, 3, 8, 4, 11, 13, 9, 10, 12, 14)
  }

  uniq <- .sort_unique_labels(labels)
  shapes <- markers[((seq_along(uniq) - 1L) %% length(markers)) + 1L]
  names(shapes) <- as.character(uniq)

  norm_scores <- rep(0.5, length(scores))
  if (!isTRUE(all.equal(lo_score, hi_score))) {
    norm_scores[valid] <- (scores[valid] - lo_score) / (hi_score - lo_score)
  }

  alpha_vals <- rep(1, length(scores))
  if (isTRUE(alpha_encoding)) {
    alpha_hi <- alpha_range[1L]
    alpha_lo <- alpha_range[2L]
    if (isTRUE(all.equal(alpha_hi, alpha_lo))) {
      alpha_vals <- rep(mean(alpha_range), length(scores))
    } else {
      alpha_vals[valid] <- alpha_lo + norm_scores[valid] * (alpha_hi - alpha_lo)
    }
  }
  alpha_vals[!valid] <- 0.2

  cluster_means <- vapply(uniq, function(lab) {
    v <- scores[(labels == lab) & valid]
    if (length(v) == 0L) 0 else mean(v)
  }, numeric(1L))
  draw_order_clusters <- uniq[order(-cluster_means)]

  df <- data.frame(
    x = coords[, 1L], y = coords[, 2L],
    cluster = factor(as.character(labels), levels = as.character(uniq)),
    score = scores,
    alpha = pmin(pmax(alpha_vals, 0), 1)
  )
  if (isTRUE(sort_order)) {
    df$.rank_cluster <- match(as.character(df$cluster),
                              as.character(draw_order_clusters))
    df$.rank_within <- -norm_scores
    df <- df[order(df$.rank_cluster, df$.rank_within), , drop = FALSE]
    df$.rank_cluster <- NULL
    df$.rank_within <- NULL
  }

  color_title <- colorbar_label %||% scores_name
  fill_scale <- if (identical(cmap, "Greens_r")) {
    ggplot2::scale_fill_gradient(low = "#00441b", high = "#f7fcf5",
                                 limits = c(lo_score, hi_score),
                                 name = color_title)
  } else {
    ggplot2::scale_fill_gradientn(
      colors = grDevices::hcl.colors(256L, palette = cmap),
      limits = c(lo_score, hi_score),
      name = color_title
    )
  }

  shape_title <- if (!is.null(annotation) &&
                     identical(annotation_style, "legend")) {
    paste0(annotation, "\nCluster")
  } else {
    "Cluster"
  }

  p <- ggplot2::ggplot(df,
      ggplot2::aes(x = .data$x, y = .data$y, fill = .data$score,
                   shape = .data$cluster, alpha = .data$alpha)) +
    ggplot2::geom_point(size = marker_size, stroke = 0.5, color = "gray40") +
    fill_scale +
    ggplot2::scale_shape_manual(values = shapes, name = shape_title) +
    ggplot2::scale_alpha_identity(guide = "none") +
    ggplot2::labs(x = xlabel, y = ylabel, title = title) +
    ggplot2::theme_minimal(base_size = 11)

  if (!isTRUE(frameon)) {
    p <- p + ggplot2::theme(
      panel.border = ggplot2::element_blank(),
      panel.grid = ggplot2::element_blank(),
      axis.line = ggplot2::element_blank(),
      axis.text = ggplot2::element_blank(),
      axis.ticks = ggplot2::element_blank()
    )
  }
  if (!isTRUE(legend)) p <- p + ggplot2::guides(shape = "none")
  if (!isTRUE(colorbar)) p <- p + ggplot2::guides(fill = "none")

  if (!is.null(annotation) && identical(annotation_style, "box")) {
    p <- p + ggplot2::labs(subtitle = annotation) +
      ggplot2::theme(plot.subtitle = ggplot2::element_text(
        size = 9, color = "gray30"
      ))
  }

  p
}
