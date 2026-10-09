# Fails when an exported function of CARVE is used in neither a vignette nor
# the tutorial. Run from code/carve-r:
#
#   Rscript data-raw/check_exports_used.R [extra .Rmd files]
#
# A function counts as used when its name is a symbol in an R chunk, whether
# it is called or passed as a value, as KMeans is in
# estimator_grid(KMeans, ...). Prose and inline code do not count.

chunk_code <- function(path) {
  lines <- readLines(path, warn = FALSE)
  starts <- grep("^\\s*```\\{r", lines)
  ends <- grep("^\\s*```\\s*$", lines)
  code <- character(0)
  for (start in starts) {
    end <- ends[ends > start][1L]
    if (!is.na(end) && end > start + 1L) {
      code <- c(code, lines[(start + 1L):(end - 1L)])
    }
  }
  code
}

chunk_symbols <- function(path) {
  code <- chunk_code(path)
  if (length(code) == 0L) {
    return(character(0))
  }
  data <- utils::getParseData(parse(text = code, keep.source = TRUE))
  unique(data$text[data$token %in% c("SYMBOL_FUNCTION_CALL", "SYMBOL")])
}

namespace <- readLines("NAMESPACE")
exports <- sub("^export\\((.*)\\)$", "\\1", grep("^export\\(", namespace, value = TRUE))

documents <- c(
  list.files("vignettes", pattern = "[.]Rmd$", full.names = TRUE),
  file.path("..", "notebooks", "R_Tutorial.Rmd"),
  commandArgs(trailingOnly = TRUE)
)
documents <- documents[file.exists(documents)]

used <- unique(unlist(lapply(documents, chunk_symbols)))
missing <- sort(setdiff(exports, used), method = "radix")
cat(sprintf("%d documents, %d exports, %d not used\n", length(documents), length(exports), length(missing)))
if (length(missing) > 0L) {
  cat("Not used in any vignette or the tutorial:\n", paste0("  ", missing, "\n"), sep = "")
  quit(status = 1L)
}
cat("Every export is used.\n")
