# Fixtures are written by data-raw/make_fixtures.py from the Python package.
# JSON has no NaN, so the fixtures store NaN as null. Inside an array null
# reads back as NA; a null scalar field reads back as NULL.
read_fixture <- function(name) {
  testthat::skip_if_not_installed("jsonlite")
  jsonlite::fromJSON(
    testthat::test_path("fixtures", paste0(name, ".json")),
    simplifyDataFrame = FALSE
  )
}

fixture_vector <- function(x) {
  if (is.list(x)) {
    x <- unlist(lapply(x, function(v) if (is.null(v)) NA else v))
  }
  as.numeric(x)
}

fixture_matrix <- function(x) {
  if (is.list(x)) {
    x <- do.call(rbind, lapply(x, fixture_vector))
  }
  storage.mode(x) <- "double"
  x
}

fixture_number <- function(x) {
  if (is.null(x)) NA_real_ else as.numeric(x)
}

nan_to_na <- function(x) {
  x[is.nan(x)] <- NA
  x
}

# TRUE when two label vectors put -1 on the same samples and group the other
# samples the same way, whatever the label values. Used to compare clusterings
# with Python's, which number clusters differently.
same_partition <- function(a, b) {
  if (!identical(which(a < 0), which(b < 0))) {
    return(FALSE)
  }
  keep <- a >= 0
  if (!any(keep)) {
    return(TRUE)
  }
  counts <- table(a[keep], b[keep])
  all(rowSums(counts > 0) == 1L) && all(colSums(counts > 0) == 1L)
}
