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
