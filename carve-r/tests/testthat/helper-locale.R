# Sets the collation locale until the calling test ends. R's sort() follows
# it and Python's sorted() does not. Under the C locale the two orders agree,
# so tests of the lists in error messages set a locale where they differ.
# The test is skipped when the locale is not installed.
local_collation <- function(locale = "en_US.UTF-8", envir = parent.frame()) {
  old <- Sys.getlocale("LC_COLLATE")
  if (!nzchar(suppressWarnings(Sys.setlocale("LC_COLLATE", locale)))) {
    testthat::skip(sprintf("Collation locale %s is not installed.", locale))
  }
  withr::defer(Sys.setlocale("LC_COLLATE", old), envir = envir)
}
