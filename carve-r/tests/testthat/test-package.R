test_that("the package is version 2.0.0", {
  expect_identical(as.character(utils::packageVersion("CARVE")), "2.0.0")
})

test_that("an unexpected warning fails a test", {
  expect_identical(getOption("warn"), 2L)
})
