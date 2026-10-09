# Any warning a test does not expect fails it, the counterpart of
# filterwarnings = error in the Python suite. expect_warning() and
# expect_no_warning() still work under warn = 2.
withr::local_options(list(warn = 2), .local_envir = testthat::teardown_env())
