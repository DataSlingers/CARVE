# BiocCheck report

Run on 2026-10-09 with BiocCheck 1.44.2 (Bioconductor 3.21, as installed here) on CARVE 2.0.0, branch
r-package-rework at 2f52cc2, from the built tarball, as a new package. It gates nothing; it is here
for the decision to submit to Bioconductor. The log's wording is kept below. Nothing it reports was
fixed in this stage.

BiocCheck reported 2 errors, 1 warning and 12 notes.

## Errors

- New package 'y' version not 99 (i.e., x.99.z); Package version: 2.0.0
- Unable to find your email in the Support Site: HTTP 404 Not Found.

## Warnings

- New package x version starting with non-zero value (e.g., 1.y.z, 2.y.z); got '2.0.0'.

## Notes

- 'LazyData:' in the 'DESCRIPTION' should be set to false or removed
- Update R version dependency from 4.1.0 to 4.5.0
- Consider adding the maintainer's ORCID iD in 'Authors@R' with 'comment=c(ORCID="...")'
- No 'fnd' role found in Authors@R. If the work is supported by a grant, consider adding the 'fnd' role to the list of authors.
- Avoid 1:...; use seq_len() or seq_along(). Found in plotting.R (line 695, column 29) and utils.R (line 625, column 19)
- Avoid redundant 'stop' and 'warn*' in signal conditions. Found in R/runner.R (line 168, column 9)
- Use accessors; don't access S4 class slots via '@' in examples/vignettes. Found in vignettes/single-cell.Rmd
- Avoid '<<-' if possible (found 2 times): R/sweep.R (line 237, column 13) and R/utils.R (line 196, column 14)
- The recommended function length is 50 lines or less. There are 13 functions greater than 50 lines. The longest is fit_carve() (R/carve.R) at 213 lines; the log names no others except draw_metric_lines() (R/plotting.R) at 75 lines.
- Consider shorter lines; 890 lines (8%) are > 80 characters long.
- Consider multiples of 4 spaces for line indents; 2999 lines (28%) are not.
- Cannot determine whether maintainer is subscribed to the Bioc-Devel mailing list (requires admin credentials).
