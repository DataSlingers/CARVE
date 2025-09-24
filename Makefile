.PHONY: r-setup
r-setup:
	Rscript -e "if (!requireNamespace('BiocManager', quietly=TRUE)) install.packages('BiocManager', repos='https://cran.rstudio.com/'); BiocManager::install(c('scDesign3','SingleCellExperiment','SummarizedExperiment','Matrix','rvinecopulib'), ask=FALSE, update=TRUE)"