.PHONY: r-setup
r-setup:
	Rscript -e "if (!requireNamespace('BiocManager', quietly=TRUE)) install.packages('BiocManager', repos='https://cran.rstudio.com/'); BiocManager::install(c('scDesign3','SingleCellExperiment','SummarizedExperiment','Matrix','rvinecopulib'), ask=FALSE, update=TRUE)"

.PHONY: m3c-setup
m3c-setup:
	Rscript -e "if (!requireNamespace('BiocManager', quietly=TRUE)) install.packages('BiocManager', repos='https://cran.rstudio.com/'); BiocManager::install('M3C', ask=FALSE, update=FALSE)"

.PHONY: levine-setup
levine-setup:
	Rscript -e "if (!requireNamespace('BiocManager', quietly=TRUE)) install.packages('BiocManager', repos='https://cran.rstudio.com/'); BiocManager::install('HDCytoData', ask=FALSE, update=FALSE)"
