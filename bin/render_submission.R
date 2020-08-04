#!/usr/bin/env Rscript
#
# Prepare submission for MovieLens project:
# - render PDF file from submission.Rmd
# - extract R code from submission.Rmd as base for submission.R

library(here)
library(rmarkdown)
library(knitr)

# render PDF report for submission
render(input = here("submission/submission.Rmd"),
       output_file = "submission.pdf",
       output_dir = here("submission"),
       knit_root_dir = here("submission"))

# extract R code as base for submission.R (will be further modified)
purl(input = here("submission/submission.Rmd"),
     output = here("submission/submission.R"),
     documentation = 2)
