#!/usr/bin/env Rscript
#
# Render PDF file from submission.Rmd

library(here)

render(input = here("submission/submission.Rmd"),
       output_file = paste0(Sys.Date(), "_submission.pdf"),
       output_dir = here("submission"),
       knit_root_dir = here("submission"))
