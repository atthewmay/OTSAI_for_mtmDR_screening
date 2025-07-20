#!/bin/bash
# source ~/.zshrc  # Source .zshrc to load the run_python function
#
#
./03_run_trials.sh "gpt-4o-2024-08-06,gpt-4o-mini" &&
python code_files/04_grade_output_dir.py --output_dir outputs/model_run_demo_7_17_25 &&
#
# Can run the following, but nicer in Rstudio for display
Rscript code_files/05_ROC_analysis.R

