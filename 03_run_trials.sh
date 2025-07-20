#!/bin/bash

# source ~/.zshrc  # Source .zshrc to load the run_python function

run_python() {
    PYTHONPATH="$PYTHONPATH:$(pwd)" python "$@"
}

# Command-line arguments
models_param=$1  # Accept a comma-separated list of models from the command line

# Parse models into an array
IFS=',' read -r -a models <<< "$models_param"

# Default settings
FULLDATA=False
# FULLDATA=True

if [ "$FULLDATA" = "True" ]; then
    image_directory="cropped_data/" # Full data directory
    output_dir="outputs/full_model_run_DATE" # replace with a dir name
else
#     image_directory="test_data/validation_subset/"
    image_directory="test_data/subset_png_data/" # a very small subset of the data for quick verification of functionality
    output_dir="outputs/full_model_run_validation_subset_DATE" 

fi

message_pairs=(
    "system_header_basic,"
    "system_header_with_background,"
    "system_header_with_background,few_shot_with_background"
)
post_message="user_reminder_post_message"

for model in "${models[@]}"; do
    for pair in "${message_pairs[@]}"; do
        IFS=',' read -r system_message pre_messages <<< "$pair"
        output_file="${output_dir}/DataSource_$(basename "${image_directory%/}")_${model}__${system_message}__${pre_messages}.json"


        cmd="python -u code_files/run_model.py --image_directory \"$image_directory\" --model \"$model\" --system_message \"$system_message\" --pre_messages \"$pre_messages\" --output_file \"$output_file\" --post_messages \"$post_message\" --kwargs default"

        # Echo the command (for debugging/logging)
        echo "$cmd"

        # Execute the command
        eval "$cmd"
        echo "Completed for model: $model, system_message: $system_message, pre_messages: $pre_messages, post_message: $post_message, output file: $output_file"
    done
done

