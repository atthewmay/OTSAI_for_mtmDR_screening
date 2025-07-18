import pandas as pd
import shutil
import os

gts = pd.read_csv("abramoff_ground_truths.csv") # pull this from Dr. Abramoff's uiowa website (the reference standard)
random_seed=42
samples = gts.groupby('rDR').apply(lambda x:x.sample(n=15,replace=False,random_state=random_seed)).reset_index(drop=True)

source_dir = "cropped_data/"
destination_dir = "test_data/validation_subset"
os.makedirs(destination_dir, exist_ok=True)

# Get the list of IDs to match (assuming IDs are in a column named 'ID')
id_list = samples['examid'].astype(str).tolist()

# Loop through files in the source directory and copy matching files
for file_name in os.listdir(source_dir):
    if any(id_part in file_name for id_part in id_list):  # Check if any ID is a substring of the filename
        shutil.copy(os.path.join(source_dir, file_name), os.path.join(destination_dir, file_name))

print("Files copied successfully.")
