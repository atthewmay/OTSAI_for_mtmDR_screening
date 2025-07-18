import pandas as pd
import os

root = "/Users/matthewhunt/Research/Iowa_Research/Abramoff_Projects/LLM_Messidor_Study"
file = "abramoff_ground_truths.csv"

# 1. Count the number of each category in the abramoff ground truth file. 
df = pd.read_csv(os.path.join(root,file))
print(df)
num_rDR = len(df[df['rDR']==1])
num_not_rDR = len(df[df['rDR']==0])
print(f"in the abramoff_ground_truth_file, the number of pts with rDR is {num_rDR}\nNumber of pts w/o rDR is {num_not_rDR}")

# 2. Count the overlap of sets of names from the abramoff ground truth file and my OS.listdir
data_dir = os.path.join(root,'cropped_data')
names_files = set([f.split('.')[0] for f in os.listdir(data_dir)])
names_gt = set(df['examid'])

in_both    = names_files & names_gt
only_files = names_files - names_gt
only_gt    = names_gt   - names_files

# print counts
print(f"Total in files:               {len(names_files)}")
print(f"Total in ground_truth:        {len(names_gt)}")
print(f"In both (files ∩ ground_truth): {len(in_both)}")
print(f"Only in files:                {len(only_files)}")
print(f"These ids are {only_files}")
print(f"Only in ground_truth:         {len(only_gt)}")



import pdb; pdb.set_trace()