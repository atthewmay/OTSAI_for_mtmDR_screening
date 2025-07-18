
import numpy as np
import re
import pickle
from pathlib import Path
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
import os
import json
import pandas as pd
import argparse
import utils
import eval_utils
import CONSTANTS as C

USE_AFTERMARKET_TRUE_PROBABILITIES = False
SAVE_PDFS = False

def parse_arguments():
    parser = argparse.ArgumentParser(description="Grade retinal images for diabetic retinopathy using GPT API.")
#     parser.add_argument(
#         "--image_path", type=str, default=None,
#         help="Path to the directory containing the images to be graded."
#     )
    parser.add_argument(
        "--output_dir", type=str, required=True,
    )

    return parser.parse_args()

args = parse_arguments()
score_dicts = {}


df_dict = {}
# 1. Just pull all the files into a dictionary.
for f_name in sorted(os.listdir(args.output_dir)):
    f_path = os.path.join(args.output_dir,f_name)
    model,prompt_type = eval_utils.get_model_and_prompt_type(f_name)
    print(f"processing: {model},{prompt_type}")
    df_dict[f_name] = eval_utils.pull_valid_df(f_path)
#     if "2024" in model and "Background" in prompt_type:
#         df_dict[f_name] = eval_utils.pull_valid_df(f_path)
#     else:
#         print(f"skipping {model},{prompt_type}")
# 
# # 2. Make table for performance and save original pdfs
# table_data = []
# pt_df_dict = {}
# for f_name,df in df_dict.items():
#     print(f"data for {f_name}")
#     model,prompt_type = eval_utils.get_model_and_prompt_type(f_name)
#     df = eval_utils.process_df(df,USE_AFTERMARKET_TRUE_PROBABILITIES)
#     scores_dict = eval_utils.grade_single(df)
# 
#     table_data.append({
#         "Model": model,
#         "Prompt Type": prompt_type,
#         "Recall": scores_dict["recall"],
#         "Specificity": scores_dict['specificity']
#     })
#     if SAVE_PDFS:
#         eval_utils.save_PDFS(os.path.join(args.output_dir,f_name),df)
# 
#     merged_df,patient_df = utils.make_grading_dfs(df,path_joiner="")
#     if not "gemini" in f_name:
#         pt_df_dict[f_name] = patient_df # works bc we alrady processed above
# 
# eval_utils.print_eval_table(table_data)
# 
# combined = []
# for key, subdf in pt_df_dict.items():
#     # Make a copy so we can add a column without altering original
#     tmp = subdf.copy()
#     tmp = tmp[['examid','ground_truth_rDR','prob_of_true']]
#     key = re.match(r"DataSource_cropped_data_(.*?)(?:__.json|\.json)", key).group(1)
#     tmp["dict_key"] = key  # store the Python dict key in each row
#     combined.append(tmp)
# final_df = pd.concat(combined, ignore_index=True)
# final_df.to_csv("pt_df_dict_combined.csv", index=False)
# 
# 
# 

    # For ease of debug
# pickle.dump(pt_df_dict,open("outputs/temp_pt_df_dict.pickle",'wb'))

# AS of March 2024 the following ROC curves will be done in R
# 
multi_roc_path = f"outputs/output_pdfs/{args.output_dir.split('/')[-1]}/multi_roc_original.pdf"
# Path(multi_roc_path).parent.mkdir(parents=True, exist_ok=True)
# fig = utils.roc_multiplot(pt_df_dict,annotate=False,
#                           optimal_point_adds={("gpt-4o-2024-08-06","system_header_with_background"):0.75})
# plt.savefig(multi_roc_path)
# plt.close()
# 
# 
# 3. Make table for threshold-adjusted performance (feels stupid yeah bc we could just read off), but this is useful for
# PDF to actually inspect the images themselves. Also needed for PPV and NPV
print("\n\n\t Now testing acrossing the different set points")

expert_grades_df = utils.pull_expert_ICDR_grades()

# set_point_dict = {'DataSource_validation_subset_gpt-4o-2024-08-06__system_header_with_background__.json':0.0017007224218578362 - 1e-9}
#Youdens
# set_point_dict = {'DataSource_cropped_data_gpt-4o-2024-08-06__system_header_with_background__.json':0.0017007224218578362 - 1e-11}
# my 75% sens threshold
# set_point_dict = {'DataSource_cropped_data_gpt-4o-2024-08-06__system_header_with_background__.json':0.022977366137564313 - 1e-11}


# set_point_dict = {'DataSource_cropped_data_gpt-4o-2024-08-06__system_header_basic__.json':6.61e-05} 
# AS of 3/9/25, this is the trheshold for the 80% sensiivtiy cutoff
set_point_dict = {'DataSource_cropped_data_gpt-4o-2024-08-06__system_header_with_background__.json':0.001700722 - 1e-11}
set_point_table_data = []
for f_name,set_point in set_point_dict.items():
    model,prompt_type = eval_utils.get_model_and_prompt_type(f_name)
    df = df_dict[f_name]
    df = eval_utils.process_df(df,USE_AFTERMARKET_TRUE_PROBABILITIES,set_point)
    scores_dict = eval_utils.grade_single(df)
# 
#     set_point_table_data.append({
#         "Model": model,
#         "Prompt Type": prompt_type,
#         "Recall": scores_dict["recall"],
#         "Specificity": scores_dict['specificity']
#     })

#     eval_utils.save_PDFS(os.path.join(args.output_dir,f_name),df,"_setpoint_adjust")

    # Now save all false negatives
    merged_df,patient_df = utils.make_grading_dfs(df,path_joiner="")
    fn_indices=patient_df[(patient_df['ground_truth_rDR'] == 1) & (patient_df['predicted_rDR'] == 0)].index
    print(f"All fn_indices are { fn_indices }")
    utils.compare_to_expert_grades(patient_df.loc[fn_indices],expert_grades_df)

    fn_RL_dict = {"IM1058":(".001.png",".000.png"), # a dictionary to reverse the Right-left order of images for those FN cases that have the right-left images reversed from typical... Could have used a set:)
             "IM0137":(".001.png",".000.png"),
             "IM0090":(".001.png",".000.png"),
             "IM0101":(".001.png",".000.png"),
             "IM0285":(".001.png",".000.png"),
             }
    full_FN_fig = utils.pt_level_plot(fn_indices,patient_df,merged_df,"All False Negative Examples",
                                      RL_dict = fn_RL_dict,
                                      max_images=len(fn_indices)*2)
    save_path = os.path.join(Path(multi_roc_path).parent,f"{f_name.split('.')[0]}_all_fn_egs.pdf")
    plt.savefig(save_path)

#     tp_indices = patient_df[(patient_df['ground_truth_rDR'] == 1) & (patient_df['predicted_rDR'] == 1)].index
    tp_indices = patient_df[patient_df['examid'].isin(["IM0295","IM0549","IM0266","IM0309"])].index

    fp_indices = patient_df[(patient_df['ground_truth_rDR'] == 0) & (patient_df['predicted_rDR'] == 1)].index
    np.random.seed(seed=42)
    n_pts_per_plot = 4
    ind_dict = {
            "Patient-Level True Positives" : {'inds':np.random.choice(tp_indices,n_pts_per_plot,replace=False),
                                              'RL_dict':{}},
            "Patient-Level False Positives" : {'inds':np.random.choice(fp_indices,n_pts_per_plot,replace=False),
                                              'RL_dict':{"IM1103":(".001.png",".000.png")}},
            "Patient-Level False Negatives" : {'inds':np.random.choice(fn_indices,n_pts_per_plot,replace=False),
                                              'RL_dict':fn_RL_dict},
    }
    for title,sub_d in ind_dict.items():
        plot = utils.pt_level_plot(sub_d['inds'],patient_df,merged_df,title="",
                                          RL_dict = sub_d["RL_dict"],
                                          max_images=len(sub_d['inds'])*2)
        save_path = os.path.join(Path(multi_roc_path).parent,f"{f_name.split('.')[0]}_{title.replace(' ','_')}.pdf")
        plt.savefig(save_path)




    



    



#     merged_df,patient_df = utils.make_grading_dfs(df,path_joiner="")
#     pt_df_dicts[f_name] = patient_df # works bc we alrady processed above



