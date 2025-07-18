
from pathlib import Path
from tabulate import tabulate
import re
import CONSTANTS as C
import pickle
import os
import json
import pandas as pd
import argparse
import utils

def adjust_preds_to_set_point(df,set_point):
    """returns a modified df where the parsed_json.mtmDR is changed based on the given set point probabiliyt output. df is
    before the patient_df and merged_df"""
    df_out = df.copy()
    if "parsed_json.mtmDR" in df_out.columns:
        df_out["parsed_json.mtmDR"] = df_out['prob_of_true'] > set_point
    elif "predicted_rDR" in df_out.columns:
        df_out["predicted_rDR"] = df_out['prob_of_true'] >= set_point
        df_out["predicted_rDR"] = df_out["predicted_rDR"].astype(int)
    else:
        raise Exception
    return df_out


def normalize_parsed_json(df, column='parsed_json'):
    """Bc gemini has terrible outputs, would apply this"""
    # Define default structure
    default_dict = {'Gradable': None, 'ME': None, 'mtmDR': None}

    def process_row(row):
        # Extract first element if list is non-empty; otherwise, return empty dict
        data = row[0] if isinstance(row, list) and row else {}
        # Merge with default_dict to ensure all keys exist
        return {**default_dict, **data}

    # Apply transformation
    normalized_df = df[column].apply(process_row).apply(pd.Series)
    normalized_df = normalized_df.rename(columns=lambda x: f"parsed_json.{x}")

    # Merge back to original dataframe
    return df.join(normalized_df)




def get_model_and_prompt_type(f_name):
    fnl = f_name.strip(".json").strip("__").split("__")
    model = fnl[0].split("_")[-1]
    prompt_type = C.prompt2promptname["__".join(fnl[1:])]
    return model, prompt_type

def pull_valid_df(file_name):
    """just loads it from the json and makes sure anything included is fully valid"""
    try:
        with open(file_name, "r") as file:
            data = json.load(file)
    except Exception as e:
        print(f"\n\n\t\tFailed to load {file_name}.\n\nwith exception as follows:\n")
        print(e)
        return None
    # import pdb; pdb.set_trace()
    df = pd.json_normalize(data)
    if 'gemini' in file_name:
        print(f" The count of values for the json attempted output in the model is {df.parsed_json.value_counts(dropna=False)}")
        df = normalize_parsed_json(df)

    invalid_results = df[pd.isna(df['parsed_json.mtmDR'])]

    if len(invalid_results)>0:
        print(f"number of invalid results are {invalid_results.shape[0]}. invalid_results are as follows")
        print("After applying exclusion at the patient level, the excluded DF = ")
#         df = df[~pd.isna(df['parsed_json.mtmDR'])]
        invalid_idxs = df[pd.isna(df['parsed_json.mtmDR'])].index
        invalid_names = df.loc[invalid_idxs,'image_name'].apply(lambda x: x.split("/")[-1].split('.')[0]).tolist()
        pattern = '|'.join(map(re.escape, invalid_names))  # Escape special characters in names
        invalid_df = df[df['image_name'].str.contains(pattern, na=False)]
        print(invalid_df)

        df = df[~df['image_name'].str.contains(pattern, na=False)]

        print(f"after eliminating the poorly formated examples w/o json, df shape is {df.shape}")
    else:
        print(f"all rows are valid for {file_name}")
    return df


#     os.makedirs(os.path.dirname(output_pdf_path), exist_ok=True)

def process_df(df,USE_AFTERMARKET_TRUE_PROBABILITIES=False,set_point = None):
    if USE_AFTERMARKET_TRUE_PROBABILITIES:
        if df['prob_of_true'].notna().any(): # Excludes gemini
            tps = utils.aftermarket_true_prob_extraction(df)
            df['prob_of_true'] = tps

    if set_point is not None:
        print(f"Adjusting df based on set_point = {set_point}")
        df = adjust_preds_to_set_point(df,set_point)
    return df
# 
def grade_single(df):
    merged_df,patient_df = utils.make_grading_dfs(df,path_joiner="")
    scores_dict = utils.calculate_scores(patient_df)
    utils.print_scores(scores_dict)
    scores_dict['patient_df'] = patient_df
    return scores_dict
#     pt_df = utils.grade_outputs(df,output_pdf=output_pdf_path)
#     scores_dict = utils.calculate_scores(patient_df)

def save_PDFS(file_name,df,name_append=""):
    """file_name is the whole path. Makes the FP TP FN TN figure and an ROC curve at top"""
    output_pdf = " ".join(get_model_and_prompt_type(file_name)).replace(" ","_")+f"{name_append}.pdf"
    parent_pdf_dir = file_name.split('/')[-2]
    output_pdf_path = f"outputs/output_pdfs/{parent_pdf_dir}/{output_pdf}"
    Path(output_pdf_path).parent.mkdir(parents=True, exist_ok=True)
    print("will save to")
    print(output_pdf_path)
    merged_df,patient_df = utils.make_grading_dfs(df,path_joiner="")
    utils.make_and_save_images(patient_df,merged_df,output_pdf_path)

def print_eval_table(table_data,
                    prompt_types = ["Basic Instructions", "Background Knowledge","Few Shot"],
                    models = ["gpt-4o-2024-08-06", "gpt-4o-mini", "grok-2-vision-1212", "gemini-1.5-pro"]):
    def pull_element(table_data,prompt_type,model):
        output = [e for e in table_data if e['Model']==model and e['Prompt Type'] == prompt_type]
        if len(output)!=1:
            print(f"output is unexpected length: {len(output)}. Returning None")
            return {}
        return output[0]



    table = [[""]+models]
    for i,prompt_type in enumerate(prompt_types):
        row = [prompt_type]
        for j,model in enumerate(models):
            data = pull_element(table_data,prompt_type,model)
            recall = f"{data.get('Recall'):.2f}" if data.get('Recall') else data.get('Recall')
            specificity = f"{data.get('Specificity'):.2f}" if data.get('Specificity') else data.get('Specificity')
            row.append(f"{recall} -- {specificity}")
        table.append(row)
    print(table)
    print(tabulate(table,headers="firstrow", tablefmt="grid"))
    print(tabulate(table,headers="firstrow", tablefmt="csv"))
