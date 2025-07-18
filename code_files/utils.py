from PIL import Image
import cv2
import random
import math
import io
import numpy as np
from collections import defaultdict, Counter
from sklearn.metrics import roc_curve, auc
import CONSTANTS as C
from scipy.stats import mode

import json
import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from matplotlib.backends.backend_pdf import PdfPages
import eval_utils


# Dictionary with model strings and their respective input/output token costs per 1,000 tokens
model_costs = {
    "gpt-4o": {"input_cost_per_1M": 2.5, "output_cost_per_1M": 10.0},  # Example costs for GPT-4
    "gpt-4o-2024-08-06": {"input_cost_per_1M": 2.5, "output_cost_per_1M": 10.0},  # Example costs for GPT-4
    "gpt-4o-mini": {"input_cost_per_1M": 0.150, "output_cost_per_1M": 0.6},  # Example costs for GPT-4o-mini
    "grok-beta":{"input_cost_per_1M": 5, "output_cost_per_1M": 15},
    "grok-vision-beta":{"input_cost_per_1M": 10, "output_cost_per_1M": 15},
    "grok-2-vision-1212":{"input_cost_per_1M": 2, "output_cost_per_1M": 10},
    "gemini-1.5-pro":{"input_cost_per_1M": 1.25, "output_cost_per_1M": 5},
}

model2api_key = {
    "gpt-4o":"GPT_API_SECRET_KEY",
    "gpt-4o-2024-08-06":"GPT_API_SECRET_KEY",
    "gpt-4o-mini":"GPT_API_SECRET_KEY",
    "grok-beta":"GROK_API_SECRET_KEY",
    "grok-vision-beta":"GROK_API_SECRET_KEY",
    "grok-2-vision-1212":"GROK_API_SECRET_KEY",
    "gemini-1.5-pro":"GEMINI_API_SECRET_KEY",
    "gemini-1.5-flash":"GEMINI_API_SECRET_KEY",
}
# "gpt-4o-2024-08-06,gpt-4o-mini"
# "gemini-1.5-pro,gemini-1.5-flash"
# "grok-vision-beta,grok-2-vision-1212"

gpt_kwargs = '{"temperature":0,"logprobs":true,"top_logprobs":20}'
grok_kwargs = '{"temperature":0,"logprobs":true,"top_logprobs":8}'
# gemini_kwargs = '{"temperature":0,"response_logprobs":true,"logprobs":5}'
gemini_kwargs = '{"temperature":0}'
model2default_kwargs = {
    "gpt-4o":gpt_kwargs,
    "gpt-4o-2024-08-06":gpt_kwargs,
    "gpt-4o-mini":gpt_kwargs,
    "grok-beta":grok_kwargs,
    "grok-vision-beta":grok_kwargs,
    "grok-2-vision-1212":grok_kwargs,
    "gemini-1.5-pro":gemini_kwargs ,
    "gemini-1.5-flash":gemini_kwargs ,
}

# Function to calculate the cost of an API call
def calculate_prompt_cost(response, model):
    # Retrieve the cost per 1k tokens for the specified model
    if model not in model_costs:
        raise ValueError(f"Model '{model}' not found in model_costs dictionary.")
    
    model_info = model_costs[model]
    input_cost_per_1M = model_info["input_cost_per_1M"]
    output_cost_per_1M = model_info["output_cost_per_1M"]
    
    # Extract token usage from the response
    if "gemini" in model:
        usage = response.usage_metadata
        prompt_tokens = usage.prompt_token_count
        completion_tokens = usage.candidates_token_count
    else:
        usage = response.usage
        prompt_tokens = usage.prompt_tokens
        completion_tokens = usage.completion_tokens

    # Calculate the prompt and completion costs
    prompt_cost = (prompt_tokens / 1e6) * input_cost_per_1M
    completion_cost = (completion_tokens / 1e6) * output_cost_per_1M
    total_cost = prompt_cost + completion_cost

    return {
        "model": model,
        "prompt_cost": prompt_cost,
        "completion_cost": completion_cost,
        "total_cost": total_cost
    }

def get_reponse_text(response,model_name):
    if "gemini" in model_name:
        response_text = response.text  
        response_text = gemini_post_process(response_text)
    else:
        response_text = response.choices[0].message.content
    return response_text

def gemini_post_process(response_text):
    """does this for some reason '```json\n{\n  'RDR': false,\n  'ME': false,\n  'Gradable': true\n}\n```'"""
    return response_text.strip('`json\n') #strip any of those chars

def get_response_content(response,model_name):
    if "gemini" in model_name:
        content = response.candidates[0]
    else:
        content = response.choices[0].message.content
    return content

def postprocessed_parsed_json(parsed_json,model_name):
    if "gemini" in model_name:
        try:
            assert len(parsed_json)==3
        except:
            import pdb; pdb.set_trace()
        merged_dict = {}
        for d in data:
            merged_dict.update(d)
        return merged_dict
    return parsed_json



def downsize_image(image_path, max_size=768):
    """
    Resize the image to ensure its dimensions are within max_size x max_size.
    Args:
        image_path (str): Path to the input image.
        max_size (int): Maximum width/height for the image.
    Returns:
        bytes: Resized image in bytes format.
    """
    with Image.open(image_path) as img:
        # Ensure the image is in RGB or RGBA mode (PNG supports transparency)
        img = img.convert("RGBA")
        
        # Get original dimensions
        original_width, original_height = img.size
        
        # Check if resizing is needed
        if max(original_width, original_height) > max_size:
            # Calculate new size preserving aspect ratio
            scaling_factor = max_size / max(original_width, original_height)
            new_width = int(original_width * scaling_factor)
            new_height = int(original_height * scaling_factor)
            
            # Resize the image
            img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
        
        # Save the image to bytes in PNG format
        image_bytes = io.BytesIO()
        img.save(image_bytes, format="PNG")
        return image_bytes.getvalue()


def make_grading_dfs(predictions_df,path_joiner=""):
    ground_truth_df = pd.read_csv(os.path.join(path_joiner,"abramoff_ground_truths.csv"))  # Ground truth CSV with examid and rDR columns
    
    # Step 1: Extract ID from `image_name` in predictions_df
    predictions_df['examid'] = predictions_df['image_name'].apply(lambda x: x.split('/')[-1].split('.')[0])
    
    # Step 2: Merge predictions with ground truth based on examid
    merged_df = pd.merge(predictions_df, ground_truth_df, on='examid', how='inner')
    
    # Step 3: Calculate the predicted and ground truth labels for referable diabetic retinopathy (rDR) on each image
    merged_df['predicted_rDR'] = merged_df['parsed_json.mtmDR'].astype(int)
    merged_df['ground_truth_rDR'] = merged_df['rDR']
    
    # Step 4: Aggregate at the patient level (examid), using max to indicate referable if either eye is positive
    patient_df = merged_df.groupby('examid').agg({
        'ground_truth_rDR': 'max',  # If either eye has rDR, the patient is referable in ground truth
        'predicted_rDR': 'max',      # If either eye is predicted as referable, the patient is referable in prediction
        'prob_of_true': 'max'      # If either eye is predicted as referable, the patient is referable in prediction
    }).reset_index()
    return merged_df,patient_df

def pull_expert_ICDR_grades(path_joiner = ""):
    """3/22/25 -- pulls the voted expert grade from the experts grade file"""
    df = pd.read_csv(os.path.join(path_joiner,"abramoff_expert_ICDR_grades_local.csv"),
                     na_values=[""])
    for col in df.columns:
        df[col] = df[col].astype("Int64")



#     icdr_cols = [col for col in df.columns if "ICDR" in col]
    icdr_cols = ["Han_ICDR","Williams_ICDR","Walker_ICDR"]
    dme_cols = ["Han_DME","Williams_DME","Walker_DME"]
    non_consensus = []
    def icdr_vote(row):
        vals = row[icdr_cols].values
        m = mode(vals, keepdims=True).mode[0]
        if np.sum(vals == m) == 1:  # no agreement
            non_consensus.append(row[0])
            return int(np.median(vals))
        return int(m)
    df["ICDR_voted"] = df.apply(icdr_vote, axis=1)
    print(f"Total ICDR nonconsensus length is {len(non_consensus)}, as follows {non_consensus}")

# DME voting: take mode (0 or 1 only)
#     dme_cols = [col for col in df.columns if "DME" in col]
    df["DME_voted"] = df[dme_cols].mode(axis=1)[0].astype(int)
    print("saving the csv file")
    df.to_csv("abramoff_expert_ICDR_grades_voted.csv")
    return df

def calculate_scores(patient_df):
    scores_dict = {
                "accuracy" : accuracy_score(patient_df['ground_truth_rDR'], patient_df['predicted_rDR']),
                "precision" : precision_score(patient_df['ground_truth_rDR'], patient_df['predicted_rDR']),
                "recall" : recall_score(patient_df['ground_truth_rDR'], patient_df['predicted_rDR']),  # Sensitivity,
                "specificity" : recall_score(patient_df['ground_truth_rDR'], patient_df['predicted_rDR'], pos_label=0),
                "f1" : f1_score(patient_df['ground_truth_rDR'], patient_df['predicted_rDR']),
                "conf_matrix" : confusion_matrix(patient_df['ground_truth_rDR'], patient_df['predicted_rDR'])
                }
    return scores_dict

# def grade_outputs(predictions_df,path_joiner="",output_pdf = None):
def print_scores(scores_dict):
    """now just prints the stats for the already generated scores dictionary"""

    """formerly included
    merged_df,patient_df = make_grading_dfs(predictions_df,path_joiner="",output_pdf = None)
    scores_dict = calculate_scores(patient_df)
    """
   
    # Display results
    print("Metrics for patient-level analysis")
    print(f"Accuracy: {scores_dict['accuracy']:.2f}")
    print(f"Precision: {scores_dict['precision']:.2f}")
    print(f"Sensitivity (Recall): {scores_dict['recall']:.2f}")
    print(f"Specificity: {scores_dict['specificity']:.2f}")
    print(f"F1 Score: {scores_dict['f1']:.2f}")
    print("Confusion Matrix:")
    print(scores_dict['conf_matrix'])
    
    if sum(scores_dict['conf_matrix'].shape)>=4:
        print("               Predicted Normal    Predicted Disease")
        print(f"Actual Normal       {scores_dict['conf_matrix'][0, 0]:>5}               {scores_dict['conf_matrix'][0, 1]:>5}")
        print(f"Actual Disease      {scores_dict['conf_matrix'][1, 0]:>5}               {scores_dict['conf_matrix'][1, 1]:>5}")

def compare_to_expert_grades(patient_df,expert_grades_df):
    """3/22/25: this function takes the pt level predictions, and shows the distribuion of expert grades (voted
    standard) for those indices. Note you need to prefilter for specified indices (e.g. all FN)"""

    patient_df["id_int"] = patient_df["examid"].str.extract(r'(\d+)$').astype(int)
    merged = patient_df.merge(expert_grades_df, left_on="id_int", right_on="unique imageid", how="inner")
    print(merged[["id_int","ICDR_voted","DME_voted"]])
    print(merged["ICDR_voted"].value_counts())
    print(merged["DME_voted"].value_counts())

def make_and_save_images(patient_df,merged_df,output_pdf):
    figs = make_pt_level_plot(patient_df,merged_df)
    roc_fig = None
    if patient_df['prob_of_true'].notna().any():
        roc_fig = plot_roc_curve(patient_df)
        figs.insert(0,roc_fig)
#     pdf_or_None = plot_relevant_images(merged_df,output_pdf=output_pdf) # Returns none if not a pdf
    with PdfPages(output_pdf) as pdf:
        for fig in figs:
            pdf.savefig(fig)
    return roc_fig



def plot_relevant_images(merged_df,path_joiner=""):
    """This performs a random 3x3 plotting for each of tp,tn,fp,fn. At the image not patient level.

    output_pdf is a file_location"""

# Assuming 'ground_truth_rDR' and 'predicted_rDR' are the columns for ground truth and predictions
# and 'image_path' contains the file paths to the images.

# Calculate the confusion matrix

# Define helper functions to filter each category
    def get_indices(df, condition):
        return df[condition].index

# True Positives (TP): ground_truth = 1 and predicted = 1
    tp_indices = get_indices(merged_df, (merged_df['ground_truth_rDR'] == 1) & (merged_df['predicted_rDR'] == 1))

# True Negatives (TN): ground_truth = 0 and predicted = 0
    tn_indices = get_indices(merged_df, (merged_df['ground_truth_rDR'] == 0) & (merged_df['predicted_rDR'] == 0))

# False Positives (FP): ground_truth = 0 but predicted = 1
    fp_indices = get_indices(merged_df, (merged_df['ground_truth_rDR'] == 0) & (merged_df['predicted_rDR'] == 1))

# False Negatives (FN): ground_truth = 1 but predicted = 0
    fn_indices = get_indices(merged_df, (merged_df['ground_truth_rDR'] == 1) & (merged_df['predicted_rDR'] == 0))

    def create_image_plot(indices, title, max_images=9):
        selected_indices = np.random.choice(indices, size=min(len(indices), max_images), replace=False)
        fig = plt.figure(figsize=(20, 20))
        for i, idx in enumerate(selected_indices):
            img = plt.imread(os.path.join(path_joiner,merged_df.loc[idx, 'image_name']))
            ax = plt.subplot(3, 3, i + 1)
            img_n = merged_df.loc[idx, 'image_name'].split('/')[-1]
            if 'parsed_json.ME' in merged_df.columns:
                ax.set_title(f"{img_n} mtm?: {merged_df.loc[idx,'parsed_json.mtmDR']}, DME: {merged_df.loc[idx,'parsed_json.ME']}")
            else:
                ax.set_title(f"{img_n} mtm?: {merged_df.loc[idx,'parsed_json.mtmDR']}")
            plt.imshow(img)
            plt.axis('off')
        plt.suptitle("Image-level "+title)
        return fig


# Display images for each category
    # Save all plots to a single PDF
    figs = [create_image_plot(tp_indices, "True Positives (TP)"),
            create_image_plot(tn_indices, "True Negatives (TN)"),
            create_image_plot(fp_indices, "False Positives (FP)"),
            create_image_plot(fn_indices, "False Negatives (FN)")]
    return figs



def pt_level_plot(indices, patient_df,merged_df, title, RL_dict = {}, path_joiner = "",max_images=12):
    """probably don't need the merged_df"""
    ncols = 4
    nrows = math.ceil(max_images/4)
    scale=3
    fig,ax = plt.subplots(nrows, ncols,figsize=(ncols*scale*0.8, nrows*scale),dpi=600)
    ax = ax.flatten()
    for i, idx in enumerate(indices):
        name = patient_df.loc[idx,'examid']
        if name in RL_dict:
            right_n = name+RL_dict[name][0]
            left_n = name+RL_dict[name][1]
        else:
            right_n = name+".000.png"
            left_n = name+".001.png"

        img_h,img_w = 768,768

        right_index = merged_df[merged_df['image_name'].str.contains(right_n, na=False)].index
        left_index = merged_df[merged_df['image_name'].str.contains(left_n, na=False)].index

        right_img = plt.imread(os.path.join(path_joiner,merged_df.loc[right_index,'image_name'].item()))
        left_img = plt.imread(os.path.join(path_joiner,merged_df.loc[left_index,'image_name'].item()))
        print(f"Right size: {right_img.shape}, Left size: {left_img.shape}")
        right_img = cv2.resize(right_img,(img_h,img_w))
        left_img = cv2.resize(left_img,(img_h,img_w))
        print(f"Right size: {right_img.shape}, Left size: {left_img.shape}")
# 
#         if 'parsed_json.ME' in merged_df.columns:
#         else:
#             ax.set_title(f"{img_n} mtm?: {merged_df.loc[idx,'parsed_json.RDR']}")
        ax[i*2].imshow(right_img,aspect='auto')
#         ax[i*2].set_title(f"{name} mtm: {merged_df.loc[right_index,'parsed_json.mtmDR'].item()}, DME: {merged_df.loc[right_index,'parsed_json.ME'].item()}")
#         ax[i*2].set_title(f"{name} mtm: {merged_df.loc[right_index,'parsed_json.mtmDR'].item()}")
        ax[i*2+1].imshow(left_img,aspect='auto')
#         ax[i*2+1].set_title(f"{name} mtm: {merged_df.loc[left_index,'parsed_json.mtmDR'].item()}")
        plt.axis('off')

    for ax_i in ax:
        ax_i.set_xticks([])
        ax_i.set_yticks([])
        ax_i.set_xticklabels([])
        ax_i.set_yticklabels([])
        ax_i.set_aspect('equal')
        ax_i.spines['top'].set_visible(False)
        ax_i.spines['right'].set_visible(False)
        ax_i.spines['left'].set_visible(False)
        ax_i.spines['bottom'].set_visible(False)
        ax_i.set_frame_on(False)
    plt.subplots_adjust(left=0, right=1, top=0.95, bottom=0.05, wspace=0, hspace=0)

# Remove margins
#     plt.suptitle("Patient-Level "+title)
    return fig


def get_indices(df, condition):
    return df[condition].index

def make_pt_level_plot(patient_df,merged_df,path_joiner=""):
    """This performs a random 3x3 plotting for each of tp,tn,fp,fn. At the patient level.
    You need both dfs to get the image sources though.

    If you have indices in mind, don't use this function nad instead just use the pt_level_plot above directly.

    output_pdf is a file_location"""

# Assuming 'ground_truth_rDR' and 'predicted_rDR' are the columns for ground truth and predictions
# and 'image_path' contains the file paths to the images.

# Calculate the confusion matrix

# Define helper functions to filter each category

# True Positives (TP): ground_truth = 1 and predicted = 1
    index_dict = {
        "tp_indices" : get_indices(patient_df, (patient_df['ground_truth_rDR'] == 1) & (patient_df['predicted_rDR'] == 1)),
        "tn_indices" : get_indices(patient_df, (patient_df['ground_truth_rDR'] == 0) & (patient_df['predicted_rDR'] == 0)),
        "fp_indices" : get_indices(patient_df, (patient_df['ground_truth_rDR'] == 0) & (patient_df['predicted_rDR'] == 1)),
        "fn_indices" : get_indices(patient_df, (patient_df['ground_truth_rDR'] == 1) & (patient_df['predicted_rDR'] == 0)),
    }
    titles = {
        "tp_indices" : "True Positives (TP)",
        "tn_indices" : "True Negatives (TN)",
        "fp_indices" : "False Positives (FP)",
        "fn_indices" : "False Negatives (FN)"
    }

    max_pts = 6
    figs = [ ]
    for k,indices in index_dict.items():
        selected_indices = np.random.choice(indices, size=min(len(indices), max_pts), replace=False)
        fig = pt_level_plot(selected_indices,patient_df,merged_df,titles[k])
        figs.append(fig)
    return figs


# Calculate ROC curve and AUC
def make_roc_fig(fpr,tpr,roc_auc):
    fig = plt.figure()
    plt.plot(fpr, tpr, color="darkorange", lw=2, label=f"ROC curve (area = {roc_auc:.2f})")
    plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Receiver Operating Characteristic")
    plt.legend(loc="lower right")
    return fig


def plot_roc_curve(patient_df):
    """output_pdf is a file string. 
    this also works at the patient level and thus the patient_df takes the max of the predicted_prob, bc if either
    eye satisfies the threshold, it's predicted as a case"""
    fpr, tpr, thresholds = roc_curve(patient_df['ground_truth_rDR'], patient_df['prob_of_true'])
    roc_auc = auc(fpr, tpr)

# Plot the ROC curve
    fig = make_roc_fig(fpr,tpr,roc_auc)
    return fig
#     if output_pdf:
#         with PdfPages(output_pdf) as pdf:
#             pdf.savefig(make_roc_fig(fpr,tpr))
#             return pdf
#     else:
#         make_roc_fig(fpr,tpr)
#         plt.show()
#         return None

def dict_match(d, model_name, instruction_part):
    # Iterate over keys and check if both substrings are present
    for key in d:
        if model_name in key and instruction_part in key:
            return d[key]  # Return the matching value
    return None  # If no match found

def roc_multiplot(pt_df_dict,
                  model_names = ["gpt-4o-2024-08-06","gpt-4o-mini","grok-2-vision-1212","gemini-1.5-pro"],
                  prompt_names = ["system_header_basic",
                        "system_header_with_background",
                        "system_header_with_background__few_shot_with_background"],
                  annotate = False,
                  optimal_point_adds = {("gpt-4o-2024-08-06","system_header_with_background"):None},
                  ):
    """Take as input a dict of dicts of of pt-level dfs, and output a figure with multiple ROCs
    e.g. {"GPT...":{system_basic:pt_df,system basic + bg:pt_df}},
    if supplied will be a set of tuples
    optimal_point_adds is a dict of tuples mapped to a NOne or prespecified sensitivity_threshold"""

    ncols = min(len(model_names),3) # Make it most 3 cols
    nrows = math.ceil(len(model_names)/ncols)
    scale = 5
    fig,ax = plt.subplots(nrows,ncols,figsize = (scale*(ncols+0.5),scale*nrows))

    ax = ax.flatten()
    for i,model_name in enumerate(model_names):
        for j,prompt_name in enumerate(prompt_names):
            patient_df = dict_match(pt_df_dict,model_name,prompt_name)
            if patient_df is None:
                continue
            prompt_name_short = C.prompt2promptname[prompt_name]
            fpr, tpr, thresholds = roc_curve(patient_df['ground_truth_rDR'], patient_df['prob_of_true'])
            roc_auc = auc(fpr, tpr)
            line_obj, = ax[i].plot(fpr, tpr, lw=1.5,alpha = 0.6, label=f"{prompt_name_short}: AUROC = {roc_auc:.2f}") 
            line_color = line_obj.get_color()  # Get the color assigned to this line

            if annotate:
                print(f"for {model_name} and {prompt_name}")
                annotate_roc(ax[i],line_color,fpr, tpr, thresholds,j,patient_df)
            elif (model_name,prompt_name) in optimal_point_adds:
                optimal_idx = add_optimal_point(ax[i],fpr,tpr,optimal_point_adds[(model_name,prompt_name)])
                print(f"the threshold for the optimal idx is {thresholds[optimal_idx]}, giving sensitivity/specificity of {tpr[optimal_idx]} :: {fpr[optimal_idx]}")

        ax[i].plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
        ax[i].set_xlim([0.0, 1.0])
        ax[i].set_ylim([0.0, 1.0])
        ax[i].set_xlabel("False Positive Rate")
        ax[i].set_ylabel("True Positive Rate")
        ax[i].set_title(f"{model_name}")
        ax[i].legend(loc="lower right")
    return fig,ax


def annotate_roc(ax,line_color,fpr, tpr, thresholds,j,patient_df=None):
    """adds some thresholding info if we want. j is just the current index you're on fot he line plotting"""
    def make_single_annotation(index,ax,line_color,fpr, tpr, threshold,j,patient_df=None):
        annotation_text = f"{threshold:.2e}"

        if patient_df is not None:
            df = eval_utils.adjust_preds_to_set_point(patient_df,set_point = threshold)
            conf_mat = confusion_matrix(df['ground_truth_rDR'], df['predicted_rDR'])
            FP = conf_mat[0,1]
            FN = conf_mat[1,0]
            annotation_text = annotation_text + f"  FP:{FP},FN:{FN}"

        ax.annotate(annotation_text, (fpr[index], tpr[index]), textcoords="offset points",
                       xytext=(5,10*j), ha='left',
                       fontsize=8,color=line_color)


    if patient_df is not None:
        patient_df = patient_df.copy()

    scatter_indices = np.linspace(0, len(thresholds) - 1, num=5, dtype=int)  # Pick ~10 points for clarity
    ax.scatter(fpr[scatter_indices], tpr[scatter_indices], color=line_color, s=10, zorder=3)

    # Annotate threshold values

    for k in scatter_indices:
        threshold = thresholds[k]
        make_single_annotation(k,ax,line_color,fpr, tpr, threshold,j,patient_df)

    J_scores = tpr - fpr
    optimal_idx = np.argmax(J_scores)  # Index of max J
    optimal_threshold = thresholds[optimal_idx]

    print(f"For Youden's J, the optimal idx and threshold is {optimal_idx},{optimal_threshold}. This gives a sensitivity of {tpr[optimal_idx]} and a specificity of {1-fpr[optimal_idx]}\n")
    make_single_annotation(optimal_idx,ax,'r',fpr, tpr, optimal_threshold,j,patient_df)
    ax.scatter(fpr[optimal_idx], tpr[optimal_idx], color='r', s=10, zorder=3)

def add_optimal_point(ax,fpr, tpr,sensitivity_threshold=None):
    """add the Youden stat optimal point as a scatter"""
    if sensitivity_threshold is None:
        optimal_idx = np.argmax(tpr - fpr)  # Index of max J
    else:
        optimal_idx = np.argmax(tpr>=sensitivity_threshold) if np.any(tpr >= sensitivity_threshold) else None
        assert optimal_idx is not None
    ax.scatter(fpr[optimal_idx], tpr[optimal_idx], color='r', s=10, zorder=3)
    return optimal_idx




true_variations=("true", "True", "TRUE","\ttrue","\tTrue",",true")
false_variations=("false", "False", "FALSE","\tfalse","\tFalse",",false","fals")
true_set = set(true_variations)
false_set = set(false_variations)

def get_correct_top_logprobs(response):
    """as of 2/13/25, we've used a different json output structure and it's causing errors bc T/F for mtmDR is no longer
    a consistent index. There are more newline chars, etc. So instead we search thru top few tokens to identify which
    has intersection with the variations"""

    top_n = 3
    bool_index = None
    max_intersection_length = 0
    for i in range(len(response.choices[0].logprobs.content)):
        top_tokens = [e.token.strip() for e in response.choices[0].logprobs.content[i].top_logprobs[0:top_n]]
        intersection_length = len(set(top_tokens).intersection(true_set)) + len(set(top_tokens).intersection(false_set))
        if intersection_length>max_intersection_length:
            bool_index = i
            max_intersection_length = intersection_length

    if bool_index is None:
        return None
    top_logprobs = response.choices[0].logprobs.content[bool_index].top_logprobs  # 2/13/25is now the 6th field in the output after new prompt
    return top_logprobs


def extract_true_probability(response, true_variations=true_variations):
    """the true_variations can be explored by printing the following dictOfCounters. on the 60 val set we have
11: 11 = Counter({' false': 60, ' true': 60, ' ': 60, ' null': 60, ' "': 56, 'false': 53, ' False': 52, '\tfalse': 47, '\xa0': 46, '<|end|>': 36, 'true': 20, ' True': 20, '\ttrue': 17, ',true': 6, ' fals': 4, ' **': 3})
    """
    # Ensure JSON parsing is successful

    # Get log probabilities for each token from the response
#     top_logprobs = response.choices[0].logprobs.content[11].top_logprobs  # 12th field in the output
#     try:
# #         top_logprobs = response.choices[0].logprobs.content[7].top_logprobs  # is now the 8th field in the output after new prompt
#         top_logprobs = response.choices[0].logprobs.content[5].top_logprobs  # 2/13/25is now the 6th field in the output after new prompt
#     except IndexError as e:
#         return None


    top_logprobs = get_correct_top_logprobs(response)  # 2/13/25is now the 6th field in the output after new prompt
    if top_logprobs is None:
        print(f"Unable to get top logprobs for this sample with response message of {response.choices[0].message.content}")
        return None
#     import pdb; pdb.set_trace()
    # Initialize cumulative probability for "true"
    cumulative_logprob = None
    for e in top_logprobs:
        token, logprob = e.token, e.logprob
        if token.strip() in true_variations:  # Strip spaces and match to true variations
            if cumulative_logprob is None:
                cumulative_logprob = logprob
            else:
                cumulative_logprob = np.logaddexp(cumulative_logprob, logprob)  # Sum in log-space

    # Convert log-prob to regular probability
    true_probability = np.exp(cumulative_logprob) if cumulative_logprob is not None else 0.0
    return true_probability

def aftermarket_true_prob_extraction(df,true_variations=true_variations,false_variations = false_variations):
    """in case you screwed up, use this function to get the probability after the fact. Return a pd series"""
    true_probability_list = []
    boolean_list_index = None
    for i in range(len(df.loc[0,'raw_response.choices'][0]['logprobs']['content'])): # using the first row to get this
        tokens = [e['token'] for e in df.loc[0,'raw_response.choices'][0]['logprobs']['content'][i]['top_logprobs']]
        true_intersect = set([e.strip() for e in tokens[:3]]).intersection(true_variations)
        false_intersect = set([e.strip() for e in tokens[:3]]).intersection(false_variations)
        if len(true_intersect) >0 or len(false_intersect) >0:
            boolean_list_index = i
            break
    if not boolean_list_index:
        raise Exception
    print(f"Using boolean_list_index of {i}")
    for i in range(df.shape[0]):
        top_logprobs = df.loc[i,'raw_response.choices'][0]['logprobs']['content'][boolean_list_index]['top_logprobs']

        cumulative_logprob = None
        for e in top_logprobs:
            token, logprob = e['token'], e['logprob']
            if token.strip() in true_variations:  # Strip spaces and match to true variations
                if cumulative_logprob is None:
                    cumulative_logprob = logprob
                else:
                    cumulative_logprob = np.logaddexp(cumulative_logprob, logprob)  # Sum in log-space

        # Convert log-prob to regular probability
        true_probability = np.exp(cumulative_logprob) if cumulative_logprob is not None else 0.0
        true_probability_list.append(true_probability)
    return true_probability_list




class dictOfCounters():
    """takes a response as input, and creates a dict of index:counter of the top-logprob tokens"""
    def __init__(self):
        self.counter_dict = defaultdict(Counter)

    def update(self,response_choices_0):
        """takes response.choices[0] and updates this complicated dict of counters"""
        if not hasattr(response_choices_0, 'logprobs'):
            raise Exception
        for i,word_pred in enumerate(response_choices_0.logprobs.content):
            self.counter_dict[i].update([e.token for e in word_pred.top_logprobs])

    def __str__(self):
        return "\n".join(f"{idx}: {key} = {value}" for idx, (key, value) in enumerate(self.counter_dict.items()))



def reopen_file_and_get_processed_images(output_file_path):
    """requires validly formated json in the first place. Removes the ] and puts a , in the same line as the }"""
    with open(output_file_path, "r+") as output_file:
        # Load JSON to validate it
        existing_data = json.load(output_file)
        processed_images = {entry["image_name"] for entry in existing_data}

        # Read the file content
        output_file.seek(0)  # Move to the beginning
        content = output_file.read()


        # Strip trailing whitespace or newlines
        stripped_content = content.rstrip()

        # Rewrite the file with stripped content
        output_file.seek(0)  # Go back to the beginning
        output_file.write(stripped_content)
        output_file.truncate()  # Remove anything after the new content

        # Check if the file ends with a closing bracket
        if stripped_content.endswith("]"):
            # Move to just before the closing bracket
            output_file.seek(len(stripped_content) - 1)
            output_file.write(",")  # Add a comma to allow appending
    return processed_images

def clean_json_end(output_file_path):
    """remove the last non-white-space character to trim the last comma"""

    with open(output_file_path, "rb+") as output_file:
        output_file.seek(-1, os.SEEK_END)  # Go to the last byte
        if output_file.read(1) == b'\n':  # Check if it's a newline
            output_file.seek(-1, os.SEEK_END)  # Move back again
            output_file.truncate()  # Remove the newline
        output_file.seek(-1, os.SEEK_END)  # Move back again
        current_position = output_file.tell()
        cur_char = output_file.read(1)
        print(cur_char)

        output_file.seek(current_position)
        i=0
        while cur_char in (b' ', b'\n', b','):  # Check for comma or whitespace
            print(cur_char)
            output_file.seek(-1, os.SEEK_CUR)  # Move back two bytes to recheck previous character
            current_position = output_file.tell()
            cur_char = output_file.read(1)
            output_file.seek(current_position)
            i+=1

        if i==0:
            return
        output_file.seek(1, os.SEEK_CUR)  # Move forward one byte to overwrite correctly
        output_file.write(b"\n]")  # Write the closing bracket
        output_file.truncate()  # Remove any leftover content