library(readr)
library(dplyr)
library(ggplot2)
library(pROC)


setwd("/Users/matthewhunt/Research/Iowa_Research/Abramoff_Projects/LLM_Messidor_Study/")
df <- read_csv("pt_df_dict_combined.csv")
# Now split df by the key column:
my_list <- split(df, df$dict_key)


parse_key <- function(key_string) {
  idx <- regexpr("__", key_string, fixed = TRUE)
  if (idx == -1) {
    # No "__" found; return the entire string as "model"
    return(list(model = key_string, prompt = NA_character_))
  } else {
    model_name  <- substr(key_string, 1, idx - 1)
    prompt_name <- substr(key_string, idx + 2, nchar(key_string))
    return(list(model = model_name, prompt = prompt_name))
  }
}



df_all <- data.frame()

for (key in names(my_list)) {
  # Extract the data frame
  patient_df <- my_list[[key]]
  
  # Parse model/prompt from the name
  parsed <- parse_key(key)
  model_name  <- parsed$model
  prompt_name <- parsed$prompt
  
  # Compute the ROC curve
  ro <- roc(
    response  = patient_df$ground_truth_rDR,
    predictor = patient_df$prob_of_true,
    direction = "<"  # if higher scores => more likely positive
  )
  
  # AUC for labeling
  ro_auc <- auc(ro)
  
  # Extract TPR/FPR across all thresholds
  # coords(..., transpose=FALSE) returns a data frame
  coords_df <- coords(ro, x = "all", ret = c("threshold", "tpr", "fpr"), transpose=FALSE)
  
  # Add columns for model, prompt, and AUC (repeated for each row)
  coords_df$model  <- model_name
  coords_df$prompt <- prompt_name
  coords_df$auc    <- as.numeric(ro_auc)
  
  coords_df <- coords_df[order(-coords_df$threshold),]
  # Combine
  df_all <- bind_rows(df_all, coords_df)
}

# Optionally, create a new label column that includes AUC in the legend:

clean_prompt_names <- c(
  "system_header_basic"="Minimal",
  "system_header_with_background"="Background",
  "system_header_with_background__few_shot_with_background"="Few-shot"
  
)


# Add a cleaned-up label for each row
df_all <- df_all %>%
  mutate(
    prompt = clean_prompt_names[prompt],  # Map raw names to readable labels
    prompt_label = paste0(prompt, " (AUC=", round(auc, 2), ")")  # Final legend text
  )

df_all <- df_all %>%
  mutate(prompt= factor(prompt, levels = c("Minimal", 
                                                        "Background", 
                                                        "Few-shot")))


df_labels <- df_all %>%
  group_by(model, prompt) %>%
  summarize(
    auc_label = first(sprintf("%s (AUC=%.2f)", prompt, auc)),
    .groups = "drop"
  ) %>%
  group_by(model) %>%
  mutate(
    # position each label differently so they stack
    label_rank = row_number(),
    x = 0.65,                       # fixed x
    y = 0.15 - 0.05 * (label_rank - 1)  
    # e.g. first label at y=0.15, second at 0.10, etc.
  ) %>%
  ungroup()

# 2) Plot the main ROC curves
ggplot(df_all, aes(
  x = fpr, y = tpr,
  
  color = prompt,
  group = interaction(model, prompt)
)) +
  geom_line(size = 0.8) +
  # diagonal reference line
  geom_abline(intercept = 0, slope = 1, linetype = "dashed", color = "gray",alpha=1) +
  # one facet per model
  facet_wrap(~ model,ncol=1) +
  # 3) Add text labels in bottom-right
  geom_text(
    data = df_labels,
    aes(x = x, y = y, 
        label = auc_label,
        color = prompt,
        ),  # match text color to ROC color
    size = 5,        # adjust as desired
    show.legend = FALSE  # hide separate legend for this geom
  ) +
  # 4) Force x and y from 0..1
  scale_x_continuous(limits = c(0, 1)) +
  scale_y_continuous(limits = c(0, 1)) +
  # 5) Remove the standard legend entirely and “gray box” behind facet titles
  theme_bw() +
  theme(
    legend.position = "none",
    strip.background = element_blank()  # remove gray facet strip background
  ) +
  labs(
    x = "False positive rate",
    y = "True positive rate",
  ) + 
  theme(
    # axis.text = element_text(size = 10),        # axis tick labels
    axis.title = element_text(size = 20),       # axis titles
    strip.text = element_text(size = 18),       # facet labels
    plot.title = element_text(size = 18, hjust = 0.5)  # plot title, centered
  )














#### NOW MAKING THE CI Plot #####

# 1) Create the ROC object
generate_roc_with_ci <- function(df_single,name){
  my_roc <- roc(
    response  = df_single$ground_truth_rDR,
    predictor = df_single$prob_of_true,
    direction = "<"  # if higher score => more likely positive
  )
  my_roc_auc = auc(my_roc)
  # Extract the original ROC curve points
  df_roc <- as.data.frame(coords(my_roc, x = "all",
                                 ret = c("threshold","specificty", "tpr", "fpr"),
                                 transpose = FALSE))
  
  df_roc <- df_roc[order(df_roc$threshold), ]
  # df_roc$specificity <- 1-df_roc$fpr
  df_roc$specificity <- round(1 - df_roc$fpr, 4)
  
  # 2) Compute the TPR confidence bands via ci.se
  #    We'll sample specificities across 0..1 (50 points here).
  #    pROC does a bootstrap internally (you can raise boot.n if you want).
  # unique_spec <- as.data.frame(unique(df_roc$specificity))
  
  my_ci_se <- ci.se(
    my_roc, 
    # specificities = seq(0, 1, length.out = 874),
    specificities = df_roc$specificity,
    # specificities = unique_spec,
    conf.level    = 0.95,
    boot.n        = 2000,
    stratified    = TRUE,
    
  )
  
  unique_idx <- which(!duplicated(rownames(my_ci_se)))
  my_ci_se_unique <- my_ci_se[unique_idx, , drop = FALSE]
  
  # "my_ci_se" is an object that includes TPR median & lower/upper CI for each specificity
  # We'll convert it into a data frame for ggplot
  
  df_ci <- as.data.frame(my_ci_se_unique)
  # By default, rownames(df_ci) = the specificity values
  # df_ci$specificity <- as.numeric(rownames(df_ci))
  df_ci <- tibble::rownames_to_column(df_ci, var = "specificity")
  # df_ci$specificity <- as.numeric(sub("^X", "", df_ci$specificity))
  df_ci$specificity <- round(as.numeric(sub("^X", "", df_ci$specificity)), 4)
  
  
  
  df_ci <- df_ci %>%
    rename(
      tpr_lower  = `2.5%`,
      tpr_median = `50%`,
      tpr_upper  = `97.5%`
    ) %>%
    mutate(
      specificity = as.numeric(specificity),
      # fpr         = 1 - specificity
    )
  
  
  # Now combine with df_roc, matching by specificity
  df_roc_synced <- left_join(df_roc, df_ci, by = "specificity")
  
  inds = which(df_roc_synced$tpr_lower >= 0.80)
  idx = inds[length(inds)]
  
  threshold = df_roc_synced[idx,]$threshold
  sensitivity = df_roc_synced[idx,]$tpr
  specificity = 1-df_roc_synced[idx,]$fpr
  
  print(paste0("At the sensitivity threshold the the threshold is",threshold))
  print(paste0("The sensitivity is ",sensitivity))
  print(paste0("The specificity is ",specificity))
  
  
  # If none qualify, idx will be NA
  if (!is.na(idx)) {
    highlight_row <- df_roc_synced[idx, ]
  } else {
    highlight_row <- NULL
  }
  
  df_roc_synced <- df_roc_synced[order(-df_roc_synced$threshold), ]
  p<-ggplot(df_roc_synced, aes(x = fpr, y = tpr)) +
    # 95% CI band
    geom_ribbon(aes(ymin = tpr_lower, ymax = tpr_upper),
                fill = "gray", alpha = 0.3) +
    # Original ROC
    # geom_line(size = 1.2,color = "#00BA38") +
    geom_line(size = 1.2) +
    
    geom_point(data = highlight_row,
               aes(x = fpr, y = tpr),
               color = "red", size = 3) +
    geom_abline(intercept = 0, slope = 1, linetype = "dashed") +
    coord_cartesian(xlim = c(0, 1), ylim = c(0, 1)) +
    theme_bw() +
    theme(
      strip.background = element_blank(),  # remove gray facet strip background
      legend.position = "none",
      plot.title = element_text(hjust = 0.5)  # Center title
    ) +
    labs(
      # title = name,
      x = "False positive rate",
      y = "True positive rate"
    )
  
  # Check if TPR is non-decreasing:
  is_monotonic <- all(diff(df_roc_synced$tpr) >= 0)
  
  if (!is_monotonic) {
    cat("Warning: The TPR decreases at some points along the ROC curve.\n")
    # Identify indices where the difference is negative:
    dip_indices <- which(diff(df_roc_synced$tpr) < 0)
    print(df_roc_synced[dip_indices, ])
  } else {
    cat("TPR is monotonic non-decreasing as FPR increases.\n")
  }
  result = list("plot"=p,
                "threshold"=threshold,
                "sensitivity"=sensitivity,
                "specificity"=specificity,
                "auc"=my_roc_auc
                )
  return(result) 
}


fig_list=list()
results_list = list()
for (i in seq_along(names(my_list))){
  name = names(my_list)[[i]]  
  df_single = my_list[[name]]
  
  print(paste0("now processing for ",name))
  result = generate_roc_with_ci(df_single,name)
  p <- result[["plot"]]
  result["plot"] <- NULL
  print(p)
  fig_list[[name]] <- p
  results_list[[name]] <- result 
  # list(result[["sensitivity"]],result[["specificity"]],result[["auc"]],result[["threshold"]])
}

# Now take the following plots and arrange them into a grid format
library(gridExtra)
library(grid)

unique_prompts <- c("system_header_basic","system_header_with_background","system_header_with_background__few_shot_with_background")
unique_models <- c("gpt-4o-2024-08-06","gpt-4o-mini","grok-2-vision-1212")

nrow_plots <- length(unique_prompts)
ncol_plots <- length(unique_models)

# Create an empty matrix (list) with extra row and column for headers
total_rows <- nrow_plots + 1  # first row for column headers
total_cols <- ncol_plots + 1  # first column for row headers
plot_matrix <- matrix(list(), nrow = total_rows, ncol = total_cols)
table_matrix <- matrix(list(), nrow = total_rows, ncol = total_cols)

# Top-left cell left empty
plot_matrix[[1,1]] <- nullGrob()

# Fill top row with model names
fontsize=10
for (j in seq_len(ncol_plots)) {
  plot_matrix[[1, j + 1]] <- textGrob(clean_prompt_names[unique_prompts[j]], rot=45, gp = gpar(fontsize = fontsize, fontface = "bold"))
  table_matrix[[1, j + 1]] <- clean_prompt_names[unique_prompts[j]]
}

# Fill left column with prompt names (rotated)
for (i in seq_len(nrow_plots)) {
  plot_matrix[[i + 1, 1]] <- textGrob(unique_models[i], rot = 45, gp = gpar(fontsize = fontsize, fontface = "bold"))
  table_matrix[[i + 1, 1]] <- unique_models[i]
}

# Fill the remaining cells with the corresponding plots
for (i in seq_len(nrow_plots)) {
  for (j in seq_len(ncol_plots)) {
    # Construct the key as "model__prompt"
    key <- paste0(unique_models[i], "__", unique_prompts[j])
    plot_matrix[[i + 1, j + 1]] <- fig_list[[key]]
    if (!is.null(results_list[[key]])) {
      # Unpack the results (each is a list element)
      sens <- round(as.numeric(results_list[[key]][1]), 2)
      spec <- round(as.numeric(results_list[[key]][2]), 2)
      auc_val <- round(as.numeric(results_list[[key]][3]), 2)

      table_matrix[[i + 1, j + 1]] <- sprintf("%.2f / %.2f (%.2f)", sens, spec, auc_val)
    } else {
      table_matrix[[i + 1, j + 1]] <- ""
    }
  }
}

print("The sensitivity/specificity (AUC) tabele for the optimal set point is below")
write.table(t(table_matrix), file = "", sep = "\t", 
            row.names = FALSE, col.names = FALSE, quote = FALSE)

# Arrange the grid
final_grid <- arrangeGrob(grobs = plot_matrix, nrow = total_rows, ncol = total_cols)
grid.newpage()
grid.draw(final_grid)


#### 2.1 Now plot the single curve also placing the provider points as judged against the reference std. 

expert_sens_and_spec <- function(df){
  # reports the sens and spec of the experts in terms of the GT (voted_GT)
  results <- data.frame(expert=character(), sensitivity=numeric(), specificity=numeric())
  for (n in c("Han_", "Walker_", "Williams_")) {
    ns <- c(n, "voted_GT")
    sub_df <- df[, grepl(n, names(df)) | names(df) == "voted_GT"]
    names(sub_df) <- gsub(n, "", names(sub_df))  # remove prefix from col names
    
    sub_df$grader_mtmDR <- as.integer(sub_df$ICDR > 1 | sub_df$DME > 0)
    
    TP <- sum(sub_df$grader_mtmDR == 1 & sub_df$voted_GT == 1)
    TN <- sum(sub_df$grader_mtmDR == 0 & sub_df$voted_GT == 0)
    FP <- sum(sub_df$grader_mtmDR == 1 & sub_df$voted_GT == 0)
    FN <- sum(sub_df$grader_mtmDR == 0 & sub_df$voted_GT == 1)
    
    sensitivity <- TP / (TP + FN)
    specificity <- TN / (TN + FP)
    
    results <- rbind(results, data.frame(expert=n, sensitivity=sensitivity, specificity=specificity))
  }
  
  return(results)
}

# expert_sens_and_spec2 <- function(df){
#   # reports the sens and spec of the experts in terms of the GT (voted_GT)
#   df = copy(df)
#   df$ICDR_voted<-NULL
#   df$DME_voted<-NULL
#   
#   results <- data.frame(expert=character(), sensitivity=numeric(), specificity=numeric())
#   names_vec = c("Han_", "Walker_", "Williams_")
#   for (n in names_vec) {
#     other_experts_df <- # select those other non-n columns
#     remaining_expert_df <- #
#     other_experts_df$voted_GT
#     
#     # construct this column
#     other_experts_df[["voted_GT"]]<-as.integer(expert_grades_df$ICDR_voted>1 | expert_grades_df$DME_voted>0)
#     
#     
#     ns <- c(n, "voted_GT")
#     sub_df <- df[, grepl(n, names(df)) | names(df) == "voted_GT"]
#     names(sub_df) <- gsub(n, "", names(sub_df))  # remove prefix from col names
#     
#     sub_df$grader_mtmDR <- as.integer(sub_df$ICDR > 1 | sub_df$DME > 0)
#     
#     TP <- sum(sub_df$grader_mtmDR == 1 & sub_df$voted_GT == 1)
#     TN <- sum(sub_df$grader_mtmDR == 0 & sub_df$voted_GT == 0)
#     FP <- sum(sub_df$grader_mtmDR == 1 & sub_df$voted_GT == 0)
#     FN <- sum(sub_df$grader_mtmDR == 0 & sub_df$voted_GT == 1)
#     
#     sensitivity <- TP / (TP + FN)
#     specificity <- TN / (TN + FP)
#     
#     results <- rbind(results, data.frame(expert=n, sensitivity=sensitivity, specificity=specificity))
#   }
#   return(results)
# }

add_expert_points <- function(p, expert_sens_and_spec_df) {
  # Plot expert points: x=FPR (1 - specificity), y=sensitivity
  p <- p + 
    geom_point(
      data = expert_sens_and_spec_df, 
      aes(x = 1 - specificity, y = sensitivity),
      color = "blue", size = 3
    )
  
  # Sort by FPR
  df_sorted <- expert_sens_and_spec_df[order(1 - expert_sens_and_spec_df$specificity), ]
  
  # Build vectors for trapezoidal integration, adding (0, 0) and (1, 1)
  x <- c(0, 1 - df_sorted$specificity, 1) 
  y <- c(0, df_sorted$sensitivity, 1)
  
  # Trapezoidal rule
  auc <- sum( diff(x) * (head(y, -1) + tail(y, -1)) / 2 )
  
  cat("Trapezoidal expert AUC:", round(auc, 3), "\n")
  
  return(p)
}


expert_grades_df = read_csv("abramoff_expert_ICDR_grades_voted.csv")
expert_grades_df[["voted_GT"]]<-as.integer(expert_grades_df$ICDR_voted>1 | expert_grades_df$DME_voted>0)
p<-fig_list[["gpt-4o-2024-08-06__system_header_with_background"]]
expert_sens_and_spec_df <- expert_sens_and_spec(expert_grades_df)

# Hand-crafted from the 2013 paper...
expert_sens_and_spec_df <- as.data.frame(list("expert"=c("1","2","3") ,
                                              "sensitivity" = c(0.803,0.714,0.910),
                                              "specificity" = c(0.983,0.985,0.950))
)

p<-add_expert_points(p,expert_sens_and_spec_df)
p + labs(title="Model: gpt-4o-2024-08-06, prompting strategy: Background")


#### 3. Now making the UpSetR Plot showing the intersections of the false negatives. 
# Would consider the ComplexUpset plot to illlustrate the breakdown of the different classes of severity this way too. 

# extractConfMatrixSets <- function(
#   dfs_list,
#   results_list,
#   measure_col = "prob_of_true",
#   label_col   = "ground_truth_rDR"
# ) {
#   # Create a named list where each element is another list of TP, FP, TN, FN IDs
#   sets_list <- lapply(names(dfs_list), function(nm) {
#     df  <- dfs_list[[nm]]
#     thr <- results_list[[nm]]$threshold
#     
#     # Binary predictions based on threshold
#     pred <- ifelse(df[[measure_col]] >= thr, 1, 0)
#     true <- df[[label_col]]
#     
#     list(
#       TP = df$examid[pred == 1 & true == 1],
#       FP = df$examid[pred == 1 & true == 0],
#       TN = df$examid[pred == 0 & true == 0],
#       FN = df$examid[pred == 0 & true == 1]
#     )
#   })
#   
#   names(sets_list) <- names(dfs_list)
#   sets_list
# }

relabel_preds <- function(dfs_list,
                            results_list,
                            measure_col = "prob_of_true",
                            label_col   = "ground_truth_rDR"){
  # returns a new dfs_list that includes a pred column as well
  dfs_list_with_preds <- lapply(names(dfs_list), function(nm) {
    df  <- dfs_list[[nm]]
    thr <- results_list[[nm]]$threshold
    
    # Binary predictions based on threshold
    pred <- ifelse(df[[measure_col]] >= thr, 1, 0)
    df[["pred"]] <- pred
    df
  })
  names(dfs_list_with_preds) <- names(dfs_list)
  return(dfs_list_with_preds)
}

extractConfMatrixSets <- function(dfs_list_with_preds){
  sets_list <- lapply(dfs_list_with_preds, function(df){
    pred = df["pred"]
    true = df["ground_truth_rDR"] 
    print(df)
    list(
      TP = df$examid[ pred == 1 & true == 1],
      FP = df$examid[pred == 1 & true == 0],
      TN = df$examid[pred == 0 & true == 0],
      FN = df$examid[pred == 0 & true == 1]
      )
  })
  names(sets_list) <- names(dfs_list_with_preds)
  sets_list
}

IOU_subset <- function (
  sets_list,
  filter_phrases
) {
  lapply(filter_phrases, function(filter_phrase) { 
    subsets <- sets_list[grepl(filter_phrase, names(sets_list))]
    global_union <- length(unique(unlist(subsets)))
    global_intersection <- length(Reduce(intersect,subsets))
    print(paste0("For ",filter_phrase))
    print(paste0("Global union is ", global_union))
    print(paste0("Global intersection is ", global_intersection))
    print(paste0("IOU is ", global_intersection/global_union))
  })
}

df_list_with_preds <- relabel_preds(my_list, results_list)
conf_matrix_sets <- extractConfMatrixSets(df_list_with_preds)

# conf_matrix_sets <- conf_matrix_sets[grepl("gpt-4o-2024-08-06", names(conf_matrix_sets))]


# 1. Extract just the FN sets for each model:
fn_sets <- lapply(conf_matrix_sets, function(x) x$FN)

model_list <- list("gpt-4o-2024-08-06","gpt-4o-mini", "grok-2-vision-1212")
IOU_subset(fn_sets,model_list)

# Ensure each set is named after the model:
names(fn_sets) <- names(conf_matrix_sets)

# 2. Convert these sets into a matrix format UpSetR can handle
library(UpSetR)
fn_sets_matrix <- fromList(fn_sets)

# 3. Plot the UpSet figure
upset(fn_sets_matrix, 
      sets=rev(names(fn_sets_matrix)),
      order.by = "freq",      # sort intersections by size
      main.bar.color = "black", 
      sets.bar.color = "black")

# Now compare to the expert grades
compare_to_expert_grades <- function(set_list, expert_grades_df) {
  # Extract numeric part from e.g. "IM0002" -> 2
  lapply(names(set_list), function(nm){
    set1=set_list[[nm]]
    set1 <- as.integer(sub(".*?(\\d+)$", "\\1",set1))
    filtered_df <- expert_grades_df[expert_grades_df[["unique imageid"]] %in% set1,]
    # Show selected columns
    cat("\n----", nm, "----\n")
    
    print(filtered_df[, c("unique imageid","ICDR_voted", "DME_voted")])
    
    
    # Print counts
    cat("ICDR grade counts:\n")
    print(table(filtered_df$ICDR_voted))
    
    cat("DME counts:\n")
    print(table(filtered_df$DME_voted))
    x=NULL
  })

  return(NULL)
}


expert_grades_df = read_csv("abramoff_expert_ICDR_grades_voted.csv")
compare_to_expert_grades(fn_sets,expert_grades_df)




# 
# df_ci <- as.data.frame(my_ci_se)
# # By default, row names = specificities, columns = X2.5, X50, X97.5
# df_ci <- tibble::rownames_to_column(df_ci, var = "specificity")
# 
# # Rename columns for clarity
# df_ci <- df_ci %>%
#   rename(
#     tpr_lower  = `2.5%`,
#     tpr_median = `50%`,
#     tpr_upper  = `97.5%`
#   ) %>%
#   mutate(
#     specificity = as.numeric(specificity),
#     fpr         = 1 - specificity
#   )
# 
# # Identify first row where lower CI >= 0.75
# # idx <- which(df_ci$tpr_lower >= 0.75)[1]

# 
# ggplot() +
#   # (a) 95% CI ribbon in gray from bootstrapped values
#   geom_ribbon(data = df_ci, aes(x = fpr, ymin = tpr_lower, ymax = tpr_upper),
#               fill = "gray", alpha = 0.3) +
#   # (b) The original ROC curve (from `my_roc`)
#   geom_line(data = df_roc, aes(x = fpr, y = tpr), 
#             size = 0.8, color = "blue") +
#   # (c) If highlight row exists, mark it in red
#     # geom_point(data = highlight_row,
#     #            aes(x = fpr, y = tpr_lower),
#     #            color = "red", size = 3) +
#     geom_vline(data=highlight_row, aes(xintercept = fpr,
#                                        color = "red",
#                                        linetype = "dashed"
#                                          )) +
# 
#   # (d) Reference diagonal
#   geom_abline(intercept = 0, slope = 1, linetype = "dashed", color = "gray60") +
#   # (e) Axis range + labels
#   coord_cartesian(xlim = c(0, 1), ylim = c(0, 1)) +
#   theme_bw() + 
#   theme(
#     strip.background = element_blank(),  # remove gray facet strip background
#     legend.position = "none",
#     plot.title = element_text(hjust = 0.5)  # Center title
#   # ) +
#   labs(
#     title = "gpt-4o-2024-08-06: Background Knowledge",
#     x = "False Positive Rate",
#     y = "True Positive Rate"
#   ) 
# 

# 
# df_labels <- df_all %>%
#   group_by(model, prompt) %>%
#   slice_max(threshold) %>%  # pick the row with the largest threshold
#   ungroup() %>%
#   mutate(
#     curve_label = paste0(prompt, " (AUC=", round(auc, 2), ")")
#   )
# 
# ggplot(df_all, aes(x = fpr, y = tpr,
#                    color = prompt,   # same color per prompt across subplots
#                    group = interaction(model, prompt))) +
#   geom_line(size = 1.2, alpha = 0.8) +                # Draw the ROC lines
#   geom_abline(intercept = 0, slope = 1, linetype = "dashed", color = "gray") + 
#   facet_wrap(~ model) +                               # One subplot per model
#   # Plot the AUC label on the line itself:
#   geom_text(
#     data = df_labels,
#     aes(label = curve_label),
#     size = 3,                # text size
#     hjust = 0,               # left-justify
#     vjust = 1,               # shift text slightly above the point
#     check_overlap = TRUE      # prevent text collisions
#   ) +
#   labs(
#     x = "False Positive Rate",
#     y = "True Positive Rate",
#     color = "Prompt"
#   ) +
#   theme_bw() +
#   theme(
#     legend.position = "bottom", 
#     panel.grid.minor = element_blank(),
#     panel.grid.major = element_line(color = "gray90")
#   )

# Create the ROC plot
# ggplot(df_all, aes(x = fpr, y = tpr, 
#                    color = prompt,  # Color by cleaned prompt name (so 3 colors per subplot)
#                    group = prompt)) +
#   geom_line(size = 1.2, alpha = 0.8) +  # Draw the ROC curves
#   geom_abline(intercept = 0, slope = 1, linetype = "dashed", color = "gray") +  # Random guess line
#   facet_wrap(~ model) +  # One subplot per model
#   labs(x = "False Positive Rate",
#        y = "True Positive Rate",
#        color = "Prompt Type") +  # Legend title
#   theme_bw() +
#   theme(
#     legend.position = "bottom",  # Places legend inside plot
#     legend.box = "vertical",  # Stack legend items vertically
#     legend.background = element_rect(fill = alpha("white", 0.8)),  # Slightly transparent background
#     panel.grid.minor = element_blank(),  # Remove minor grid lines
#     panel.grid.major = element_line(color = "gray90")
#   ) +
#   guides(color = guide_legend(ncol = 1))  # Force a vertical legend inside the plot

