import pandas as pd
from plotnine import (
    ggplot, aes, geom_line, geom_point, facet_grid, stat_summary,
    theme, element_text, theme_bw, geom_boxplot, geom_bar, ylim,
    geom_tile, geom_text, geom_vline, coord_cartesian, coord_trans,
    theme_set, geom_area, facet_wrap
)
from plotnine.scales import scale_y_log10, scale_x_log10, scale_x_discrete, scale_y_reverse, scale_fill_cmap, scale_y_continuous
import os
import torch
import numpy as np
from tqdm import tqdm
import math
import json
from analyse import compute_all_fits, power_law_mapping, bayesian_law_mapping
from scipy import stats
import pickle
from scipy import stats
from copy import deepcopy
import glob

def format_data(data):

    # rename columns
    data = data.rename(columns={
        "shots": "Shots",
        "k": "Ex. length",
        "hmm": "HMM",
        "sft": "SFT",
        "sft_amount": "# SFT examples",
        "dpo_amount": "# DPO examples",
        "prob": "Probability",
        "acc": "Accuracy",
        "nll": "NLL",
        "layers": "Model",
        "dataset": "Dataset",
        "tokens": "Tokens",
    })

    # law names
    if "law" in data.columns:
        data = data.rename(columns={
            "nrmse": "NRMSE (NLL)",
            "nrmse_prob": "NRMSE",
            "rmse": "RMSE (NLL)",
            "rmse_prob": "RMSE",
            "law": "Law",
            "priors": "Priors",
            "gammas": "In-distrib. probs.",
            "betas": "OOD probs.",
            "K": "ICL efficiency"
        })
        data['Law'] = data['Law'].map({
            "bayesian_original": "Bayesian (O.)",
            "bayesian_sampling": "Bayesian (Sa.)",
            "bayesian_scoring": "Bayesian (Sc.)",
            "bayesian_old": "Bayesian (Old)",
            "power": "Power",
            "bounded": "Bounded",
            "logistic": "Logistic",
        })
        data["Law type"] = data["Law"].apply(lambda x: "Bayesian" if x.startswith("Bayesian") else "Non-Bayesian")

    # relabel some of the values
    labellers = {
        "Model": lambda x: f"{x}-layer GPT",
        "HMM": lambda x: f"HMM {x}"
    }
    
    for column in labellers:
        if column not in data.columns: continue
        model_labeller = labellers[column]
        model_order = [model_labeller(x) for x in data[column].unique()]
        data[column] = data[column].map(model_labeller)
        data[column] = pd.Categorical(data[column], categories=model_order, ordered=True)
    
    # rename columns
    data = data.rename(columns={"model": "Model"})

    return data

# LLMs
llm_dfs = []
for file in glob.glob("logs/real-lms/*.csv"):
    if "Llama-3.1-8B" not in file: continue
    model, dataset, _ = file.split('___')
    llm_df = pd.read_csv(file)
    llm_df["model"] = model
    llm_df["dataset"] = dataset
    llm_dfs.append(llm_df)

llm_df = pd.concat(llm_dfs)
llm_df["model"] = llm_df["model"].apply(lambda x: x.replace("logs/real-lms/", ""))
llm_df["model"] = llm_df["model"].map({
    "google_gemma-2b-it": "Gemma 2B",
    "google_gemma-7b-it": "Gemma 7B",
    "google_gemma-1.1-2b-it": "Gemma 1.1 2B",
    "meta-llama_Meta-Llama-3.1-405B-Instruct-Turbo": "Llama 3.1 405B",
    "meta-llama_Llama-3.2-3B-Instruct": "Llama 3.2 3B",
    "meta-llama_Llama-3.2-1B-Instruct": "Llama 3.2 1B",
    "meta-llama_Llama-3.1-8B-Instruct": "Llama 3.1 8B",
    "meta-llama_Llama-3.1-8B": "Llama 3.1 8B (base)",
})
llm_df["dataset"] = llm_df["dataset"].apply(lambda x: x.replace("evals_", ""))

# only collect first 90% of shots
filter = (llm_df["model"] == "none")
for model in llm_df["model"].unique():
    for dataset in llm_df["dataset"].unique():
        subset = llm_df[(llm_df["dataset"] == dataset) & (llm_df["model"] == model)]
        max_shots = subset["shots"].max()
        keep = 0.9 * max_shots
        filter |= ((llm_df["dataset"] == dataset) & (llm_df["model"] == model) & (llm_df["shots"] <= keep))

# filter
print(len(llm_df))
llm_df = llm_df[filter]
print(len(llm_df))

all_params_llm = []
all_models_llm = {}
for model in llm_df["model"].unique():
    print("model:", model)
    for dataset in tqdm(llm_df["dataset"].unique()):
        subset = llm_df[(llm_df["model"] == model) & (llm_df["dataset"] == dataset)]
        num_hmms = len(subset["hmm"].unique()) + 1
        params, models = compute_all_fits(
            subset=subset, max_shots=1.0, quiet=True, patience=100, epochs=100,
            lr=5e-2, num_hmms=num_hmms, i=0, metadata={},
            mode="lbfgs", loss_mode="mse_prob"
        )
        # compute_all_fits(subset, num_hmms=num_hmms, epochs=1000, lr=5e-2, patience=200, quiet=True)
        temp_df = pd.DataFrame(params)
        temp_df["model"] = model
        temp_df["dataset"] = dataset
        all_params_llm.append(temp_df)
        all_models_llm[(model, dataset)] = models

params_llm_df = pd.concat([x for x in all_params_llm if "base" not in x["model"].unique()[0]])

# make latex table for appendix
average_nrmse = params_llm_df.copy()
average_nrmse = average_nrmse.groupby(['model', 'dataset', 'law'])['nrmse_prob'].mean().unstack()

# bold min value in each row
def bold_min(row):
    min_val = row.min()
    # Apply bold formatting to the minimum value
    return row.apply(lambda x: f'\\textbf{{{x:.4f}}}' if x == min_val else f'{x:.4f}')

# table fmt
average_nrmse = average_nrmse.apply(bold_min, axis=1)
latex_table = average_nrmse.to_latex(escape=False)
latex_table = latex_table.replace('_', '\\_').replace('\\cline{1-8}', '\\midrule')
print(latex_table)

average_nrmse = params_llm_df.copy()
average_nrmse = average_nrmse.groupby(['model', 'dataset', 'law'])['nrmse_prob'].mean().reset_index().groupby(['model', 'law'])['nrmse_prob'].mean().unstack()

# table fmt
average_nrmse = average_nrmse.apply(bold_min, axis=1)
latex_table = average_nrmse.to_latex(escape=False)
latex_table = latex_table.replace('_', '\\_').replace('\\cline{1-8}', '\\midrule')
print(latex_table)

average_nrmse = params_llm_df.copy()
average_nrmse = average_nrmse.groupby(['model', 'dataset', 'law'])['nrmse_prob'].mean().reset_index().groupby(['model', 'law'])['nrmse_prob'].mean().reset_index().groupby(['law'])['nrmse_prob'].mean()

# table fmt
# average_nrmse = average_nrmse.apply(bold_min, axis=1)
latex_table = average_nrmse.to_latex(escape=True)
latex_table = latex_table.replace('\\cline{1-8}', '\\midrule')
print(latex_table)

extrap = params_llm_df.groupby(['dataset', 'model', 'hmm', 'law'])['nrmse_prob'].mean().unstack().reset_index()
extrap

print("Interpolation test")
for model in extrap["model"].unique():
    print(model)
    for law in params_llm_df["law"].unique():
        for law2 in params_llm_df["law"].unique():
            if law == law2:
                continue
            # print(law, law2)
            ttest = stats.ttest_rel(
                extrap[extrap["model"] == model][law],
                extrap[extrap["model"] == model][law2]
            )
            print(f"{'✅' if ttest.pvalue < 0.05 else '❌'} {'>' if ttest.statistic > 0 else '<'} {law:>20} vs {law2:<20}: {ttest.statistic:>8.4f}, {ttest.pvalue:>8.4f}, {ttest.df:>8}")

extrap = params_llm_df.groupby(['dataset', 'model', 'hmm', 'law'])['nrmse_prob'].mean().unstack().reset_index()
extrap

print("Interpolation test")
for law in params_llm_df["law"].unique():
    for law2 in params_llm_df["law"].unique():
        if law == law2:
            continue
        # print(law, law2)
        ttest = stats.ttest_rel(
            extrap[law],
            extrap[law2]
        )
        print(f"{'✅' if ttest.pvalue < 0.05 else '❌'} {'>' if ttest.statistic > 0 else '<'} {law:>20} vs {law2:<20}: {ttest.statistic:>8.4f}, {ttest.pvalue:>8.4f}, {ttest.df:>8}")

params_llm_df_comp = pd.concat([x for x in all_params_llm if "Llama 3.1 8" in x["model"].unique()[0]])

params_llm_df_comp["model"].unique()

temp = format_data(llm_df[llm_df["model"].str.contains("Llama 3.1 8")])
temp["Model"] = temp["Model"].apply(lambda x: "Base" if "base" in x else "Instruct")
temp["Dataset"] = temp["Dataset"].apply(lambda x: x.split("_")[-1])
temp = temp[temp["HMM"] == "HMM 0"]
print(temp)
plot = (
    ggplot(temp, aes(x="Shots", y="Probability", color="Model", group="Model")) + geom_line() +
    facet_wrap("Dataset") + scale_x_log10() + theme(figure_size=(5, 3))
)
plot.save("paper/base_v_instruct.pdf", width=5, height=3)

base_v_instruct_comp = []
posteriors_comp = []
for model in llm_df["model"].unique():
    if "Llama 3.1 8B" not in model: continue
    for dataset in llm_df["dataset"].unique():
        law = all_models_llm[(model, dataset)]['scoring']
        subset = llm_df[(llm_df["dataset"] == dataset) & (llm_df["model"] == model)]
        max_shots = subset["shots"].max()
        est = law.estimate_nll(max_shots, 0, add_metrics=True)
        posteriors = est["posteriors"]
        prob = list(map(lambda x: math.exp(-x), est["nll"]))
        for hmm in range(1):
            for i in range(1, len(posteriors)):
                posteriors_comp.append({
                    "Shots": i,
                    "HMM": f"HMM {hmm}",
                    "Posterior": posteriors[i, hmm].item(),
                    "Model": "Base" if "base" in model else "Instruct",
                    "Dataset": dataset.split("_")[-1],
                })
                base_v_instruct_comp.append({
                    "Shots": i,
                    "Probability": subset[subset["shots"] == i]["prob"].mean().item(),
                    "Est. Probability": prob[i],
                    "Model": "Base" if "base" in model else "Instruct",
                    "Dataset": dataset.split("_")[-1],
                })

plot = (
    ggplot(pd.DataFrame(posteriors_comp), aes(x="Shots", y="Posterior", color="Model")) + facet_wrap("Dataset") +
    geom_line() + scale_x_log10() + theme(figure_size=(5, 3))
)
plot.save("paper/base_v_instruct_posterior.pdf", width=5, height=3)

plot = (
    ggplot(pd.DataFrame(base_v_instruct_comp), aes(x="Shots", y="Probability", color="Model")) + facet_wrap("Dataset") +
    geom_line(alpha=0.5) + scale_x_log10() + theme(figure_size=(5, 3)) +
    geom_line(aes(y="Est. Probability", color="Model"), linetype="dashed", alpha=1.0)
)
plot.save("paper/base_v_instruct_prob.pdf", width=5, height=3)