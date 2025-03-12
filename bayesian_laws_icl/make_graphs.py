import pandas as pd
from plotnine import (
    ggplot, aes, geom_line, geom_point, facet_grid, stat_summary,
    theme, element_text, theme_bw, geom_boxplot, geom_bar, ylim,
    geom_tile, geom_text, geom_vline, coord_cartesian, coord_trans,
    theme_set, geom_area, facet_wrap
)
from plotnine.scales import scale_y_log10, scale_x_log10, scale_x_discrete, scale_y_reverse, scale_fill_cmap, scale_y_continuous, scale_color_cmap
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

DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"

# make plots pretty
theme_set(
    theme_bw() +
    theme(
        axis_text_x=element_text(rotation=90),
        text=element_text(family="Inter"),
        strip_text=element_text(size=8)
    )
)

def make_df(perc, data):
    res = pd.DataFrame(data)
    res["perc"] = perc
    return res

# load data
for layers in [1, 2, 3, 4, 8, 12, 16]:
    doc = f"logs/{layers}-1,1,1,1,1-1,0,0,0,0/trainer_state.json"
    if not os.path.exists(doc):
        continue
    with open(doc, "r") as f:
        data = json.load(f)
    train_loss = [x for x in data["log_history"] if "loss" in x][-1]["loss"]
    eval_loss = [x for x in data["log_history"] if "eval_loss" in x][-1]["eval_loss"]
    print(f"\\texttt{{gpt}}$_{layers}$ & ${train_loss:.3f}$ & ${eval_loss:.3f}$ \\\\")

datas = []
for layers in [1, 2, 3, 4, 8, 12, 16]:
    doc = f"logs/{layers}-1,1,1,1,1-1,0,0,0,0/in_context_probs.csv"
    if not os.path.exists(doc):
        continue
    data = pd.read_csv(doc)
    data['shots'] = data['shots'] + 1
    data['layers'] = layers
    datas.append(data)
data = pd.concat(datas)

# keep only final checkpoints of finetuning
filter = (data['sft_amount'] == -1)
for sft_amount in data['sft_amount'].unique():
    for layers in data['layers'].unique():
        maxi = data[(data['layers'] == layers) & (data['sft_amount'] == sft_amount)]['sft'].max()
        filter |= ((data['sft'] == maxi) & (data['layers'] == layers) & (data['sft_amount'] == sft_amount)) 
data = data[filter]

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

pretrain_data = data[data["sft_amount"] == 0].copy()
pretrain_data["tokens"] = pretrain_data["shots"] * (pretrain_data["k"] + 1)

format_data(pretrain_data)

# ICL plots (no fits)

# plot = (
#     ggplot(format_data(pretrain_data), aes(x="Tokens", y="Accuracy", color="Ex. length", group="Ex. length")) +
#     stat_summary(geom="line") + facet_grid("~Model") + theme(figure_size=(7, 2.5))
# )
# plot.save('paper/icl_curve.pdf', width=7, height=2.5)

# plot = (
#     ggplot(format_data(pretrain_data), aes(x="Tokens", y="Accuracy", color="Ex. length", group="Ex. length")) +
#     geom_line() + facet_grid("Model~HMM") + theme(figure_size=(7, 7))
# )
# plot.save('paper/icl_curve_hmm.pdf', width=7, height=7)

# plot shots vs. prob, faceted by layers, and only for pretrain

plot = (
    ggplot(format_data(pretrain_data[pretrain_data["layers"] >= 3]), aes(x="Shots", y="Probability", color="Model")) +
    facet_grid("Ex. length~HMM", labeller="label_both") +
    geom_line() + scale_x_log10() 
)
plot.save('paper/shots_v_prob.pdf', width=8, height=8)

sft_data = deepcopy(data)
sft_data["tokens"] = sft_data["shots"] * (sft_data["k"] + 1)
# sft_data['k'] = sft_data['k'].astype(str)

sft_df = pd.DataFrame(sft_data)
sft_df = sft_df.groupby(["shots", "k", "hmm", "sft", "sft_amount", "layers", "tokens"]).mean().reset_index()
sft_df["nll_avg"] = sft_df["nll"]
sft_df["nll"] = sft_df["prob"].map(lambda x: -math.log(x))
sft_df = sft_df[sft_df["layers"] >= 3]

for K in sft_df["k"].unique():
    temp = sft_df[sft_df["k"] == K]
    plot = (
        ggplot(format_data(temp), aes(x="Shots", y="Probability", color="# SFT examples", group="# SFT examples")) +
        facet_grid("Model~HMM", labeller="label_both") +
        geom_line() + scale_x_log10() +
        scale_color_cmap(trans="log10")
    )
    plot.save(f'paper/shots_v_prob_sft_{K}.pdf', width=8, height=8)

dpo_datas = []
for layers in [1, 2, 3, 4, 8, 12, 16]:
    file = f"logs/{layers}-1,1,1,1,1-1,0,0,0,0-dpo/in_context_probs.csv"
    if not os.path.exists(file): continue
    dpo_data = pd.read_csv(file)
    dpo_data['shots'] = dpo_data['shots'] + 1
    dpo_data['layers'] = layers
    dpo_datas.append(dpo_data)

if len(dpo_datas) > 0:
    dpo_df = pd.concat(dpo_datas)

    filter = (dpo_df['sft_amount'] == -1)
    for layers in dpo_df['layers'].unique():
        for sft_amount in dpo_df['sft_amount'].unique():
            maxi = dpo_df[(dpo_df['layers'] == layers) & (dpo_df['sft_amount'] == sft_amount)]['sft'].max()
            filter |= ((dpo_df['sft'] == maxi) & (dpo_df['layers'] == layers) & (dpo_df['sft_amount'] == sft_amount)) 
    dpo_df = dpo_df[filter]
    # add pretrain data
    dpo_df = pd.concat([pretrain_data, dpo_df])
    dpo_df['dpo_amount'] = dpo_df['sft_amount']
    dpo_df = dpo_df[dpo_df["layers"] >= 3]

    for K in dpo_df["k"].unique():
        temp = dpo_df[dpo_df["k"] == K]
        plot = (
            ggplot(format_data(temp), aes(x="Shots", y="Probability", color="# DPO examples", group="# DPO examples")) +
            facet_grid("Model~HMM", labeller="label_both") +
            geom_line() + scale_x_log10() +
            scale_color_cmap(trans="log10")
        )
        plot.save(f'paper/shots_v_prob_dpo_{K}.pdf', width=8, height=8)

# Bayesian law fits

all_params = {}
all_models = {}

if os.path.exists("paper/ginc_models"):
    with open("paper/ginc_models", "rb") as f:
        all_models = pickle.load(f)

if os.path.exists("paper/ginc_params"):
    with open("paper/ginc_params", "rb") as f:
        all_params = pickle.load(f)

pretrain_data_mean = pretrain_data.groupby(["shots", "k", "hmm", "sft", "sft_amount", "layers"]).mean().dropna().reset_index()
pretrain_data_mean["nll_avg"] = pretrain_data_mean["nll"]
pretrain_data_mean["nll"] = pretrain_data_mean["prob"].map(lambda x: -math.log(x))
pretrain_data_mean

## Extrapolation
# 
# With 5%, 10%, 20%, and 50% of the data.

for perc in [0.05, 0.1, 0.2, 0.5]:
    all_params[perc] = []
    all_models[perc] = {}
    
    for i in range(1):
        for k in pretrain_data_mean["k"].unique():
            for layers in tqdm(pretrain_data_mean["layers"].unique()):
                if layers <= 2: continue
                subset = pretrain_data_mean[(pretrain_data_mean["k"] == k) & (pretrain_data_mean["layers"] == layers)]
                some_params, some_models = compute_all_fits(
                    subset=subset, max_shots=perc, quiet=True, patience=100, epochs=100,
                    lr=5e-2, num_hmms=5, i=i, metadata={"layers": layers, "k": k},
                    mode="lbfgs", loss_mode="mse_prob"
                )
                all_params[perc].extend(some_params)
                for key in some_models:
                    if isinstance(key, tuple): # (law, hmm)
                        all_models[perc][(key[0], k, layers, key[1])] = some_models[key]
                    else: # just law
                        all_models[perc][(key, k, layers)] = some_models[key]

    df = pd.DataFrame(all_params[perc])
    print(perc)
    print(df.groupby(["layers", "law"])["nrmse_prob"].mean().unstack().to_latex(float_format="{:.4f}".format))
    nans_by_law = df["nrmse_prob"].isna().groupby(df["law"]).sum()
    print(nans_by_law)
    # print("\n")
    # print(df.groupby(["layers", "law"])["log_nrmse"].mean().unstack().to_latex(float_format="{:.3f}".format))
    print("\n\n")

df = pd.concat(map(lambda x: make_df(x, all_params[x]), all_params.keys()))
df

# make latex table for appendix
average_nrmse = df[df["perc"] < 1.0].copy()
average_nrmse["perc"] = average_nrmse["perc"].apply(lambda x: f"${x:.0%}$".replace("%", "\\%"))
average_nrmse = average_nrmse.groupby(['perc', 'layers', 'law'])['nrmse_prob'].mean().unstack()

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

extrap = df[(df["perc"] < 1)].groupby(['perc', 'law'])['nrmse_prob'].mean().reset_index()
# extrap["perc"] = extrap["perc"].map(lambda x: f"{x:.2f}")
extrap["law_type"] = extrap["law"].map(lambda x: "bayesian" if (x.split("_")[0] == "bayesian") else "non-bayesian")
print(extrap)

extrap = df[(df["layers"] > 2) & (df["perc"] == 0.1)].groupby(['perc', 'hmm', 'layers', 'k', 'law'])['nrmse_prob'].mean().unstack().reset_index()
extrap

print("Extrapolation test")
for law in df["law"].unique():
    for law2 in df["law"].unique():
        if law == law2:
            continue
        # print(law, law2)
        ttest = stats.ttest_rel(
            extrap[law],
            extrap[law2]
        )
        print(f"{'✅' if ttest.pvalue < 0.05 else '❌'} {'>' if ttest.statistic > 0 else '<'} {law:>20} vs {law2:<20}: {ttest.statistic:>8.4f}, {ttest.pvalue:>8.4f}")


## Interpolation
# 
# All of the data!

perc = 1.0
all_params[perc] = []
for i in range(1):
    all_models[(perc, i)] = {}
    for k in pretrain_data_mean["k"].unique():
        for layers in tqdm(pretrain_data_mean["layers"].unique()):
            if layers <= 2: continue
            subset = pretrain_data_mean[(pretrain_data_mean["k"] == k) & (pretrain_data_mean["layers"] == layers)]
            some_params, some_models = compute_all_fits(
                subset=subset, max_shots=perc, quiet=True, patience=100, epochs=100,
                lr=5e-2, num_hmms=5, i=i, metadata={"layers": layers, "k": k},
                mode="lbfgs", loss_mode="mse_prob"
            )
            for d in some_params:
                d["layers"] = layers
                d["k"] = k
            all_params[perc].extend(some_params)
            for key in some_models:
                if isinstance(key, tuple): # (law, hmm)
                    all_models[(perc, i)][(key[0], k, layers, key[1])] = some_models[key]
                else: # just law
                    all_models[(perc, i)][(key, k, layers)] = some_models[key]

# save all params
with open("paper/ginc_params", "wb") as f:
    pickle.dump(all_params, f)

# save all models
with open("paper/ginc_models", "wb") as f:
    pickle.dump(all_models, f)

df = pd.DataFrame(all_params[perc])
print(f"Dropped {len(df[df['nrmse'].isna() == True])} of {len(df)} observation(s)")
df = df[df["nrmse"].isna() == False]
print(perc)
print(df.groupby(["layers", "law"])["nrmse"].mean().unstack().to_latex(float_format="{:.4f}".format))
nans_by_law = df["nrmse"].isna().groupby(df["law"]).sum()
print(nans_by_law)

df = pd.concat(map(lambda x: make_df(x, all_params[x]), all_params.keys()))
interp = df[(df["layers"] > 2) & (df["perc"] == 1.0)].groupby(['perc', 'hmm', 'layers', 'k', 'law'])['nrmse_prob'].mean().unstack().reset_index()
interp

print("Interpolation test")
for law in df["law"].unique():
    for law2 in df["law"].unique():
        if law == law2:
            continue
        # print(law, law2)
        ttest = stats.ttest_rel(
            interp[law],
            interp[law2]
        )
        print(f"{'✅' if ttest.pvalue < 0.05 else '❌'} {'>' if ttest.statistic > 0 else '<'} {law:>20} vs {law2:<20}: {ttest.statistic:>8.4f}, {ttest.pvalue:>8.4f}")


## Plots

perc = 1.0
params_df = pd.DataFrame(all_params[perc])
params_df.to_csv('paper/params.csv', index=False)
params_df

plot = (
    ggplot(format_data(params_df), aes(x="Model", y="NRMSE", group="Law", color="Law", fill="Law")) +
    stat_summary(aes(linetype="Law type"), geom="line") +
    stat_summary(geom="point") +
    # facet_grid("~Ex. length", labeller="label_both") +
    theme(axis_text_x=element_text(rotation=45, ha="right"))
)
plot.save('paper/law_comparison.pdf', width=5, height=3)

print(params_df.groupby("law")["nrmse_prob"].mean().to_latex(float_format="{:.4f}".format))
print(params_df.groupby("law")["nrmse_prob"].std().to_latex(float_format="{:.4f}".format))

params_df_50 = pd.DataFrame(all_params[0.1])
print(params_df_50.groupby("law")["nrmse"].mean().to_latex(float_format="{:.4f}".format))
print(params_df_50.groupby("law")["nrmse"].std().to_latex(float_format="{:.4f}".format))

laws_list = sorted(list(params_df["law"].unique()))
print(laws_list)
for layers in params_df["layers"].unique():
    print(layers, end='')
    for k in params_df["k"].unique():
        print(f" & {k}", end='')
        vals = []
        for law in laws_list:
            nrmse = params_df[(params_df["layers"] == layers) & (params_df["k"] == k) & (params_df["law"] == law)]["nrmse"].mean()
            vals.append(nrmse)
        mini = min(vals)
        for val in vals:
            print(f" & {val:.4f}" if val != mini else f" & \\textbf{{{val:.4f}}}", end='')
        print(' \\\\')
    print('\\midrule')
        

bayesian_df = format_data(params_df)
bayesian_df = bayesian_df[bayesian_df["Law"] == "Bayesian (Sc.)"]
bayesian_df

plot = (
    ggplot(bayesian_df, aes(x="Model", y="Priors", group="HMM", color="HMM", fill="HMM")) +
    # stat_summary(geom="line") + stat_summary()
    stat_summary(geom="bar", position="stack") + facet_grid("Law~Ex. length", labeller="label_both") +
    scale_y_continuous(limits=(0, 1.0001), expand=(0, 0)) + theme(axis_text_x=element_text(rotation=45, ha="right"))
)
plot.save('paper/priors.pdf', width=5, height=3)

plot = (
    ggplot(bayesian_df, aes(x="Model", y="Priors", group="HMM", color="HMM", fill="HMM")) +
    stat_summary(geom="line") + stat_summary() + facet_grid("Law~Ex. length")
    + ylim(0, 1)
)
plot.save('paper/priors2.pdf', width=5, height=2.5)

plot = (
    ggplot(bayesian_df, aes(x="Model", y="In-distrib. probs.", group="HMM", color="HMM", fill="HMM")) +
    stat_summary(geom="line") + stat_summary() + facet_grid("Law~Ex. length", labeller="label_both")
    + ylim(0, 1)
)
plot.save('paper/gammas.pdf', width=6, height=4)

plot = (
    ggplot(bayesian_df, aes(x="Model", y="OOD probs.", group="HMM", color="HMM", fill="HMM")) +
    stat_summary(geom="line") + stat_summary() + facet_grid("Law~Ex. length", labeller="label_both")
    + ylim(0, 1)
)
plot.save('paper/betas.pdf', width=6, height=4)

layers = 16
k = 10
subset = format_data(params_df[(params_df["layers"] == layers) & (params_df["k"] == k)])

entries = []
for sampling_hmm in range(5):
    for scoring_hmm in range(5):
        # BO
        BO = subset[(subset["Law"] == "Bayesian (O.)") & (subset["HMM"] == f"HMM {sampling_hmm}")][f"P"]
        val = (BO.apply(lambda x: torch.tensor(x)).sum() / len(BO)).tolist()[scoring_hmm]
        entries.append({"Law": "Bayesian (O.)", "Sampling": sampling_hmm, "Scoring": scoring_hmm, "val": val})

        if sampling_hmm == scoring_hmm:
            val = subset[(subset["Law"] == "Bayesian (Sc.)") & (subset["HMM"] == f"HMM {sampling_hmm}")][f"In-distrib. probs."].mean()
            entries.append({"Law": "Bayesian (Sc.)", "Sampling": sampling_hmm, "Scoring": scoring_hmm, "val": val})
            val = subset[(subset["Law"] == "Bayesian (Sa.)") & (subset["HMM"] == f"HMM {sampling_hmm}")][f"In-distrib. probs."].mean()
            entries.append({"Law": "Bayesian (Sa.)", "Sampling": sampling_hmm, "Scoring": scoring_hmm, "val": val})
        else:
            val = subset[(subset["Law"] == "Bayesian (Sc.)") & (subset["HMM"] == f"HMM {scoring_hmm}")][f"OOD probs."].mean()
            entries.append({"Law": "Bayesian (Sc.)", "Sampling": sampling_hmm, "Scoring": scoring_hmm, "val": val})
            val = subset[(subset["Law"] == "Bayesian (Sa.)") & (subset["HMM"] == f"HMM {scoring_hmm}")][f"OOD probs."].mean()
            entries.append({"Law": "Bayesian (Sa.)", "Sampling": sampling_hmm, "Scoring": scoring_hmm, "val": val})

entries_df = pd.DataFrame(entries)
plot = (
    ggplot(entries_df, aes(x="Scoring", y="Sampling", fill="val", label="val")) + geom_tile() +
    scale_y_reverse() + facet_grid("~Law") + geom_text(format_string="{:.2f}") + theme_bw() +
    theme(figure_size=(10, 4)) + scale_fill_cmap(cmap_name="magma", limits=(0, entries_df["val"].max()))
)
plot.save('paper/P_pretrained_16_5.pdf', width=10, height=4)

bayesian_df["ICL efficiency"] = bayesian_df["ICL efficiency"].map(lambda x: x.item() if isinstance(x, torch.Tensor) else x)
plot = (
    ggplot(bayesian_df, aes(x="Model", y="ICL efficiency", group="Ex. length", color="Ex. length")) +
    stat_summary(geom="line") + stat_summary() + facet_grid("Law~", labeller="label_both") + theme(axis_text_x=element_text(rotation=45, ha="right"))
)
plot.save('paper/K.pdf', width=5, height=3)

# SFT

sft_data = deepcopy(data)
sft_data["tokens"] = sft_data["shots"] * (sft_data["k"] + 1)
# sft_data['k'] = sft_data['k'].astype(str)

sft_df = pd.DataFrame(sft_data)
sft_df = sft_df.groupby(["shots", "k", "hmm", "sft", "sft_amount", "layers", "tokens"]).mean().reset_index()
sft_df["nll_avg"] = sft_df["nll"]
sft_df["nll"] = sft_df["prob"].map(lambda x: -math.log(x))
print(sft_df["sft_amount"].unique())
sft_df

all_params_sft = []
all_models_sft = {}
for sft_amount in sft_df["sft_amount"].unique():
    all_models_sft[sft_amount] = {}
    for k in sft_df["k"].unique():
        print(sft_amount, k)
        for layers in tqdm(sft_df["layers"].unique()):
            if layers <= 2: continue
            subset = sft_df[(sft_df["sft_amount"] == sft_amount) & (sft_df["k"] == k) & (sft_df["layers"] == layers)]
            some_params, some_models = compute_all_fits(
                subset=subset, max_shots=1.0, quiet=True, patience=100, epochs=100,
                lr=5e-2, num_hmms=5, i=0, metadata={"layers": layers, "k": k, "sft_amount": sft_amount},
                mode="lbfgs", loss_mode="mse_prob"
            )
            all_params_sft.extend(some_params)
            for key in some_models:
                if isinstance(key, tuple):
                    all_models_sft[sft_amount][(key[0], k, layers, key[1])] = some_models[key]
                else:
                    all_models_sft[sft_amount][(key, k, layers)] = some_models[key]

# save all params
with open("paper/ginc_sft_params", "wb") as f:
    pickle.dump(all_params_sft, f)

# save all models
with open("paper/ginc_sft_models", "wb") as f:
    pickle.dump(all_models_sft, f)

params_sft_df = pd.DataFrame(all_params_sft)
params_sft_df.to_csv('paper/params_sft.csv', index=False)

nans_by_law = params_sft_df["nrmse"].isna().groupby(params_sft_df["law"]).sum()
print(nans_by_law)

params_sft_df.loc[params_sft_df['sft_amount'] == 0, 'sft_amount'] += 1

len(params_sft_df)

print(params_sft_df[(params_sft_df["sft_amount"] > 1)].groupby("law")["nrmse"].mean().to_latex(float_format="{:.4f}".format))
print(params_sft_df[(params_sft_df["sft_amount"] > 1)].groupby("law")["nrmse"].std().to_latex(float_format="{:.4f}".format))

interp = pd.DataFrame(all_params_sft).groupby(['sft_amount', 'hmm', 'layers', 'k', 'law'])['nrmse_prob'].mean().unstack().reset_index()
interp = interp[(interp["sft_amount"] > 1)]

print("Interpolation test")
for law in pd.DataFrame(all_params_sft)["law"].unique():
    for law2 in pd.DataFrame(all_params_sft)["law"].unique():
        if law == law2:
            continue
        # print(law, law2)
        ttest = stats.ttest_rel(
            interp[law],
            interp[law2]
        )
        print(f"{'✅' if ttest.pvalue < 0.05 else '❌'} {'>' if ttest.statistic > 0 else '<'} {law:>20} vs {law2:<20}: {ttest.statistic:>8.4f}, {ttest.pvalue:>8.4f}")


plot = (
    ggplot(format_data(params_sft_df), aes(x="# SFT examples", y="NRMSE", group="Ex. length", color="Ex. length")) +
    stat_summary(geom="line") + stat_summary() + facet_grid("Law~Model") +
    scale_x_log10() + scale_y_log10()
)
plot.save('paper/law_comparison_sft.pdf', width=7, height=2.5)

plot = (
    ggplot(format_data(params_sft_df[(params_sft_df["law"] == "bayesian_scoring") & (params_sft_df["k"] == 10)]), aes(x="# SFT examples", y="Priors", group="HMM", color="HMM", fill="HMM")) +
    # stat_summary(geom="line") + stat_summary()
    stat_summary(geom="area", position="stack") + facet_grid("~Model") +
    scale_x_log10() + scale_y_continuous(ylim=(0, 1.0001), expand=(0, 0))
)
plot.save('paper/priors_sft.pdf', width=5, height=3)

temp = format_data(params_sft_df[(params_sft_df["law"] == "bayesian_scoring") & (params_sft_df["k"] == 10)])
temp["HMM"] = temp["HMM"].apply(lambda x: "Favoured" if x == "HMM 0" else "Disfavoured")
plot = (
    ggplot(temp, aes(x="# SFT examples", y="In-distrib. probs.", group="Model", color="Model", fill="Model")) +
    # stat_summary(geom="line") + stat_summary()
    stat_summary(geom="line") + stat_summary(geom="point") + facet_grid("~HMM", labeller="label_context") +
    theme_bw() + theme(axis_text_x=element_text(rotation=90)) + scale_x_log10()
)
plot.save('paper/gammas_sft.pdf', width=5, height=3)

plot = (
    ggplot(temp, aes(x="# SFT examples", y="OOD probs.", group="Model", color="Model", fill="Model")) +
    # stat_summary(geom="line") + stat_summary()
    stat_summary(geom="line") + stat_summary(geom="point") + facet_grid("~HMM", labeller="label_context") +
    theme_bw() + theme(axis_text_x=element_text(rotation=90)) + scale_x_log10()
)
plot.save('paper/betas_sft.pdf', width=5, height=3)

plot = (
    ggplot(temp, aes(x="# SFT examples", y="ICL efficiency", group="Model", color="Model", fill="Model")) +
    # stat_summary(geom="line") + stat_summary()
    stat_summary(geom="line") + stat_summary() + facet_grid("Law~") +
    theme_bw() + theme(axis_text_x=element_text(rotation=90)) + scale_x_log10() + scale_y_log10()
)
plot.save('paper/K_sft.pdf', width=5, height=3)

# make latex table for appendix
average_nrmse = params_sft_df[params_sft_df["sft_amount"] > 1].copy()
average_nrmse = average_nrmse.groupby(['sft_amount', 'layers', 'law'])['nrmse_prob'].mean().unstack()

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

# DPO

dpo_datas = []
for layers in [1, 2, 3, 4, 8, 12, 16]:
    file = f"logs/{layers}-1,1,1,1,1-1,0,0,0,0-dpo/in_context_probs.csv"
    if not os.path.exists(file): continue
    dpo_data = pd.read_csv(file)
    dpo_data['shots'] = dpo_data['shots'] + 1
    dpo_data['layers'] = layers
    dpo_datas.append(dpo_data)
dpo_df = pd.concat(dpo_datas)

filter = (dpo_df['sft_amount'] == -1)
for layers in dpo_df['layers'].unique():
    for sft_amount in dpo_df['sft_amount'].unique():
        maxi = dpo_df[(dpo_df['layers'] == layers) & (dpo_df['sft_amount'] == sft_amount)]['sft'].max()
        filter |= ((dpo_df['sft'] == maxi) & (dpo_df['layers'] == layers) & (dpo_df['sft_amount'] == sft_amount)) 
dpo_df = dpo_df[filter]
dpo_df["tokens"] = dpo_df["shots"] * (dpo_df["k"] + 1)

all_params_dpo = []
all_models_dpo = {}
for sft_amount in dpo_df["sft_amount"].unique():
    all_models_dpo[sft_amount] = {}
    for k in dpo_df["k"].unique():
        print(sft_amount, k)
        for layers in tqdm(dpo_df["layers"].unique()):
            if layers <= 2: continue
            subset = dpo_df[(dpo_df["sft_amount"] == sft_amount) & (dpo_df["k"] == k) & (dpo_df["layers"] == layers)]
            some_params, some_models = compute_all_fits(
                subset=subset, max_shots=1.0, quiet=True, patience=100, epochs=100,
                lr=5e-2, num_hmms=5, i=0, metadata={"layers": layers, "k": k, "sft_amount": sft_amount},
                mode="lbfgs", loss_mode="mse_prob"
            )
            all_params_dpo.extend(some_params)
            for key in some_models:
                if isinstance(key, tuple):
                    all_models_dpo[sft_amount][(key[0], k, layers, key[1])] = some_models[key]
                else:
                    all_models_dpo[sft_amount][(key, k, layers)] = some_models[key]

# save all params
with open("paper/ginc_dpo_params", "wb") as f:
    pickle.dump(all_params_dpo, f)

# save all models
with open("paper/ginc_dpo_models", "wb") as f:
    pickle.dump(all_models_dpo, f)

pretrained_params = [x for x in all_params_sft if x['sft_amount'] == 0]

params_dpo_df = pd.DataFrame(all_params_dpo + pretrained_params)
params_dpo_df.to_csv('paper/params_dpo.csv', index=False)
params_dpo_df.loc[params_dpo_df['sft_amount'] == 0, 'sft_amount'] += 1
params_dpo_df["dpo_amount"] = params_dpo_df["sft_amount"]

nans_by_law = params_dpo_df["nrmse"].isna().groupby(params_dpo_df["law"]).sum()
print(nans_by_law)

print(params_dpo_df[(params_dpo_df["sft_amount"] > 1)].groupby("law")["nrmse"].mean().to_latex(float_format="{:.4f}".format))
print(params_dpo_df[(params_dpo_df["sft_amount"] > 1)].groupby("law")["nrmse"].std().to_latex(float_format="{:.4f}".format))

interp = pd.DataFrame(all_params_dpo).groupby(['sft_amount', 'hmm', 'layers', 'k', 'law'])['nrmse_prob'].mean().unstack().reset_index()
interp

print("Interpolation test")
for law in pd.DataFrame(all_params_dpo)["law"].unique():
    for law2 in pd.DataFrame(all_params_dpo)["law"].unique():
        if law == law2:
            continue
        # print(law, law2)
        ttest = stats.ttest_rel(
            interp[law],
            interp[law2]
        )
        print(f"{'✅' if ttest.pvalue < 0.05 else '❌'} {'>' if ttest.statistic > 0 else '<'} {law:>20} vs {law2:<20}: {ttest.statistic:>8.4f}, {ttest.pvalue:>8.4f}")


plot = (
    ggplot(format_data(params_dpo_df[(params_dpo_df["k"] == 10)]), aes(x="# DPO examples", y="NRMSE", group="Law", color="Law")) +
    stat_summary(aes(linetype="Law type"), geom="line") + stat_summary(geom="point") + facet_grid("~Model") + scale_x_log10() + scale_y_log10() +
    theme(strip_text=element_text(size=6))
)
plot.save('paper/law_comparison_dpo.pdf', width=5, height=3)

plot = (
    ggplot(format_data(params_dpo_df[(params_dpo_df["law"] == "bayesian_scoring") & (params_dpo_df["k"] == 10)]), aes(x="# SFT examples", y="Priors", group="HMM", color="HMM", fill="HMM")) +
    # stat_summary(geom="line") + stat_summary()
    stat_summary(geom="area", position="stack") + facet_grid("~Model") +
    scale_x_log10() + scale_y_continuous(ylim=(0, 1.0001), expand=(0, 0))
)
plot.save('paper/priors_dpo.pdf', width=5, height=3)

temp = format_data(params_dpo_df[(params_dpo_df["law"] == "bayesian_scoring") & (params_dpo_df["k"] == 10)])
temp["HMM"] = temp["HMM"].apply(lambda x: "Favoured" if x == "HMM 0" else "Disfavoured")
plot = (
    ggplot(temp, aes(x="# SFT examples", y="In-distrib. probs.", group="Model", color="Model", fill="Model")) +
    # stat_summary(geom="line") + stat_summary()
    stat_summary(geom="line") + stat_summary(geom="point") + facet_grid("~HMM", labeller="label_context") +
    theme_bw() + theme(axis_text_x=element_text(rotation=90)) + scale_x_log10()
)
plot.save('paper/gammas_dpo.pdf', width=5, height=3)

plot = (
    ggplot(temp, aes(x="# SFT examples", y="OOD probs.", group="Model", color="Model", fill="Model")) +
    # stat_summary(geom="line") + stat_summary()
    stat_summary(geom="line") + stat_summary(geom="point") + facet_grid("~HMM", labeller="label_context") +
    theme_bw() + theme(axis_text_x=element_text(rotation=90)) + scale_x_log10()
)
plot.save('paper/betas_dpo.pdf', width=5, height=3)

plot = (
    ggplot(temp, aes(x="# SFT examples", y="ICL efficiency", group="Model", color="Model", fill="Model")) +
    # stat_summary(geom="line") + stat_summary()
    stat_summary(geom="line") + stat_summary(geom="point") + scale_x_log10()
)
plot.save('paper/K_dpo.pdf', width=5, height=3)

# make latex table for appendix
average_nrmse = params_dpo_df[params_dpo_df["dpo_amount"] > 1].copy()
average_nrmse = average_nrmse.groupby(['dpo_amount', 'layers', 'law'])['nrmse_prob'].mean().unstack()

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

# LLMs
llm_dfs = []
for file in glob.glob("logs/real-lms/*.csv"):
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
plot = (
    ggplot(temp, aes(x="Shots", y="Probability", color="Model", group="Model")) + geom_line() +
    facet_wrap("Dataset") + scale_x_log10() + theme(figure_size=(5, 3))
)
plot.save("paper/base_v_instruct.pdf", width=5, height=3)

posteriors_comp = []
for model in llm_df["model"].unique():
    if "Llama 3.1 8B" not in model: continue
    for dataset in llm_df["dataset"].unique():
        law = all_models_llm[(model, dataset)]['scoring']
        subset = llm_df[(llm_df["dataset"] == dataset) & (llm_df["model"] == model)]
        max_shots = subset["shots"].max()
        posteriors = law.estimate_nll(max_shots, 0, add_metrics=True)["posteriors"]
        for hmm in range(1):
            for i in range(1, len(posteriors)):
                posteriors_comp.append({
                    "Shots": i,
                    "HMM": f"HMM {hmm}",
                    "Posterior": posteriors[i, hmm].item(),
                    "Model": "Base" if "base" in model else "Instruct",
                    "Dataset": dataset.split("_")[-1],
                })

plot = (
    ggplot(pd.DataFrame(posteriors_comp), aes(x="Shots", y="Posterior", color="Model")) + facet_wrap("Dataset") +
    geom_line() + scale_x_log10() + theme(figure_size=(5, 3))
)
plot.save("paper/base_v_instruct_posterior.pdf", width=5, height=3)