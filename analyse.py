"""
Code for fitting Bayesian laws to the ICL behaviour from the
training/SFT/RLHF experiments.
"""

from plotnine import (
    ggplot, aes, geom_line, geom_point, facet_grid, stat_summary,
    theme, element_text, theme_bw
)
from plotnine.scales import scale_y_log10, scale_x_log10
from scipy.optimize import curve_fit
from scipy.stats import binom
import pandas as pd
import numpy as np
import argparse
import os
import math
import torch
import json
from tqdm import tqdm
from copy import deepcopy
from collections import defaultdict


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class PowerLawFit(torch.nn.Module):
    def __init__(self):
        super(PowerLawFit, self).__init__()
        self.C = torch.nn.Parameter(torch.tensor(1.0))
        self.alpha = torch.nn.Parameter(torch.tensor(1.0))
        self.K = torch.nn.Parameter(torch.tensor(1.0))
    
    def get_params(self):
        return {
            "C": self.C.exp().item(),
            "alpha": self.alpha.exp().item(),
            "K": self.K.exp().item()
        }
    
    def forward(self, shots):
        if type(shots) == int:
            shots = torch.tensor([shots], dtype=torch.float32).to(DEVICE)
        return self.C.exp() * (shots).pow(-self.alpha.exp()) + self.K.exp()


class BoundedPowerLawFit(torch.nn.Module):
    def __init__(self):
        super(BoundedPowerLawFit, self).__init__()
        self.C = torch.nn.Parameter(torch.tensor(1.0))
        self.alpha = torch.nn.Parameter(torch.tensor(1.0))
        self.K = torch.nn.Parameter(torch.tensor(1.0))
        self.n_c = torch.nn.Parameter(torch.tensor(1.0))

    def get_params(self):
        return {
            "C": self.C.exp().item(),
            "alpha": self.alpha.exp().item(),
            "K": self.K.exp().item(),
            "n_c": self.n_c.exp().item()
        }
    
    def forward(self, shots):
        if type(shots) == int:
            shots = torch.tensor([shots], dtype=torch.float32).to(DEVICE)
        return self.C.exp() * (1 + shots / self.n_c.exp()).pow(-self.alpha.exp()) + self.K.exp()


class LogisticLawFit(torch.nn.Module):
    def __init__(self):
        super(LogisticLawFit, self).__init__()
        self.C = torch.nn.Parameter(torch.tensor(1.0))
        self.alpha = torch.nn.Parameter(torch.tensor(1.0))
        self.K = torch.nn.Parameter(torch.tensor(1.0))
        self.n_c = torch.nn.Parameter(torch.tensor(1.0))
    
    def get_params(self):
        return {
            "C": self.C.exp().item(),
            "alpha": self.alpha.exp().item(),
            "K": self.K.exp().item(),
            "n_c": self.n_c.item()
        }
    
    def forward(self, shots):
        if type(shots) == int:
            shots = torch.tensor([shots], dtype=torch.float32).to(DEVICE)
        return -(self.C.exp() / (1 + (-self.alpha.exp() * (shots - self.n_c)).exp()) + self.K.exp()).log()
    

power_law_mapping = {
    "power": PowerLawFit,
    "bounded": BoundedPowerLawFit,
    "logistic": LogisticLawFit,
}


def fit_power_law(subset: pd.DataFrame, type="power"):
    # fit power law
    model = power_law_mapping[type]()
    model.to(DEVICE)
    iterator = tqdm(range(50))
    patience = 5
    batch_size = 5
    history = []
    subset = subset.sample(frac=1.0)

    optimizer = torch.optim.Adam(model.parameters(), lr=5e-2)

    for _ in iterator:
        avg_loss = 0.0
        loss = 0.0
        for i in range(0, len(subset), batch_size):
            optimizer.zero_grad()
            batch = subset.iloc[i:i+batch_size]
            shots = torch.tensor(batch['shots'].values, dtype=torch.float32).to(DEVICE)
            true_nll = torch.tensor(batch['nll'].values, dtype=torch.float32).to(DEVICE)
            est_nll = model(shots)
            loss = ((true_nll - est_nll)**2).sum()
            loss.backward()
            optimizer.step()
            avg_loss += loss.item()

        # print
        avg_loss /= len(subset)
        history.append(avg_loss)
        result = {
            "loss": avg_loss,
            "C": model.C.item(),
            "alpha": model.alpha.item(),
            "K": model.K.item(),
        }
        if hasattr(model, 'n_c'):
            result['n_c'] = model.n_c.item()
        iterator.set_postfix(result)

        # early stopping
        if len(history) > patience and all([math.isclose(history[-1], x, rel_tol=5e-2) for x in history[-patience:]]):
            break
    
    return model


class BayesianLawFit(torch.nn.Module):
    def __init__(self, num_hmms):
        super(BayesianLawFit, self).__init__()
        self.num_hmms = num_hmms
        self.masks = torch.eye(num_hmms, dtype=torch.bool).to(DEVICE)
        self.priors = torch.nn.Parameter(torch.zeros(num_hmms))
        self.gammas = torch.nn.Parameter(torch.zeros(num_hmms) + 1)
        self.betas = torch.nn.Parameter(torch.zeros(num_hmms) - 1)
        self.K = torch.nn.Parameter(torch.tensor(0.0))
    
    def get_prior(self):
        return torch.nn.functional.softmax(self.priors, dim=0)

    def get_gammas(self):
        return torch.sigmoid(self.gammas)

    def get_betas(self):
        return torch.sigmoid(self.betas)
    
    def get_K(self):
        return torch.exp(self.K)

    def get_params(self):
        return {
            "priors": self.get_prior().tolist(),
            "gammas": self.get_gammas().tolist(),
            "betas": self.get_betas().tolist(),
            "K": self.get_K().item()
        }

    def forward(self, shots, hmm):
        priors = self.get_prior().log()
        gammas = self.get_gammas().log()
        betas = self.get_betas().log()
        K = self.get_K()
        shots = shots * K
        if isinstance(shots, torch.Tensor):
            shots = shots.unsqueeze(-1)
        # print("hmm:", hmm)
        # print("K:", self.K.item())
        # print("p(hmm):", priors.exp().tolist())
        p_under_dist = torch.where(self.masks[hmm], gammas, betas)
        # print("p(d | hmm):", p_under_dist.exp().tolist())
        p_seq_under_dist = p_under_dist * shots
        # print("p(D | hmm):", p_seq_under_dist.exp().tolist())
        posteriors = torch.nn.functional.softmax(priors + p_seq_under_dist, dim=-1) # already in log space
        # print("p(hmm | D):", posteriors.tolist())
        p_data = (posteriors * p_under_dist.exp()).sum(dim=-1)
        # print("p(d | D, hmm):", (posteriors * p_under_dist.exp()).tolist())
        # print("p(d | D):", p_data.item())
        est_nll = -torch.log(p_data)
        # print("NLL:", nll, est_nll.item())
        # input()
        return est_nll


def fit_bayesian_law(subset: pd.DataFrame):
    # fit power law
    model = BayesianLawFit(len(subset['hmm'].unique()))
    model.to(DEVICE)
    iterator = tqdm(range(100))
    patience = 5
    batch_size = 20
    history = []
    subset = subset.sample(frac=1.0)

    optimizer = torch.optim.Adam(model.parameters(), lr=5e-2)

    for _ in iterator:
        avg_loss = 0.0
        loss = 0.0
        for i in range(0, len(subset), batch_size):
            optimizer.zero_grad()
            batch = subset.iloc[i:i+batch_size]
            shots = torch.tensor(batch['shots'].values, dtype=torch.float32).to(DEVICE)
            hmm = torch.tensor(list(map(int, batch['hmm'].values)), dtype=torch.int32).to(DEVICE)
            true_nll = torch.tensor(batch['nll'].values, dtype=torch.float32).to(DEVICE)
            est_nll = model(shots, hmm)
            loss = ((true_nll - est_nll)**2).sum()
            loss.backward()
            optimizer.step()
            avg_loss += loss.item()

        # print
        avg_loss /= len(subset)
        history.append(avg_loss)
        result = {
            "loss": avg_loss,
            "gamma_0": model.get_gammas()[0].item(),
            "beta_0": model.get_betas()[0].item(),
            "K": model.get_K().item(),
        }
        result.update({f"p_{i}": p.item() for i, p in enumerate(model.get_prior())})
        iterator.set_postfix(result)

        # early stopping
        if len(history) > patience and all([math.isclose(history[-1], x, rel_tol=5e-2) for x in history[-patience:]]):
            break
    
    return model



def analyse_folder(
    pretrain: str="1,1,1,1,1",
    sft: str="1,0,0,0,0",
    layers: str="4,8,12"
):
    # load data
    dfs = []
    for layer in layers.split(","):
        # set up dir
        directory = f"logs/{layer}-{pretrain}-{sft}/"
        if os.path.exists(f"{directory}/in_context_probs.csv"):
            # load data
            data = pd.read_csv(f"{directory}/in_context_probs.csv")
            data['layers'] = int(layer)
            data['method'] = 'sft'
            dfs.append(data)
        else:
            print(f"Directory {directory} does not exist")
        
        # dpo
        directory = f"logs/{layer}-{pretrain}-{sft}-dpo/"
        if os.path.exists(f"{directory}/in_context_probs.csv"):
            # load data
            data = pd.read_csv(f"{directory}/in_context_probs.csv")
            data['layers'] = int(layer)
            data['method'] = 'dpo'
            dfs.append(data)
        else:
            print(f"Directory {directory} does not exist")

        # set up inf setting dir
        directory = f"logs/{layer}-{sft}-{sft}/"
        if os.path.exists(f"{directory}/in_context_probs.csv"):
            # load data
            data2 = pd.read_csv(f"{directory}/in_context_probs.csv")
            data2['layers'] = int(layer)
            sft_dummy = 2 * data['sft'].max()
            data2['sft'] = 'none'
            data2['sft_amount'] = 0
            data2['method'] = 'sft'
            dfs.append(data2)
        else:
            print(f"Directory {directory} does not exist")

    # format df
    df_all = pd.concat(dfs)
    df_all = df_all[df_all['shots'] > 0] # remove 0 shots
    df_all['hmm'] = df_all['hmm'].astype(str)
    # no sft for inf setting
    order = list(map(str, sorted(df_all['sft_amount'].unique()))) + ['none']
    df_all['sft_amount'] = df_all.apply(lambda x: 'none' if x['sft'] == 'none' else x['sft_amount'], axis=1)
    df_all['sft_amount'] = pd.Categorical(df_all['sft_amount'].astype(str), categories=order, ordered=True)
    df_all['sft'] = df_all['sft'].astype(str)

    # directory for plots
    directory = f"figs/{pretrain}-{sft}/"
    if not os.path.exists(directory):
        os.makedirs(directory)
    else:
        for file in os.listdir(directory):
            os.remove(f"{directory}/{file}")

    # fit power laws
    params_dfs = []
    for method in sorted(list(df_all['method'].unique())):
        for layer in layers.split(","):
            # get subset
            layer = int(layer)
            df = df_all[
                (df_all['layers'] == layer) &
                (df_all['sft'].isin(["True", "False", "none"])) &
                (df_all['method'] == method)
            ]
            if len(df) == 0:
                continue

            # get power law fit
            print(f"Power law fit for {layer}-layer model, {method} method")
            all_params = defaultdict(dict)
            all_params_list = []

            # each exp
            for sft_amount in df['sft_amount'].unique():
                for k in df['k'].unique():
                    print(f"{sft_amount} SFT -- shot length {k}")
                    subset = df[(df['sft_amount'] == sft_amount) & (df['k'] == k)]

                    # BAYESIAN LAW
                    model = fit_bayesian_law(subset)
                    all_params['bayesian'][(sft_amount, k)] = model
                    for hmm in subset['hmm'].unique():
                        idx = int(hmm)
                        # collect params and store for plotting
                        params = {
                            "sft_amount": sft_amount,
                            "k": k,
                            "hmm": hmm,
                            "law": "bayesian",
                        }
                        params.update(model.get_params())
                        for key in params:
                            if isinstance(params[key], list) or isinstance(params[key], torch.Tensor):
                                params[key] = params[key][int(hmm)]
                        all_params_list.append(params)
                    
                    # POWER LAWS
                    for hmm in subset['hmm'].unique():
                        subset_hmm = subset[subset['hmm'] == hmm]
                        for law in power_law_mapping.keys():
                            model = fit_power_law(subset_hmm, type=law)
                            all_params[law][(sft_amount, k, int(hmm))] = model

                            # collect params and store for plotting
                            params = {
                                "sft_amount": sft_amount,
                                "k": k,
                                "hmm": hmm,
                                "law": law,
                            }
                            params.update(model.get_params())
                            all_params_list.append(params)
                            

            # store bayesian law estimates in df
            def estimate_nll(row):
                model = all_params['bayesian'][(row['sft_amount'], row['k'])]
                return model(row['shots'], int(row['hmm'])).item()
            df.loc[:, "est_nll_bayesian"] = df.apply(estimate_nll, axis=1)
            df.loc[:, "mse_bayesian"] = (df["nll"] - df["est_nll_bayesian"])**2

            # and power law estimates
            for law in power_law_mapping.keys():
                def estimate_nll(row):
                    model = all_params[law][(row['sft_amount'], row['k'], int(row['hmm']))]
                    return model(row['shots']).item()
                df[f"est_nll_{law}"] = df.apply(estimate_nll, axis=1)
                df[f"mse_{law}"] = (df["nll"] - df[f"est_nll_{law}"])**2

            # print average MSE for each model
            print(f"Average MSE for {layer}-layer model, {method} method")
            for law in list(power_law_mapping.keys()) + ['bayesian']:
                print(f"{law}:", df.groupby(['sft_amount', 'k', 'hmm', 'method', 'sft'], observed=True).mean()[f"mse_{law}"].mean())
            
            # store mses in params list
            for i in range(len(all_params_list)):
                row = all_params_list[i]
                mse = df[
                    (df['sft_amount'] == row['sft_amount']) &
                    (df['k'] == row['k']) &
                    (df['hmm'] == row['hmm'])
                ][f'mse_{row["law"]}'].mean()
                rmse = math.sqrt(mse)
                nrmse = rmse / df['nll'].mean()
                all_params_list[i]['mse'] = mse
                all_params_list[i]['rmse'] = rmse
                all_params_list[i]['nrmse'] = nrmse

            # plot
            suffix = f"-{layer}-{method}"
            df = df.drop(columns=['sft'])
            df_summary = df.groupby(['sft_amount', 'k', 'hmm', 'shots', 'method'], observed=True).mean().reset_index()
            plot = (
                ggplot(df_summary, aes(x='shots', y='prob', color='sft_amount', group='sft_amount')) +
                facet_grid("k~hmm", labeller="label_both") +
                geom_line()
            )
            plot.save(f"{directory}/in_context_probs{suffix}.png", dpi=300)

            plot = (
                ggplot(df_summary) +
                facet_grid('k~hmm', labeller='label_both') +
                geom_line(aes(x='shots', y='est_nll_bayesian', color='sft_amount', group='sft_amount')) +
                geom_point(aes(x='shots', y='nll', color='sft_amount'), size=1.0, stroke=0, alpha=0.4) +
                scale_y_log10() + scale_x_log10()
            )
            plot.save(f"{directory}/in_context_probs_nll{suffix}.png", dpi=300)

            all_law_params_df = pd.DataFrame(all_params_list)
            all_law_params_df.to_csv(f"{directory}/all_law_params{suffix}.csv", index=False)
            all_law_params_df['layers'] = layer
            all_law_params_df['method'] = method
            params_dfs.append(all_law_params_df)


def analyse_loss(
    pretrain: str="1,1,1,1,1",
    sft: str="1,0,0,0,0",
    layers: str="4,8,12"
):
    dfs = []
    for layer in layers.split(","):
        # set up dir
        directory = f"logs/{layer}-{pretrain}-{sft}/"
        if os.path.exists(f"{directory}/trainer_state.json"):
            print("Loading", directory)
            # load data
            with open(f"{directory}/trainer_state.json", "r") as f:
                data = pd.DataFrame(json.load(f)["log_history"])
            data['layers'] = int(layer)
            dfs.append(data)
        else:
            print(f"Directory {directory} does not exist")
    
    # format df
    df_all = pd.concat(dfs)
    order = list(map(str, sorted(df_all['layers'].unique())))
    df_all['layers'] = df_all['layers'].astype(str)
    df_all['layers'] = pd.Categorical(df_all['layers'], categories=order, ordered=True)

    # plot
    plot = (
        ggplot(df_all, aes(x='epoch', y='loss', color='layers', group='layers')) +
        geom_line() + theme_bw() +
        theme(axis_text_x=element_text(rotation=90))
    )
    plot.save(f"figs/loss-{pretrain}-{sft}.pdf", width=4, height=3)


def analyse_params(
    pretrain: str="1,1,1,1,1",
    sft: str="1,0,0,0,0",
    layers: str="4,8,12"
):
    dfs = []
    for layer in layers.split(","):
        # set up dir
        directory = f"figs/{pretrain}-{sft}/"
        if os.path.exists(f"{directory}/all_law_params-{layer}-sft.csv"):
            # load data
            data = pd.read_csv(f"{directory}/all_law_params-{layer}-sft.csv")
            data['layers'] = int(layer)
            dfs.append(data)
        else:
            print(f"File {directory}/all_law_params-{layer}-sft.csv does not exist")
    
    # format df
    df_all = pd.concat(dfs)
    order = list(map(str, sorted(df_all['layers'].unique())))
    df_all['layers'] = df_all['layers'].astype(str)

    # drop sft_amount == "none"
    df_all = df_all[df_all['sft_amount'] != "none"]
    df_all['sft_amount'] = df_all['sft_amount'].astype(float)

    # plot
    for law in df_all['law'].unique():
        df_filtered = df_all[df_all['law'] == law]

        if law == "bayesian":
            # make gamma and beta into one column, with metric as another column
            df = pd.melt(df_filtered, id_vars=['sft_amount', 'k', 'hmm', 'layers'], value_vars=['gamma', 'beta'], var_name='metric', value_name='value')

        for column in df_filtered.columns:
            if column in ['sft_amount', 'k', 'hmm', 'law', 'mse', 'layers']:
                continue
            plot = (
                ggplot(df_filtered, aes(x='sft_amount', y=column, color='layers', group='layers')) +
                geom_line() + geom_point() + theme_bw() + scale_x_log10() +
                theme(axis_text_x=element_text(rotation=90)) +
                facet_grid("k~hmm", labeller="label_both")
            )
            plot.save(f"figs/{pretrain}-{sft}/{law}-{column}-{pretrain}-{sft}.pdf", width=10, height=10)


def main():
    parser = argparse.ArgumentParser(description='Analyse results')
    parser.add_argument('--pretrain', type=str, default="1,1,1,1,1", help='Pretrain amounts')
    parser.add_argument('--sft', type=str, default="1,0,0,0,0", help='SFT amounts')
    parser.add_argument('--layers', type=str, default="1,2,3,4,8,12,16", help='Number of layers')
    parser.add_argument('--loss', action='store_true', help='Analyse loss')
    parser.add_argument('--params', action='store_true', help='Analyse params')
    args = parser.parse_args()

    if args.loss:
        analyse_loss(pretrain=args.pretrain, sft=args.sft, layers=args.layers)
    elif args.params:
        analyse_params(pretrain=args.pretrain, sft=args.sft, layers=args.layers)
    else:
        analyse_folder(pretrain=args.pretrain, sft=args.sft, layers=args.layers)


if __name__ == "__main__":
    main()
