"""
Code for fitting Bayesian laws to the ICL behaviour from the
training/SFT/RLHF experiments.
"""

from plotnine import (
    ggplot, aes, geom_line, geom_point, facet_grid, stat_summary,
    theme, element_text
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
    # prep data
    subset = subset.sample(frac=1.0)
    subset['hmm'] = subset['hmm'].astype(int)
    num_hmms = len(subset['hmm'].unique())

    # fit power law
    model = power_law_mapping[type]()
    model.to(DEVICE)
    iterator = tqdm(range(100))
    patience = 5
    batch_size = 5
    history = []

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
        if len(history) > patience and all([math.isclose(history[-1], x, rel_tol=1e-3) for x in history[-patience:]]):
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

    def forward(self, shots, hmm):
        priors = self.get_prior().log()
        gammas = self.get_gammas().log()
        betas = self.get_betas().log()
        # print("hmm:", hmm)
        # print("K:", self.K.item())
        # print("p(hmm):", priors.exp().tolist())
        p_under_dist = torch.where(self.masks[hmm], gammas, betas)
        # print("p(d | hmm):", p_under_dist.exp().tolist())
        p_seq_under_dist = p_under_dist * (shots * self.get_K()).unsqueeze(-1)
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
    # prep data
    subset = subset.sample(frac=1.0)
    subset['hmm'] = subset['hmm'].astype(int)
    num_hmms = len(subset['hmm'].unique())

    # fit power law
    model = BayesianLawFit(num_hmms)
    model.to(DEVICE)
    iterator = tqdm(range(100))
    patience = 5
    batch_size = 20
    history = []

    optimizer = torch.optim.Adam(model.parameters(), lr=5e-2)

    for _ in iterator:
        avg_loss = 0.0
        loss = 0.0
        for i in range(0, len(subset), batch_size):
            optimizer.zero_grad()
            batch = subset.iloc[i:i+batch_size]
            shots = torch.tensor(batch['shots'].values, dtype=torch.float32).to(DEVICE)
            hmm = torch.tensor(batch['hmm'].values, dtype=torch.int32).to(DEVICE)
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
        if len(history) > patience and all([math.isclose(history[-1], x, rel_tol=1e-3) for x in history[-patience:]]):
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
            data2['sft'] = sft_dummy
            data2['sft_amount'] = sft_dummy
        else:
            print(f"Directory {directory} does not exist")

    # format df
    df_all = pd.concat(dfs)
    df_all = df_all[df_all['shots'] > 0] # remove 0 shots
    df_all['hmm'] = df_all['hmm'].astype(str)

    # directory for plots
    directory = f"figs/{pretrain}-{sft}/"
    if not os.path.exists(directory):
        os.makedirs(directory)
    else:
        for file in os.listdir(directory):
            os.remove(f"{directory}/{file}")

    # fit power laws
    all_bayesian_law_params = []
    for layer in layers.split(","):
        # get subset
        layer = int(layer)
        df = df_all[(df_all['layers'] == layer) & (df_all['sft'] == df_all['sft_amount'])]

        # get power law fit
        print(f"Power law fit for {layer}-layer model")
        bayesian_law_params = {}
        other_law_params = defaultdict(dict)
        bayesian_law_params_list = []

        # each exp
        for sft_amount in df['sft_amount'].unique():
            for k in df['k'].unique():
                print(f"{sft_amount} SFT -- shot length {k}")

                # BAYESIAN LAW
                subset = df[(df['sft_amount'] == sft_amount) & (df['k'] == k)]
                model = fit_bayesian_law(subset)

                # store
                bayesian_law_params[(sft_amount, k)] = model
                for hmm in subset['hmm'].unique():
                    idx = int(hmm)
                    bayesian_law_params_list.append({
                        "sft_amount": sft_amount,
                        "k": k,
                        "hmm": idx,
                        "prior": model.get_prior()[idx].item(),
                        "gamma": model.get_gammas()[idx].item(),
                        "beta": model.get_betas()[idx].item(),
                        "K": model.get_K().item(),
                    })
                
                # POWER LAWS
                for hmm in subset['hmm'].unique():
                    subset_hmm = subset[subset['hmm'] == hmm]
                    for law in power_law_mapping.keys():
                        model = fit_power_law(subset_hmm, type=law)
                        other_law_params[law][(sft_amount, k, int(hmm))] = model

        # store bayesian law estimates in df
        def estimate_nll(row):
            model = bayesian_law_params[(row['sft_amount'], row['k'])]
            return model(row['shots'], int(row['hmm'])).item()
        df["est_nll"] = df.apply(estimate_nll, axis=1)
        df["mse"] = (df["nll"] - df["est_nll"])**2

        # and power law estimates
        for law in other_law_params.keys():
            def estimate_nll(row):
                model = other_law_params[law][(row['sft_amount'], row['k'], int(row['hmm']))]
                return model(row['shots']).item()
            df[f"est_nll_{law}"] = df.apply(estimate_nll, axis=1)
            df[f"mse_{law}"] = (df["nll"] - df[f"est_nll_{law}"])**2

        # print average MSE for each model
        print(f"Average MSE for {layer}-layer model")
        print("Bayesian:", df.groupby(['sft_amount', 'k', 'hmm']).mean()["mse"].mean())
        for law in other_law_params.keys():
            print(f"{law}:", df.groupby(['sft_amount', 'k', 'hmm']).mean()[f"mse_{law}"].mean())

        # plot
        suffix = f"-{layer}"
        df = df.drop(columns=['sft'])
        df_summary = df.groupby(['sft_amount', 'k', 'hmm', 'shots']).mean().reset_index()
        plot = (
            ggplot(df_summary, aes(x='shots', y='prob', color='sft_amount', group='sft_amount')) +
            facet_grid("k~hmm", labeller="label_both") +
            geom_line()
        )
        plot.save(f"{directory}/in_context_probs{suffix}.png", dpi=300)

        plot = (
            ggplot(df_summary) +
            facet_grid('k~hmm', labeller='label_both') +
            geom_line(aes(x='shots', y='est_nll', color='sft_amount', group='sft_amount')) +
            geom_point(aes(x='shots', y='nll', color='sft_amount'), size=1.0, stroke=0, alpha=0.4) +
            scale_y_log10() + scale_x_log10()
        )
        plot.save(f"{directory}/in_context_probs_nll{suffix}.png", dpi=300)

        bayesian_law_params_df = pd.DataFrame(bayesian_law_params_list)
        bayesian_law_params_df.to_csv(f"{directory}/bayesian_law_params{suffix}.csv", index=False)
        bayesian_law_params_df['layers'] = int(suffix.replace('-', '')) if suffix != '' else 4
        all_bayesian_law_params.append(bayesian_law_params_df)
    
    # plot power law params
    bayesian_law_params_df = pd.concat(all_bayesian_law_params)
    bayesian_law_params_df = bayesian_law_params_df.groupby(['sft_amount', 'k', 'hmm', 'layers']).mean().reset_index()

    for variable in ['prior', 'gamma', 'beta', 'K']:
        plot = (
            ggplot(bayesian_law_params_df, aes(x="sft_amount", y=variable, color="hmm", group="hmm")) +
            facet_grid("layers~k", labeller="label_both") +
            geom_line() + geom_point() +
            theme(axis_text_x = element_text(angle=-90, hjust=0.5))
        )
        plot.save(f"{directory}/bayesian_law_{variable}.png", dpi=300)


def main():
    parser = argparse.ArgumentParser(description='Analyse results')
    parser.add_argument('--pretrain', type=str, default="1,1,1,1,1", help='Pretrain amounts')
    parser.add_argument('--sft', type=str, default="1,0,0,0,0", help='SFT amounts')
    parser.add_argument('--layers', type=str, default="4,8,12,16", help='Number of layers')
    args = parser.parse_args()

    analyse_folder(**vars(args))


if __name__ == "__main__":
    main()
