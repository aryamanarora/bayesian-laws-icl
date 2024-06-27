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


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class BayesianLawFit(torch.nn.Module):
    def __init__(self, num_hmms):
        super(BayesianLawFit, self).__init__()
        self.num_hmms = num_hmms
        self.masks = []
        for i in range(num_hmms):
            mask = (torch.arange(self.num_hmms) == i).to(DEVICE)
            self.masks.append(mask)
        self.priors = torch.nn.Parameter(torch.zeros(num_hmms))
        self.gammas = torch.nn.Parameter(torch.zeros(num_hmms) + 1)
        self.betas = torch.nn.Parameter(torch.zeros(num_hmms) - 1)
        self.K = torch.nn.Parameter(torch.tensor(1.0))
    
    def get_prior(self):
        return torch.nn.functional.softmax(self.priors, dim=0)

    def get_gammas(self):
        return torch.sigmoid(self.gammas)

    def get_betas(self):
        return torch.sigmoid(self.betas)
    
    def get_K(self):
        return torch.sigmoid(self.K)

    def forward(self, shots, hmm):
        priors = self.get_prior().log()
        gammas = self.get_gammas().log()
        betas = self.get_betas().log()
        # print("hmm:", hmm)
        # print("K:", self.K.item())
        # print("p(hmm):", priors.exp().tolist())
        p_under_dist = torch.where(self.masks[hmm], gammas, betas)
        # print("p(d | hmm):", p_under_dist.exp().tolist())
        p_seq_under_dist = p_under_dist * (shots * self.get_K())
        # print("p(D | hmm):", p_seq_under_dist.exp().tolist())
        posteriors = torch.nn.functional.softmax(priors + p_seq_under_dist, dim=0) # already in log space
        # print("p(hmm | D):", posteriors.tolist())
        p_data = (posteriors * p_under_dist.exp()).sum()
        # print("p(d | D, hmm):", (posteriors * p_under_dist.exp()).tolist())
        # print("p(d | D):", p_data.item())
        est_nll = -torch.log(p_data)
        # print("NLL:", nll, est_nll.item())
        # input()
        return est_nll


def fit_bayesian_law(subset: pd.DataFrame):
    # prep data
    subset = subset.sample(frac=1.0)
    num_hmms = len(subset['hmm'].unique())

    # fit power law
    model = BayesianLawFit(num_hmms).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
    iterator = tqdm(range(50))
    patience = 5
    history = []
    for _ in iterator:
        avg_loss = 0.0
        for i, row in subset.iterrows():
            optimizer.zero_grad()
            est_nll = model(row['shots'], int(row['hmm']))
            loss = (row['nll'] - est_nll)**2
            avg_loss += loss.item()
            loss.backward()
            optimizer.step()

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
        if len(history) > patience and all([math.isclose(history[-1], x, rel_tol=1e-2) for x in history[-patience:]]):
            break
    
    return model


def analyse_folder(
    directory="logs/titrate-hmm0/"
):
    # remove pngs, csvs, and pdfs in directory (non-recursive)
    for file in os.listdir(directory):
        if file.endswith(".png") or file.endswith(".csv") or file.endswith(".pdf"):
            os.remove(os.path.join(directory, file))

    # load data
    all_bayesian_law_params = []
    for suffix in ['', '-4', '-8', '-12']:
        folders = ['1perc', '5perc', '10perc', '20perc', '50perc', '100perc', 'infperc']
        dfs = []
        for folder in folders:
            file = f"{directory}/{folder}{suffix}/in_context_probs.csv"
            if not os.path.exists(file):
                continue
            df = pd.read_csv(file)

            # add sft data
            df_sft = df[df["sft"] == "True"] if folder != 'infperc' else df[df["sft"] == False]
            df_sft['perc'] = str(float(folder.replace('perc', '')) / 100.0)
            dfs.append(df_sft)

            # and un-sft data
            if folder != 'infperc':
                df_base = df[df["sft"] == "False"]
                df_base.loc[:, 'perc'] = "0"
                dfs.append(df_base)

        if len(dfs) == 0:
            continue
        df = pd.concat(dfs)

        # format df
        df['shots'] += 1 # add one to shots to avoid log(0)
        df['hmm'] = df['hmm'].astype(str)

        # get power law fit
        print(f"Power law fit for {directory}, {suffix}")
        bayesian_law_params = {}
        bayesian_law_params_list = []

        for perc in df['perc'].unique():
            for k in df['k'].unique():
                print(f"{perc} SFT -- shot length {k}")

                # fit power law
                subset = df[(df['perc'] == perc) & (df['k'] == k)]
                model = fit_bayesian_law(subset)

                # store
                bayesian_law_params[(perc, k)] = model
                for hmm in subset['hmm'].unique():
                    idx = int(hmm)
                    bayesian_law_params_list.append({
                        "perc": perc,
                        "k": k,
                        "hmm": idx,
                        "prior": model.get_prior()[idx].item(),
                        "gamma": model.get_gammas()[idx].item(),
                        "beta": model.get_betas()[idx].item(),
                        "K": model.get_K().item(),
                    })

        # store power law estimates in df
        def estimate_nll(row):
            model = bayesian_law_params[(row['perc'], row['k'])]
            return model(row['shots'], int(row['hmm'])).item()
        df["est_nll"] = df.apply(estimate_nll, axis=1)

        # plot
        df = df.drop(columns=['sft'])
        df_summary = df.groupby(['perc', 'k', 'hmm', 'shots']).mean().reset_index()
        plot = (
            ggplot(df_summary, aes(x='shots', y='prob', color='perc', group='perc')) +
            facet_grid("k~hmm", labeller="label_both") +
            geom_line()
        )
        plot.save(f"{directory}/in_context_probs{suffix}.png", dpi=300)

        plot = (
            ggplot(df_summary) +
            facet_grid('k~hmm', labeller='label_both') +
            geom_line(aes(x='shots', y='est_nll', color='perc', group='perc')) +
            geom_point(aes(x='shots', y='nll', color='perc'), size=1.0, stroke=0, alpha=0.4) +
            scale_y_log10() + scale_x_log10()
        )
        plot.save(f"{directory}/in_context_probs_nll{suffix}.png", dpi=300)

        bayesian_law_params_df = pd.DataFrame(bayesian_law_params_list)
        bayesian_law_params_df.to_csv(f"{directory}/bayesian_law_params{suffix}.csv", index=False)
        bayesian_law_params_df['layers'] = int(suffix.replace('-', '')) if suffix != '' else 4
        all_bayesian_law_params.append(bayesian_law_params_df)
    
    # plot power law params
    bayesian_law_params_df = pd.concat(all_bayesian_law_params)
    bayesian_law_params_df = bayesian_law_params_df.groupby(['perc', 'k', 'hmm', 'layers']).mean().reset_index()

    for variable in ['prior', 'gamma', 'beta', 'K']:
        plot = (
            ggplot(bayesian_law_params_df, aes(x="perc", y=variable, color="hmm", group="hmm")) +
            facet_grid("layers~k", labeller="label_both") +
            geom_line() + geom_point() +
            theme(axis_text_x = element_text(angle=-90, hjust=0.5))
        )
        plot.save(f"{directory}/bayesian_law_{variable}.png", dpi=300)


def main():
    parser = argparse.ArgumentParser(description='Analyse results')
    parser.add_argument('--directory', type=str, default="logs/titrate-hmm0/", help='Directory to analyse')
    args = parser.parse_args()

    analyse_folder(**vars(args))


if __name__ == "__main__":
    main()
