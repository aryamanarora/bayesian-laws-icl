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


def power_law_fit(n, C, alpha, K):
    """Power law fit function. Params should be positive."""
    return C * n**(-alpha) + 0


def bayesian_fit(n, g0, gamma, beta, K):
    # return -np.log((C * (gamma - beta)) / (C - (C - 1) * (beta / gamma)**n) + beta)
    # res = (gamma - beta) / (1 - ((g0 - 1) / g0) * (beta / gamma)**(K * n)) + beta
    res = (gamma - beta) / (1 + np.exp(-K * (n - g0))) + beta
    return -np.log(res)


def bernoulli_fit(n, g0, gamma, beta):
    probs_under_g = np.array([binom.pmf(count_A, n, gamma) for count_A in range(0, n + 1)])
    probs_under_b = np.array([binom.pmf(count_A, n, beta) for count_A in range(0, n + 1)])
    p_g_given_seq = (probs_under_g * g0 / (probs_under_g * g0 + probs_under_b * (1 - g0)))
    p_g_exp = np.sum(p_g_given_seq * probs_under_g)
    p_b_exp = 1 - p_g_exp
    p_d = gamma * p_g_exp + beta * p_b_exp
    return -np.log(p_d)


def analyse_folder(
    directory="logs/titrate-hmm0/"
):
    # remove pngs, csvs, and pdfs in directory (non-recursive)
    for file in os.listdir(directory):
        if file.endswith(".png") or file.endswith(".csv") or file.endswith(".pdf"):
            os.remove(os.path.join(directory, file))

    # load data
    all_power_law_params = []
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
        power_law_params = {}
        power_law_params_list = []
        for perc in df['perc'].unique():
            # print(perc)
            for k in df['k'].unique():
                for hmm in df['hmm'].unique():
                    # get subset for this exp
                    subset = df[(df['perc'] == perc) & (df['hmm'] == hmm) & (df['k'] == k)]

                    # fit power law
                    params = [0.5, 0.9, 0.1, 1.0]
                    popt, pcov = curve_fit(
                        bayesian_fit, subset['shots'], subset['nll'],
                        p0=params, maxfev=10000,
                        bounds=([-np.inf, 0, 0, -np.inf], [+np.inf, 1, 1, +np.inf])
                    )
                    g0, gamma, beta, K = popt
                    print(f"{perc} -- {k}, {hmm}: g0={g0}, gamma={gamma}, beta={beta}, k={K}")

                    # store
                    power_law_params[(perc, k, hmm)] = (g0, gamma, beta, K)
                    power_law_params_list.append({
                        "perc": perc,
                        "k": k,
                        "hmm": hmm,
                        "g0": g0,
                        "gamma": gamma,
                        "beta": beta,
                        "K": K
                    })
            #     print()
            # print('-------------------')

        # store power law estimates in df
        def estimate_nll(row):
            g0, gamma, beta, K = power_law_params[(row['perc'], row['k'], row['hmm'])]
            return bayesian_fit(row['shots'], g0, gamma, beta, K)
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

        power_law_params_df = pd.DataFrame(power_law_params_list)
        power_law_params_df.to_csv(f"{directory}/power_law_params{suffix}.csv", index=False)
        power_law_params_df['layers'] = int(suffix.replace('-', '')) if suffix != '' else 4
        all_power_law_params.append(power_law_params_df)
    
    # plot power law params
    power_law_params_df = pd.concat(all_power_law_params)
    power_law_params_df = power_law_params_df.groupby(['perc', 'k', 'hmm', 'layers']).mean().reset_index()

    plot = (
        ggplot(power_law_params_df, aes(x="perc", y="g0", color="hmm", group="hmm")) +
        facet_grid("layers~k", labeller="label_both") +
        geom_line() + geom_point() +
        theme(axis_text_x = element_text(angle=-90, hjust=0.5))
    )
    plot.save(f"{directory}/power_law_g0.png", dpi=300)

    plot = (
        ggplot(power_law_params_df, aes(x="perc", y="gamma", color="hmm", group="hmm")) +
        facet_grid("layers~k", labeller="label_both") +
        geom_line() + geom_point() +
        theme(axis_text_x = element_text(angle=-90, hjust=0.5))
    )
    plot.save(f"{directory}/power_law_gamma.png", dpi=300)

    plot = (
        ggplot(power_law_params_df, aes(x="perc", y="beta", color="hmm", group="hmm")) +
        facet_grid("layers~k", labeller="label_both") +
        geom_line() + geom_point() +
        theme(axis_text_x = element_text(angle=-90, hjust=0.5))
    )
    plot.save(f"{directory}/power_law_beta.png", dpi=300)

    plot = (
        ggplot(power_law_params_df, aes(x="perc", y="K", color="hmm", group="hmm")) +
        facet_grid("layers~k", labeller="label_both") +
        geom_line() + geom_point() +
        theme(axis_text_x = element_text(angle=-90, hjust=0.5))
    )
    plot.save(f"{directory}/power_law_K.png", dpi=300)


def main():
    parser = argparse.ArgumentParser(description='Analyse results')
    parser.add_argument('--directory', type=str, default="logs/titrate-hmm0/", help='Directory to analyse')
    args = parser.parse_args()

    analyse_folder(**vars(args))


if __name__ == "__main__":
    main()
