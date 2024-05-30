from plotnine import (
    ggplot, aes, geom_line, geom_point, facet_grid, stat_summary,
    theme, element_text
)
from plotnine.scales import scale_y_log10, scale_x_log10
from scipy.optimize import curve_fit
import pandas as pd
import numpy as np
import argparse
import os


def power_law_fit(n, C, alpha, K):
    """Power law fit function. Params should be positive."""
    return C * n**(-alpha) + 0


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
                    params = [5.0, 0.5, 0.0]
                    popt, pcov = curve_fit(
                        power_law_fit, subset['shots'], subset['nll'],
                        p0=params, maxfev=10000,
                        # bounds=(0, +np.inf)
                    )
                    C, alpha, K = popt
                    # print(f"{perc} -- {k}, {hmm}: C={C}, α={alpha}, K={K}")

                    # store
                    power_law_params[(perc, k, hmm)] = (C, alpha, K)
                    power_law_params_list.append({
                        "perc": perc,
                        "k": k,
                        "hmm": hmm,
                        "C": C,
                        "alpha": alpha,
                        "K": K
                    })
            #     print()
            # print('-------------------')

        # store power law estimates in df
        def estimate_nll(row):
            C, alpha, K = power_law_params[(row['perc'], row['k'], row['hmm'])]
            return power_law_fit(row['shots'], C, alpha, K)
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
        ggplot(power_law_params_df, aes(x="perc", y="alpha", color="hmm", group="hmm")) +
        facet_grid("layers~k", labeller="label_both") +
        geom_line() + geom_point() +
        theme(axis_text_x = element_text(angle=-90, hjust=0.5))
    )
    plot.save(f"{directory}/power_law_alpha.png", dpi=300)

    plot = (
        ggplot(power_law_params_df, aes(x="perc", y="C", color="hmm", group="hmm")) +
        facet_grid("layers~k", labeller="label_both") +
        geom_line() + geom_point() + scale_y_log10() +
        theme(axis_text_x = element_text(angle=-90, hjust=0.5))
    )
    plot.save(f"{directory}/power_law_C.png", dpi=300)


def main():
    parser = argparse.ArgumentParser(description='Analyse results')
    parser.add_argument('--directory', type=str, default="logs/titrate-hmm0/", help='Directory to analyse')
    args = parser.parse_args()

    analyse_folder(**vars(args))


if __name__ == "__main__":
    main()
