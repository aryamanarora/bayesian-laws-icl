from plotnine import ggplot, aes, geom_line, facet_grid, stat_summary
from plotnine.scales import scale_y_log10, scale_x_log10
from scipy.optimize import curve_fit
import pandas as pd
import numpy as np
import os


def power_law_fit(n, C, alpha, K):
    return C * n**(-alpha) + K


def analyse_folder(
    directory="logs/titrate-1/"
):
    # load data
    for suffix in ['', '-8', '-12']:
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

        df = pd.concat(dfs)

        # add one to shots to avoid log(0)
        df.loc[:, 'shots'] = df['shots'] + 1

        # get power law fit
        print(f"Power law fit for {folder}{suffix}")
        power_law_params = {}
        for perc in df['perc'].unique():
            # print(perc)
            for k in df['k'].unique():
                for hmm in df['hmm'].unique():
                    # get subset for this exp
                    subset = df[(df['perc'] == perc) & (df['hmm'] == hmm) & (df['k'] == k)]

                    # fit power law
                    params = [1.0, 1.0, 1.0]
                    popt, pcov = curve_fit(power_law_fit, subset['shots'], subset['nll'], p0=params, maxfev=10000)
                    C, alpha, K = popt
                    # print(f"{perc} -- {k}, {hmm}: C={C}, α={alpha}, K={K}")

                    # store
                    power_law_params[(perc, k, hmm)] = (C, alpha, K)
            #     print()
            # print('-------------------')
        
        # store power law estimates in df
        def estimate_nll(row):
            C, alpha, K = power_law_params[(row['perc'], row['k'], row['hmm'])]
            return power_law_fit(row['shots'], C, alpha, K)
        df["est_nll"] = df.apply(estimate_nll, axis=1)

        # plot
        plot = (
            ggplot(df, aes(x='shots', y='prob', color='perc')) +
            facet_grid("k~hmm") +
            stat_summary(geom="line")
        )
        plot.save(f"in_context_probs{suffix}.pdf")

        plot = (
            ggplot(df) +
            facet_grid("k~hmm") +
            stat_summary(aes(x="shots", y="est_nll", color="perc"), geom="line") +
            stat_summary(aes(x="shots", y="nll", color="perc"), geom="point", size=1.0, stroke=0, alpha=0.4) +
            scale_y_log10() + scale_x_log10()
        )

        plot.save(f"in_context_probs_nll{suffix}.pdf")


def main():
    analyse_folder()


if __name__ == "__main__":
    main()