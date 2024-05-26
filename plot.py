from plotnine import ggplot, aes, geom_line, facet_grid, stat_summary
from plotnine.scales import scale_y_log10, scale_x_log10
import pandas as pd
import os

# load data
for suffix in ['', '-8', '-12']:
    folders = ['1perc', '5perc', '10perc', '20perc', '50perc', '100perc', 'infperc']
    dfs = []
    for folder in folders:
        file = f"logs/{folder}{suffix}/in_context_probs.csv"
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
            df_base['perc'] = "0"
            dfs.append(df_base)

    df = pd.concat(dfs)
    print(df)

    # plot
    plot = (
        ggplot(df, aes(x='shots', y='prob', color='perc')) +
        facet_grid("k~hmm") +
        stat_summary(geom="line")
    )
    plot.save(f"in_context_probs{suffix}.pdf")

    plot = (
        ggplot(df, aes(x="shots", y="nll", color="perc")) +
        facet_grid("k~hmm") +
        stat_summary(geom="line") +
        scale_y_log10() + scale_x_log10()
    )
    plot.save(f"in_context_probs_nll{suffix}.pdf")