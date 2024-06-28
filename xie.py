"""
Replicate the plots from the Xie et al. paper.
"""

from plotnine import (
    ggplot, aes, geom_line, geom_point, facet_grid, stat_summary,
    theme, element_text
)
from plotnine.scales import scale_x_log10, scale_y_log10
import pandas as pd
import argparse


def plot_data(directory: str):
    dfs = []
    for suffix in ['', '-8', '-12']:
        data = pd.read_csv(directory + suffix + '/in_context_probs.csv')
        data = data[data["sft"].isin(["False"])]
        data = data.drop(columns=["sft"])
        data['k'] = data['k'].astype(str)
        data['layers'] = int(suffix.split('-')[1]) if suffix != '' else 4
        dfs.append(data)
    data = pd.concat(dfs)

    # average over HMMs
    replicated = data.drop(columns=["hmm"]).groupby(['shots', 'k', 'layers']).mean().reset_index()
    plot = (
        ggplot(replicated, aes(x='shots', y='acc', color='k', group='k')) +
        geom_line() + facet_grid('.~layers', labeller="label_both") +
        geom_point()
    )
    plot.save(directory + '/in_context_probs_ginc.png', dpi=300)

    data = data.groupby(['shots', 'k', 'layers', 'hmm']).mean().reset_index()
    plot = (
        ggplot(data, aes(x='shots', y='acc', color='k', group='k')) +
        geom_line() + facet_grid('layers~hmm', labeller="label_both")
    )
    plot.save(directory + '/in_context_probs_ginc2.png', dpi=300)

    data['shots'] += 1
    plot = (
        ggplot(data, aes(x='shots', y='nll', color='k', group='k')) +
        geom_line() + facet_grid('layers~hmm', labeller="label_both") +
        scale_y_log10() + scale_x_log10()
    )
    plot.save(directory + '/in_context_probs_ginc3.png', dpi=300)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--directory', help='Directory containing the data files')
    args = parser.parse_args()

    plot_data(args.directory)


if __name__ == '__main__':
    main()