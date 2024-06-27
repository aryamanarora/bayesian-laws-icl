import numpy as np
import pandas as pd
from plotnine import *
from plotnine.scales import scale_x_log10, scale_y_log10
from tqdm import tqdm
from scipy.optimize import curve_fit
from collections import defaultdict
import scipy.stats as stats

NUM_EXPS = 100

def test(N, p_g, p_A_given_g, p_A_given_b, plot=False):
    # compute
    p_b = 1 - p_g
    p_B_given_g = 1 - p_A_given_g
    p_B_given_b = 1 - p_A_given_b

    # helpers
    def update_probability(prior, observation):
        if observation in ['A', 0]:
            p_g_given_A = p_A_given_g * prior / (p_A_given_g * prior + p_A_given_b * (1 - prior))
            return p_g_given_A
        elif observation in ['B', 1]:
            p_g_given_B = p_B_given_g * prior / (p_B_given_g * prior + p_B_given_b * (1 - prior))
            return p_g_given_B
        else:
            raise ValueError("Invalid observation")


    def compute_probability(prior, observation):
        if observation in ['A', 0]:
            p = p_A_given_g * prior + p_A_given_b * (1 - prior)
            return p
        elif observation in ['B', 1]:
            p = p_B_given_g * prior + p_B_given_b * (1 - prior)
            return p
        else:
            raise ValueError("Invalid observation")
        
    # let's sample! from the good distribution
    data = [(0, p_g, "MC", "p(g)")]
    for _ in tqdm(range(NUM_EXPS)):
        
        # let's use bayes rule to calculate the probability of the sequence
        # being from the good distribution iteratively
        p_g_given_seq = p_g
        p_b_given_seq = p_b
        prob_under_g, prob_under_b = 0.0, 0.0
        for n in range(1, N):
            # sample from the good distribution
            choice = np.random.choice([0, 1], p=[p_A_given_g, p_B_given_g])
            if choice == 0:
                prob_under_g += np.log(p_A_given_g)
                prob_under_b += np.log(p_A_given_b)
            else:
                prob_under_g += np.log(p_B_given_g)
                prob_under_b += np.log(p_B_given_b)

            # data probability
            data.append((n, compute_probability(p_g_given_seq, choice), "MC", "p(d)"))

            # posterior for g and b
            p_g_given_seq_old = p_g_given_seq
            p_g_given_seq = update_probability(p_g_given_seq, choice)
            p_b_given_seq = 1 - p_g_given_seq
            data.append((n, p_g_given_seq, "MC", "p(g)"))

            # posterior in one go
            p_g_given_seq_2 = np.exp(prob_under_g) * p_g / (np.exp(prob_under_g) * p_g + np.exp(prob_under_b) * p_b)
            data.append((n, p_g_given_seq_2, "MC (prob seq)", "p(g)"))

    # BERNOULLI
    print("Bernoulli")
    for n in tqdm(range(0, N)):
        if n != 0:
            probs_under_g = np.array([stats.binom.pmf(count_A, n, p_A_given_g) for count_A in range(0, n + 1)])
            probs_under_b = np.array([stats.binom.pmf(count_A, n, p_A_given_b) for count_A in range(0, n + 1)])
            p_g_given_seq = (probs_under_g * p_g / (probs_under_g * p_g + probs_under_b * p_b))
            p_g_exp = np.sum(p_g_given_seq * probs_under_g)
        else:
            p_g_exp = p_g
        p_b_exp = 1 - p_g_exp
        p_d = (p_A_given_g**2 + p_B_given_g**2) * p_g_exp + (p_A_given_g * p_A_given_b + p_B_given_g * p_B_given_b) * p_b_exp
        data.append((n, p_g_exp, "Bernoulli", "p(g)"))
        data.append((n + 1, p_d, "Bernoulli", "p(d)"))

    # CURVE
    def func(x, g0, gamma, beta):
        res = (2 * gamma - 1) * (gamma - beta) * (gamma**(2*x) * g0) / ((gamma**x) * g0 + (beta**x)*(1 - g0))
        res += 2 * gamma * beta - gamma - beta + 1
        return res
        # return (gamma - beta) * (gamma**x * g0 / (gamma**x * g0 + beta**x * (1 - g0))) + beta

    # fit the curve
    df = pd.DataFrame(data, columns=["n", "p", "method", "metric"])
    df_filtered = df[(df["metric"] == "p(d)") & (df["method"] == "MC")]
    params = [p_g, 0.8, 0.2]
    params, _ = curve_fit(
        func, df_filtered["n"], df_filtered["p"], p0=params, bounds=([0, 0, 0], [np.inf, np.inf, np.inf])
    )
    print(f"p(A|g)={p_A_given_g:.2f}, p(A|b)={p_A_given_b:.2f}: p(g)={params[0]:.5f}, gamma={params[1]:.5f}, beta={params[2]:.5f}")

    # p_0(d) and p_infty(d)
    start = (p_A_given_g**2 + p_B_given_g**2) * p_g + (p_A_given_g * p_A_given_b + p_B_given_g * p_B_given_b) * p_b
    end = (p_A_given_g**2 + p_B_given_g**2)
    print(f"start={start:.5f}, end={end:.5f}")

    # plot
    if plot:
        # add estimates to df
        for n in range(0, N):
            row = (n, func(n, *params), "fit", "p(d)")
            data.append(row)
        df = pd.DataFrame(data, columns=["n", "p", "method", "metric"])
        df_avg = df.groupby(["n", "method", "metric"]).mean().reset_index()
        plot = (
            ggplot(df_avg, aes(x="n", y="p", color="method")) + geom_line()
            + geom_hline(yintercept=start, linetype="dashed", color="red")
            + geom_hline(yintercept=end, linetype="dashed", color="blue")
            + facet_wrap("~metric")
        )
        plot.save(f"test_{p_A_given_g}_{p_A_given_b}.png", dpi=300)
    
    return df, params


def main():
    # max sequence length
    N = 500

    # priors on good and bad distributions
    p_g = 0.5

    # probability of data from good distribution under good and bad distributions
    data = []
    ct = 5
    for p in range(1, 2):
        p_A_given_g = p / ct
        for q in range(2, 3):
            p_A_given_b = q / ct
            df, params = test(N, p_g, p_A_given_g, p_A_given_b, plot=True)
            data.append((p_A_given_g, p_A_given_b, params[0], params[1]))
    
    # plot heatmap
    df = pd.DataFrame(data, columns=["p_A_given_g", "p_A_given_b", "g0", "ratio"])
    plot = (
        ggplot(df, aes(x="p_A_given_g", y="p_A_given_b", fill="ratio")) + geom_tile()
        + scale_fill_gradient(low="white", high="black")
        + geom_text(aes(label="ratio"), size=8)
    )
    plot.save("ratio.png", dpi=300)


if __name__ == "__main__":
    main()