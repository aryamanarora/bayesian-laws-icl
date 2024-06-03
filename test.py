import numpy as np
import pandas as pd
from plotnine import *
from tqdm import tqdm
from scipy.optimize import curve_fit
from collections import defaultdict

N = 100

# priors on good and bad distributions
p_g = 0.5
p_b = 1 - p_g

# probability of data from good distribution under good and bad distributions
p_A_given_g = 0.8
p_A_given_b = 0.2

# probability of data from bad distribution under good and bad distributions
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

# let's calculate the probability of the sequence being from the good distribution
data_est_new = [(0, p_g)]
# buffer = defaultdict(float)
# buffer[p_g] = 1.0
# for n in range(1, N):
#     new_buffer = defaultdict(float)
#     for p_g_cur, prob in buffer.items():
#         p_g_given_A = update_probability(p_g_cur, 'A')
#         p_g_given_B = update_probability(p_g_cur, 'B')
#         new_buffer[p_g_given_A] += prob * p_A_given_g
#         new_buffer[p_g_given_B] += prob * p_B_given_g
#     buffer = new_buffer
#     print(n, len(buffer))
#     data_est_new.append((n, sum([prob * p_g_cur for p_g_cur, prob in buffer.items()])))

for n in range(1, N):
    poss = []
    for A in range(n + 1):
        B = n - A
        prob = np.log(p_A_given_g) * A + np.log(p_B_given_g) * B
        try:
            num_perms = np.log(np.math.factorial(n)) - np.log(np.math.factorial(A)) - np.log(np.math.factorial(B))
        except:
            num_perms = 0
        p_g_given_seq = p_g
        for _ in range(A):
            p_g_given_seq = update_probability(p_g_given_seq, 'A')
        for _ in range(B):
            p_g_given_seq = update_probability(p_g_given_seq, 'B')
        # poss.append(prob * p_g_given_seq * num_perms)
        log_res = prob + np.log(p_g_given_seq) + num_perms
        print(log_res)
        poss.append(np.exp(log_res))
    data_est_new.append((n, sum(poss)))


# let's sample! from the good distribution
data = [(0, p_g)]
probs_under_g, probs_under_b = [], []
for _ in tqdm(range(100)):
    
    # let's use bayes rule to calculate the probability of the sequence
    # being from the good distribution iteratively
    p_g_given_seq = p_g
    p_b_given_seq = p_b
    for n in range(1, N):
        choice = np.random.choice([0, 1], p=[p_A_given_g, p_B_given_g])
        p_g_given_seq_old = p_g_given_seq
        p_g_given_seq = update_probability(p_g_given_seq, choice)
        p_b_given_seq = 1 - p_g_given_seq
        data.append((n, p_g_given_seq))

# predicted
print("Avg probs under good distribution: ", np.mean(probs_under_g))
print("Avg probs under bad distribution: ", np.mean(probs_under_b))
data_est = []

p_d_given_g = p_A_given_g**2 + p_B_given_g**2
p_d_given_b = p_A_given_g*p_A_given_b + p_B_given_g*p_B_given_b
print(f"p_d_given_g: {p_d_given_g:.8f}, p_d_given_b: {p_d_given_b:.8f}")

ratio = (2 * p_A_given_b**2 - 2 * p_A_given_b + 1) / (2 * p_A_given_g**2 - 2 * p_A_given_g + 1)
for n in range(N):
    p = 1 - (1 / p_g) * (p_g - 1) * (ratio)**n
    p = 1 / p
    data_est.append((n, p))

def bayesian_fit(n, g0, gamma, beta):
    p = 1 - (1 / g0) * (g0 - 1) * (beta / gamma)**n
    p = 1 / p
    return p

# fit model on df
df = pd.DataFrame(data, columns=['n', 'p'])
params = [0.5, 0.8, 0.7]
params, _ = curve_fit(bayesian_fit, df['n'], df['p'], p0=params)
data_fit = []
for n in range(N):
    p = bayesian_fit(n, *params)
    data_fit.append((n, p))
print(f"Estimated parameters: g0={params[0]:.8f}, gamma={params[1]:.8f}, beta={params[2]:.8f}")
print(f"gamma/beta: {params[1]/params[2]:.8f}")
print(f"true gamma/beta: {1/ratio:.8f}")
print(f"log b/log a: {np.log(1/ratio) / np.log(params[1]/params[2]):.8f}")

# let's plot the probability of the sequence being from the good distribution
plot = ggplot(df, aes(x='n', y='p')) + geom_point(alpha=0.3, stroke=0)
plot.save('test.png')

df_avg = df.groupby('n').mean().reset_index()
df_est = pd.DataFrame(data_est, columns=['n', 'p'])
df_est_new = pd.DataFrame(data_est_new, columns=['n', 'p'])
df_fit = pd.DataFrame(data_fit, columns=['n', 'p'])
plot = (
    ggplot() + geom_line(df_avg, aes(x='n', y='p')) +
    geom_line(df_est, aes(x='n', y='p'), color='red') +
    geom_line(df_fit, aes(x='n', y='p'), color='blue') +
    geom_line(df_est_new, aes(x='n', y='p'), color='green')
)
plot.save('test_avg.png')
