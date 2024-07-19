import matplotlib.pyplot as plt
import random
import numpy as np

samples = 500
length = 1000

p_A_given_g = 0.51
p_A_given_b = 0.49
p_B_given_g = 1 - p_A_given_g
p_B_given_b = 1 - p_A_given_b

p_A = 0.51
p_B = 1 - p_A

p_g = 0.5
p_b = 1 - p_g

# sample only As
p_gs_mean = np.zeros(length + 1)
for _ in range(samples):
    p_gs = [p_g]
    p_g_given_D = p_g
    for i in range(1, length + 1):
        symbol = np.random.choice(['A', 'B'], p=[p_A, p_B])
        if symbol == 'A':
            p_d = p_A_given_g * p_g_given_D + p_A_given_b * (1 - p_g_given_D)
            p_g_given_D = (p_A_given_g * p_g_given_D) / p_d
        else:
            p_d = p_B_given_g * p_g_given_D + p_B_given_b * (1 - p_g_given_D)
            p_g_given_D = (p_B_given_g * p_g_given_D) / p_d
        p_gs.append(p_g_given_D)
    p_gs_mean += np.array(p_gs)
p_gs_mean /= samples

p_g_predicted = [p_g]
for i in range(1, length + 1):
    # predicted (non-recurrent)
    p_d_pred = (p_A_given_g**(i * p_A) * p_B_given_g**(i * p_B)) * p_g + (p_A_given_b**(i * p_A) * p_B_given_b**(i * p_B)) * p_b
    p = (p_A_given_g**(i * p_A) * p_B_given_g**(i * p_B)) * p_g / p_d_pred
    p_g_predicted.append(p)

# plot
plt.plot(p_gs_mean, label='real')
plt.plot(p_g_predicted, label='predicted')
plt.legend()
plt.savefig('bernoulli.png')
