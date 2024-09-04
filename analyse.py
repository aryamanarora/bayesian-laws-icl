"""
Code for fitting Bayesian laws to the ICL behaviour from the
training/SFT/RLHF experiments.
"""

from plotnine import (
    ggplot, aes, geom_line, geom_point, facet_grid, stat_summary,
    theme, element_text, theme_bw
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
import json
from tqdm import tqdm
from copy import deepcopy
from collections import defaultdict


DEVICE = "cpu"


class PowerLawFit(torch.nn.Module):
    def __init__(self):
        super(PowerLawFit, self).__init__()
        self.C = torch.nn.Parameter(torch.tensor(1.0))
        self.alpha = torch.nn.Parameter(torch.tensor(1.0))
        self.K = torch.nn.Parameter(torch.tensor(1.0))

    def get_C(self):
        return self.C.exp()
    
    def get_alpha(self):
        return self.alpha.exp()
    
    def get_K(self):
        return self.K.exp()
    
    def get_params(self):
        return {
            "C": self.get_C(),
            "alpha": self.get_alpha(),
            "K": self.get_K(),
        }
    
    def forward(self, shots):
        C = self.get_C()
        alpha = self.get_alpha()
        K = self.get_K()
        if type(shots) == int:
            shots = torch.tensor([shots], dtype=torch.float32).to(DEVICE)
        return C * (shots).pow(-alpha) + K

    def estimate_nll(self, max_shots: int, hmm: int=None):
        shots = torch.tensor(range(1, max_shots), dtype=torch.float32).to(DEVICE)
        return self(shots)


class BoundedPowerLawFit(torch.nn.Module):
    def __init__(self):
        super(BoundedPowerLawFit, self).__init__()
        self.C = torch.nn.Parameter(torch.tensor(1.0))
        self.alpha = torch.nn.Parameter(torch.tensor(1.0))
        self.K = torch.nn.Parameter(torch.tensor(1.0))
        self.n_c = torch.nn.Parameter(torch.tensor(1.0))
    
    def get_C(self):
        return self.C.exp()

    def get_alpha(self):
        return self.alpha.exp()
    
    def get_K(self):
        return self.K.exp()
    
    def get_n_c(self):
        return self.n_c.exp()

    def get_params(self):
        return {
            "C": self.get_C().item(),
            "alpha": self.get_alpha().item(),
            "K": self.get_K().item(),
            "n_c": self.get_n_c().item()
        }
    
    def forward(self, shots):
        C = self.get_C()
        alpha = self.get_alpha()
        K = self.get_K()
        n_c = self.get_n_c()
        if type(shots) == int:
            shots = torch.tensor([shots], dtype=torch.float32).to(DEVICE)
        return C * (1 + shots / n_c).pow(-alpha) + K

    def estimate_nll(self, max_shots: int, hmm: int=None):
        shots = torch.tensor(range(1, max_shots), dtype=torch.float32).to(DEVICE)
        return self(shots)


class LogisticLawFit(torch.nn.Module):
    def __init__(self):
        super(LogisticLawFit, self).__init__()
        self.C = torch.nn.Parameter(torch.tensor(1.0))
        self.alpha = torch.nn.Parameter(torch.tensor(1.0))
        self.K = torch.nn.Parameter(torch.tensor(1.0))
        self.n_c = torch.nn.Parameter(torch.tensor(1.0))
    
    def get_C(self):
        return torch.sigmoid(self.C)
    
    def get_alpha(self):
        return self.alpha.exp()
    
    def get_K(self):
        return torch.sigmoid(self.K)
    
    def get_n_c(self):
        return self.n_c
    
    def get_params(self):
        return {
            "C": self.get_C().item(),
            "alpha": self.get_alpha().item(),
            "K": self.get_K().item(),
            "n_c": self.get_n_c().item()
        }
    
    def forward(self, shots):
        C = self.get_C()
        alpha = self.get_alpha()
        K = self.get_K()
        n_c = self.get_n_c()
        if type(shots) == int:
            shots = torch.tensor([shots], dtype=torch.float32).to(DEVICE)
        res = -((C - K) / (1 + (-alpha * (shots.log() - n_c)).exp()) + K).log()
        return res

    def estimate_nll(self, max_shots: int, hmm: int=None):
        shots = torch.tensor(range(1, max_shots), dtype=torch.float32).to(DEVICE)
        return self(shots)


class BayesianLawFit(torch.nn.Module):
    def __init__(self, num_hmms, sft_amounts=None, do_log_shots=False):
        super(BayesianLawFit, self).__init__()
        self.do_log_shots = do_log_shots
        self.num_hmms = num_hmms
        self.masks = torch.eye(num_hmms, dtype=torch.bool).to(DEVICE)
        if sft_amounts is not None:
            self.sft_amount_d = {sft: i for i, sft in enumerate(sft_amounts)}
            self.sft_amounts = np.vectorize(self.sft_amount_d.get)
        else:
            sft_amounts = [0]
        self.priors = torch.nn.Parameter(torch.zeros(len(sft_amounts), num_hmms))
        self.P = torch.nn.Parameter((torch.eye(num_hmms, num_hmms) * 2) - 1)
        self.K = torch.nn.Parameter(torch.zeros(len(sft_amounts)))
    
    def get_prior(self, sft_amount=None):
        return torch.nn.functional.softmax(self.priors[sft_amount], dim=-1)

    def get_P(self):
        return torch.sigmoid(self.P)
    
    def get_K(self, sft_amount=None):
        return torch.exp(self.K[sft_amount])

    def get_params(self):
        params = {
            "priors": self.get_prior().tolist(),
            "K": self.get_K().tolist(),
        }
        P = self.get_P()
        for i, row in enumerate(P):
            params[f"P_{i}"] = row.tolist()
        return params

    def get_p_under_dist(self, hmm):
        P = self.get_P().log()
        p_under_dist = P[hmm]
        return p_under_dist

    def forward(self, shots, hmm, sft_amount=[0], add_metrics=False):
        priors = self.get_prior(sft_amount).log()
        K = self.get_K(sft_amount)
        if self.do_log_shots:
            shots = shots.log()
        shots = shots * K
        if isinstance(shots, torch.Tensor):
            shots = shots.unsqueeze(-1)
        # print("hmm:", hmm)
        # print("K:", self.K.item())
        # print("p(hmm):", priors.exp().tolist())
        p_under_dist = self.get_p_under_dist(hmm)
        # print("p(d | hmm):", p_under_dist.exp().tolist())
        p_seq_under_dist = p_under_dist * shots
        # print("p(D | hmm):", p_seq_under_dist.exp().tolist())
        posteriors = torch.nn.functional.softmax(priors + p_seq_under_dist, dim=-1) # already in log space
        # print("p(hmm | D):", posteriors.tolist())
        p_data = (posteriors * p_under_dist.exp()).sum(dim=-1)
        # print("p(d | D, hmm):", (posteriors * p_under_dist.exp()).tolist())
        # print("p(d | D):", p_data.item())
        est_nll = -torch.log(p_data)
        # print("NLL:", nll, est_nll.item())
        # input()
        if add_metrics:
            return {
                "nll": est_nll,
                "posteriors": posteriors,
            }
        return est_nll

    def estimate_nll(self, max_shots: int, hmm: int=None):
        shots = torch.tensor(range(1, max_shots), dtype=torch.float32).to(DEVICE)
        hmm = torch.zeros_like(shots, dtype=torch.int32).fill_(hmm).to(DEVICE)
        return self(shots, hmm)


class BayesianLawSamplingFit(BayesianLawFit):
    def __init__(self, num_hmms, sft_amounts=None, do_log_shots=False):
        super(BayesianLawSamplingFit, self).__init__(num_hmms, sft_amounts, do_log_shots)
        self.gammas = torch.nn.Parameter(torch.zeros(num_hmms) + 1)
        self.betas = torch.nn.Parameter(torch.zeros(num_hmms) - 1)
    
    def get_gammas(self):
        return torch.sigmoid(self.gammas)

    def get_betas(self):
        return torch.sigmoid(self.betas) * torch.sigmoid(self.gammas)
    
    def get_params(self):
        return {
            "priors": self.get_prior().tolist(),
            "gammas": self.get_gammas().tolist(),
            "betas": self.get_betas().tolist(),
            "K": self.get_K().tolist(),
        }

    def get_p_under_dist(self, hmm):
        gammas = self.get_gammas().log()
        betas = self.get_betas().log()
        p_under_dist = torch.where(self.masks[hmm], gammas, betas)
        return p_under_dist


class BayesianLawScoringFit(BayesianLawFit):
    def __init__(self, num_hmms, sft_amounts=None, do_log_shots=False):
        super(BayesianLawScoringFit, self).__init__(num_hmms, sft_amounts, do_log_shots)
        self.gammas = torch.nn.Parameter(torch.zeros(num_hmms) + 1)
        self.betas = torch.nn.Parameter(torch.zeros(num_hmms) - 1)

    def get_gammas(self):
        return torch.sigmoid(self.gammas)

    def get_betas(self):
        return torch.sigmoid(self.betas) * torch.sigmoid(self.gammas)
    
    def get_K(self, sft_amount=None):
        return torch.exp(self.K[sft_amount])

    def get_params(self):
        return {
            "priors": self.get_prior().tolist(),
            "gammas": self.get_gammas().tolist(),
            "betas": self.get_betas().tolist(),
            "K": self.get_K().tolist(),
        }

    def get_p_under_dist(self, hmm):
        gammas = self.get_gammas().log()
        betas = self.get_betas().log()
        gammas = gammas[hmm].unsqueeze(1).repeat(1, self.num_hmms)
        betas = betas[hmm].unsqueeze(1).repeat(1, self.num_hmms)
        p_under_dist = self.masks[hmm] * (gammas - betas) + betas
        return p_under_dist
    

power_law_mapping = {
    "power": PowerLawFit,
    "bounded": BoundedPowerLawFit,
    "logistic": LogisticLawFit,
}


bayesian_law_mapping = {
    "original": BayesianLawFit,
    "sampling": BayesianLawSamplingFit,
    "scoring": BayesianLawScoringFit,
}


def fit_bayesian_law(
    subset: pd.DataFrame,
    sft_amount: bool=False,
    type: str="original",
    quiet: bool=False,
    patience: int=5,
    batch_size: int=20,
    epochs: int=50,
    lr: float=5e-2,
    num_hmms: int=None,
    do_log_shots: bool=False,
):
    # fit power law
    sft_amounts = None if not sft_amount else list(subset['sft_amount'].unique())
    law_type = bayesian_law_mapping[type]
    num_hmms = num_hmms or len(subset['hmm'].unique())
    model = law_type(num_hmms, sft_amounts, do_log_shots)
    model.to(DEVICE)
    iterator = tqdm(range(epochs)) if not quiet else range(epochs)
    history = []
    subset = subset.sample(frac=1.0)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for _ in iterator:
        avg_loss = 0.0
        loss = 0.0
        for i in range(0, len(subset), batch_size):
            optimizer.zero_grad()
            batch = subset.iloc[i:i+batch_size]
            shots = torch.tensor(batch['shots'].values, dtype=torch.float32).to(DEVICE)
            hmm = torch.tensor(list(map(int, batch['hmm'].values)), dtype=torch.int32).to(DEVICE)
            true_nll = torch.tensor(batch['nll'].values, dtype=torch.float32).to(DEVICE)
            if sft_amount:
                sft_amounts = torch.tensor(batch['sft_amount'].values, dtype=int).to(DEVICE)
                est_nll = model(shots, hmm, sft_amounts)
            else:
                est_nll = model(shots, hmm)
            loss = ((true_nll.log() - est_nll.log())**2).sum()
            loss.backward()
            optimizer.step()
            avg_loss += loss.item()

        # print
        avg_loss /= len(subset)
        history.append(avg_loss)
        if not quiet:
            result = {
                "loss": avg_loss,
            }
            iterator.set_postfix(result)

        # early stopping
        if len(history) > patience and all([math.isclose(history[-1], x, rel_tol=5e-3) for x in history[-patience:]]):
            break
    
    return model


def fit_power_law(
    subset: pd.DataFrame,
    type: str="power",
    quiet: bool=False,
    patience: int=5,
    batch_size: int=5,
    epochs: int=50,
    lr: float=5e-2,
):
    # fit power law
    model = power_law_mapping[type]()
    model.to(DEVICE)
    iterator = tqdm(range(epochs)) if not quiet else range(epochs)
    history = []
    subset = subset.sample(frac=1.0)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for _ in iterator:
        avg_loss = 0.0
        loss = 0.0
        for i in range(0, len(subset), batch_size):
            optimizer.zero_grad()
            batch = subset.iloc[i:i+batch_size]
            shots = torch.tensor(batch['shots'].values, dtype=torch.float32).to(DEVICE)
            true_nll = torch.tensor(batch['nll'].values, dtype=torch.float32).to(DEVICE)
            est_nll = model(shots)
            loss = ((true_nll.log() - est_nll.log())**2).sum()
            loss.backward()
            optimizer.step()
            avg_loss += loss.item()

        # print
        avg_loss /= len(subset)
        history.append(avg_loss)
        if not quiet:
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
        if len(history) > patience and all([math.isclose(history[-1], x, rel_tol=5e-3) for x in history[-patience:]]):
            break
    
    return model