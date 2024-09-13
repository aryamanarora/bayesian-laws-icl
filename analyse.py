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
import torch.nn.functional as F
import json
from tqdm import tqdm
from copy import deepcopy
from collections import defaultdict


DEVICE = "cpu"


class PowerLawFit(torch.nn.Module):
    def __init__(self):
        super(PowerLawFit, self).__init__()
        self.C = torch.nn.Parameter(torch.tensor(0.0))
        self.alpha = torch.nn.Parameter(torch.tensor(1.0))
        self.K = torch.nn.Parameter(torch.tensor(0.0))

    def get_C(self):
        return self.C
    
    def get_alpha(self):
        return F.softplus(self.alpha)
    
    def get_K(self):
        return self.K
    
    def get_params(self):
        return {
            "C": self.get_C().exp().item(),
            "alpha": self.get_alpha().item(),
            "K": self.get_K().item(),
        }
    
    def forward(self, shots, hmm=None):
        C = self.get_C()
        alpha = self.get_alpha()
        K = self.get_K()
        if type(shots) == int:
            shots = torch.tensor([shots], dtype=torch.float32).to(DEVICE)
        first_term_log = C - alpha * shots.log()
        est_nll = first_term_log.exp() + K
        return est_nll

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
        return self.C # can be anything, stored in log

    def get_alpha(self):
        return F.softplus(self.alpha) # must be positive
    
    def get_K(self):
        return F.softplus(self.K) # must be positive
    
    def get_n_c(self):
        return F.softplus(self.n_c) # must be positive

    def get_params(self):
        return {
            "C": self.get_C().exp().item(),
            "alpha": self.get_alpha().item(),
            "K": self.get_K().item(),
            "n_c": self.get_n_c().item()
        }
    
    def forward(self, shots, hmm=None):
        C = self.get_C()
        alpha = self.get_alpha()
        K = self.get_K()
        n_c = self.get_n_c()
        if type(shots) == int:
            shots = torch.tensor([shots], dtype=torch.float32).to(DEVICE)
        first_term_log = C - alpha * (1 + shots / n_c).log()
        est_nll = first_term_log.exp() + K
        return est_nll

    def estimate_nll(self, max_shots: int, hmm: int=None):
        shots = torch.tensor(range(1, max_shots), dtype=torch.float32).to(DEVICE)
        return self(shots)


class LogisticLawFit(torch.nn.Module):
    def __init__(self):
        super(LogisticLawFit, self).__init__()
        self.C = torch.nn.Parameter(torch.tensor(1.0))
        self.L = torch.nn.Parameter(torch.tensor(1.0))
        self.K = torch.nn.Parameter(torch.tensor(1.0))
        self.x_0 = torch.nn.Parameter(torch.tensor(1.0))
    
    def get_C(self):
        return self.C
    
    def get_L(self):
        return self.L
    
    def get_K(self):
        return F.softplus(self.K)
    
    def get_x_0(self):
        return F.softplus(self.x_0)
    
    def get_params(self):
        return {
            "C": self.get_C().item(),
            "L": self.get_L().exp().item(),
            "K": self.get_K().item(),
            "x_0": self.get_x_0().item(),
        }
    
    def forward(self, shots, hmm=None):
        C = self.get_C()
        L = self.get_L()
        K = self.get_K()
        x_0 = self.get_x_0()
        if type(shots) == int:
            shots = torch.tensor([shots], dtype=torch.float32).to(DEVICE)

        # to prevent division by zero or extremely small x_0
        # x_0 = torch.clamp(x_0, min=1e-6)
        
        # clamp ratio to prevent extremely large or small values
        # ratio = torch.clamp(shots / x_0, min=1e-6, max=1e6)

        # compute (shots / x0)^K safely
        # pow = torch.clamp((K * ratio.log()).exp(), min=1e-6, max=1e6)
        pow = (shots / x_0).pow(K)

        first_term_log = L - torch.log1p(pow)
        est_nll = first_term_log.exp() + C
        # if est_nll.isnan().any():
        #     print("NaN in forward pass")
        #     print("C", C)
        #     print("L", L)
        #     print("K", K)
        #     print("x_0", x_0)
        #     raise ValueError("NaN in forward pass")
        return est_nll

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
        self.P = torch.nn.Parameter(torch.eye(num_hmms) + 0.5)
        self.K = torch.nn.Parameter(torch.ones(len(sft_amounts)))
    
    def get_prior(self, sft_amount=None):
        return F.log_softmax(self.priors[sft_amount], dim=-1)

    def get_P(self):
        return -F.softplus(self.P)
    
    def get_K(self, sft_amount=None):
        return F.softplus(self.K[sft_amount])

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
        P = self.get_P()
        p_under_dist = P[hmm]
        return p_under_dist

    def forward(self, shots, hmm, sft_amount=[0], add_metrics=False):
        # get params
        priors = self.get_prior(sft_amount)
        K = self.get_K(sft_amount)
        if self.do_log_shots:
            shots = shots.log()
        if isinstance(shots, torch.Tensor):
            shots = shots.unsqueeze(-1)
        # compute entirely in log space
        p_under_dist = self.get_p_under_dist(hmm)
        denominator = priors + p_under_dist * (shots * K)
        numerator = priors + p_under_dist * (shots * K + 1)
        est_nll = -(torch.logsumexp(numerator, dim=-1) - torch.logsumexp(denominator, dim=-1))
        return est_nll

    def estimate_nll(self, max_shots: int, hmm: int=None):
        shots = torch.tensor(range(1, max_shots), dtype=torch.float32).to(DEVICE)
        hmm = torch.zeros_like(shots, dtype=torch.int32).fill_(hmm).to(DEVICE)
        return self(shots, hmm)


class BayesianLawSamplingFit(BayesianLawFit):
    def __init__(self, num_hmms, sft_amounts=None, do_log_shots=False):
        super(BayesianLawSamplingFit, self).__init__(num_hmms, sft_amounts, do_log_shots)
        self.gammas = torch.nn.Parameter(torch.ones(num_hmms))
        self.betas = torch.nn.Parameter(torch.ones(num_hmms))
    
    def get_gammas(self):
        return -F.softplus(self.gammas)

    def get_betas(self):
        return -F.softplus(self.betas) - F.softplus(self.gammas)
    
    def get_params(self):
        return {
            "priors": self.get_prior().tolist(),
            "gammas": self.get_gammas().tolist(),
            "betas": self.get_betas().tolist(),
            "K": self.get_K().tolist(),
        }

    def get_p_under_dist(self, hmm):
        gammas = self.get_gammas()
        betas = self.get_betas()
        p_under_dist = torch.where(self.masks[hmm], gammas, betas)
        return p_under_dist


class BayesianLawScoringFit(BayesianLawFit):
    def __init__(self, num_hmms, sft_amounts=None, do_log_shots=False):
        super(BayesianLawScoringFit, self).__init__(num_hmms, sft_amounts, do_log_shots)
        self.gammas = torch.nn.Parameter(torch.ones(num_hmms))
        self.betas = torch.nn.Parameter(torch.ones(num_hmms))
    
    def get_gammas(self):
        return -F.softplus(self.gammas)

    def get_betas(self):
        return -F.softplus(self.betas) # - F.softplus(self.gammas)

    def get_params(self):
        return {
            "priors": self.get_prior().tolist(),
            "gammas": self.get_gammas().tolist(),
            "betas": self.get_betas().tolist(),
            "K": self.get_K().tolist(),
        }

    def get_p_under_dist(self, hmm):
        gammas = self.get_gammas()
        betas = self.get_betas()
        m = (self.masks[hmm] == 1).squeeze(1)
        g = gammas[hmm].unsqueeze(-1).expand(-1, self.num_hmms)
        b = betas[hmm].unsqueeze(-1).expand(-1, self.num_hmms)
        p_under_dist = torch.where(m, g, b)
        return p_under_dist
    

power_law_mapping = {
    "power": PowerLawFit,
    "bounded": BoundedPowerLawFit,
    # "logistic": LogisticLawFit,
}


bayesian_law_mapping = {
    "original": BayesianLawFit,
    "sampling": BayesianLawSamplingFit,
    "scoring": BayesianLawScoringFit,
}


huber_loss = torch.nn.HuberLoss(delta=1.0)


def compute_loss(
    true_nll: torch.Tensor,
    est_nll: torch.Tensor,
    mode: str="mse_log",
):
    loss = None
    if mode == "mse_log":
        loss = ((true_nll.log() - est_nll.log())**2).sum()
    elif mode == "mse":
        loss = ((true_nll - est_nll)**2).sum()
    elif mode == "mse_prob":
        loss = (((-true_nll).exp() - (-est_nll).exp())**2).sum()
    elif mode == "huber":
        loss = huber_loss(true_nll, est_nll, reduction="sum")
    elif mode == "huber_log":
        loss = huber_loss(true_nll.log(), est_nll.log(), reduction="sum")
    return loss


def fit_law(
    # model: torch.nn.Module,
    law: str,
    subset: pd.DataFrame,
    quiet: bool=False,
    patience: int=5,
    batch_size: int=20,
    epochs: int=50,
    lr: float=5e-2,
    num_hmms: int | None=None,
    sft_amount: bool=False,
    mode: str="adam",
    do_log_shots: bool=False,
    loss_mode: str="mse_log",
):
    # set up model
    if law in power_law_mapping.keys():
        # power law
        model = power_law_mapping[law]()
        model.to(DEVICE)
    elif law in bayesian_law_mapping.keys():
        # bayesian law
        sft_amounts = None if not sft_amount else list(subset['sft_amount'].unique())
        model = bayesian_law_mapping[law](num_hmms, sft_amounts, do_log_shots)
        model.to(DEVICE)

    # shuffle data
    subset = subset.sample(frac=1.0)
    
    if mode == "adam":
        # set up optim
        iterator = tqdm(range(epochs)) if not quiet else range(epochs)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        history = []

        # optimise
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
                loss = compute_loss(true_nll, est_nll, loss_mode)
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
    elif mode == "lbfgs":
        # set up optim
        optimizer = torch.optim.LBFGS(
            model.parameters(),
            history_size=100,
            max_iter=100,
            line_search_fn="strong_wolfe",
        )

        # each optimisation step
        def closure():
            optimizer.zero_grad()
            shots = torch.tensor(subset['shots'].values, dtype=torch.float32).to(DEVICE)
            hmm = torch.tensor(list(map(int, subset['hmm'].values)), dtype=torch.int32).to(DEVICE)
            true_nll = torch.tensor(subset['nll'].values, dtype=torch.float32).to(DEVICE)
            if sft_amount:
                sft_amounts = torch.tensor(subset['sft_amount'].values, dtype=int).to(DEVICE)
                est_nll = model(shots, hmm, sft_amounts)
            else:
                est_nll = model(shots, hmm)
            loss = compute_loss(true_nll, est_nll, loss_mode)
            loss.backward()
            return loss
        
        # now optimise
        states = []
        history = []
        min_loss = float("inf")
        iterator = tqdm(range(epochs)) if not quiet else range(epochs)
        for _ in iterator:
            subset = subset.sample(frac=1.0) # shuffle
            loss = optimizer.step(closure)
            history.append(loss.item() if not loss.isnan().any() else float("inf"))
            states.append(deepcopy(model.state_dict()))
            if not quiet:
                min_loss = min(min_loss, loss.item() / len(subset))
                iterator.set_postfix({"loss": loss.item() / len(subset), "min_loss": min_loss})
            if loss.isnan().any():
                if not quiet:
                    print("NaN loss, resetting model...")
                # reset model
                model.load_state_dict(states[0])
                optimizer = torch.optim.LBFGS(
                    model.parameters(),
                    history_size=20,
                    max_iter=20,
                    line_search_fn="strong_wolfe",
                )

        # pick state with lowest loss
        best_state = states[np.argmin(history)]
        model.load_state_dict(best_state)
    else:
        raise ValueError("Invalid mode.")

    return model


def eval_fit(
    law: str,
    model: torch.nn.Module,
    shots: torch.Tensor,
    hmms: torch.Tensor,
    true_nll: torch.Tensor,
    true_prob: torch.Tensor,
    hmm: int,
    do_log_shots: bool=False,
):
    est_nll = model(shots, hmms)
    est_prob = (-est_nll).exp()
    rmse = ((true_nll - est_nll)**2).mean()**0.5
    nrmse = rmse / (true_nll.mean())
    log_rmse = ((true_nll.log() - est_nll.log())**2).mean()**0.5
    log_nrmse = rmse / (true_nll.log().mean())
    rmse_prob = ((true_prob - est_prob)**2).mean()**0.5
    nrmse_prob = rmse_prob / (true_prob.mean())
    params = {
        "hmm": hmm,
        "law": ("bayesian_" if law in bayesian_law_mapping else "") + (law if not do_log_shots else f"{law} (log)"),
        "rmse": rmse.item(),
        "nrmse": nrmse.item(),
        "log_rmse": log_rmse.item(),
        "log_nrmse": log_nrmse.item(),
        "rmse_prob": rmse_prob.item(),
        "nrmse_prob": nrmse_prob.item(),
    }
    return params


def compute_all_fits(
    subset: pd.DataFrame,
    max_shots: float=1.0,
    quiet: bool=False,
    patience: int=5,
    epochs: int=50,
    lr: float=5e-2,
    num_hmms: int=None,
    i: int=0,
    metadata: dict={},
    mode: str="adam",
    loss_mode: str="mse_log",
) -> tuple:
    """
    Compute all of the fits for a given subset of the data.

    Args:
        subset (pd.DataFrame): The subset of the data to fit.
        max_shots (float): The maximum shots to train on, as a fraction of the data.
        quiet (bool): Whether to be quiet.
        patience (int): The patience for early stopping.
        epochs (int): The number of epochs to train for.
        lr (float): The learning rate.
        num_hmms (int): The number of subdistributions.
        i (int): The index of the run, for setting seed.
        metadata (dict): The metadata to add to the results.
    """

    # set up extrapolation test if needed
    max_shots *= subset["shots"].max()
    models = {}
    all_params = []
    
    # fit bayesian law
    torch.manual_seed(42 + i)
    np.random.seed(42 + i)
    for do_log_shots in [True, False]:
        for law in bayesian_law_mapping.keys():
            law_name = law if not do_log_shots else f"{law} (log)"
            if not quiet:
                print(f"Fitting {law_name}...")
            bayesian_model = fit_law(
                law=law, subset=subset[subset["shots"] <= max_shots], sft_amount=False,
                quiet=quiet, num_hmms=num_hmms, patience=patience, epochs=epochs,
                lr=lr, do_log_shots=do_log_shots, mode=mode, loss_mode=loss_mode,
            )
            models[law_name] = bayesian_model

    # per-hmm fits
    for hmm in range(len(subset["hmm"].unique())):
        subset_hmm = subset[subset['hmm'] == hmm]
        subset_hmm_extrapolate = subset_hmm[subset_hmm["shots"] > max_shots]
        if len(subset_hmm_extrapolate) == 0:
            subset_hmm_extrapolate = subset_hmm
        shots = torch.tensor(subset_hmm_extrapolate['shots'].values, dtype=torch.float32).to(DEVICE)
        hmms = torch.tensor(list(map(int, subset_hmm_extrapolate['hmm'].values)), dtype=torch.int32).to(DEVICE)
        true_nll = torch.tensor(subset_hmm_extrapolate['nll'].values, dtype=torch.float32).to(DEVICE)
        true_prob = (-true_nll).exp()
        
        # bayesian
        for do_log_shots in [True, False]:
            for law in bayesian_law_mapping.keys():
                bayesian_model = models[law if not do_log_shots else f"{law} (log)"]
                more = bayesian_model.get_params()

                # eval
                params = eval_fit(
                    law=law, model=bayesian_model, shots=shots, hmms=hmms,
                    true_nll=true_nll, true_prob=true_prob, hmm=hmm,
                    do_log_shots=do_log_shots,
                )
                for key in more:
                    if key == "priors": params[key] = more[key][0][0][hmm]
                    elif key == "K": params[key] = more[key][0][0]
                    else: params[key] = more[key][hmm]
                params.update(metadata)
                all_params.append(params)

        # others
        for law in power_law_mapping.keys():
            # fit
            torch.manual_seed(42 + i)
            np.random.seed(42 + i)
            model = fit_law(
                law=law, subset=subset_hmm[subset_hmm["shots"] <= max_shots], quiet=quiet,
                patience=patience, epochs=epochs, lr=lr, mode=mode, loss_mode=loss_mode,
            )
            models[(law, hmm)] = model

            # eval
            params = eval_fit(
                law=law, model=model, shots=shots, hmms=hmms,
                true_nll=true_nll, true_prob=true_prob, hmm=hmm,
                do_log_shots=False,
            )
            params.update(model.get_params())
            params.update(metadata)
            all_params.append(params)

    # done
    return all_params, models