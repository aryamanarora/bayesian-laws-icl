# def fit_bayesian_law(subset: pd.DataFrame):
#     # prep data
#     subset = subset.sample(frac=1.0)
#     subset['hmm'] = subset['hmm'].astype(int)
#     num_hmms = len(subset['hmm'].unique())

#     # fit power law
#     model = BayesianLawFit(num_hmms)
#     model.to(DEVICE)
#     iterator = tqdm(range(50))
#     patience = 5
#     batch_size = 20
#     history = []

#     optimizer = torch.optim.LBFGS(model.parameters(), lr=1e-1)

#     def closure():
#         optimizer.zero_grad()
#         avg_loss = 0.0
#         for i in range(0, len(subset), batch_size):
#             batch = subset.iloc[i:i+batch_size]
#             shots = torch.tensor(batch['shots'].values, dtype=torch.float32).to(DEVICE)
#             hmm = torch.tensor(batch['hmm'].values, dtype=torch.int32).to(DEVICE)
#             true_nll = torch.tensor(batch['nll'].values, dtype=torch.float32).to(DEVICE)
#             est_nll = model(shots, hmm)
#             loss = ((true_nll - est_nll)**2).sum()
#             loss.backward()
#             avg_loss += loss.item()
#         avg_loss /= len(subset)
#         return avg_loss

#     best_loss = float("inf")
#     best_params = None
#     for _ in iterator:
#         train_loss = optimizer.step(closure)
#         history.append(train_loss)
#         print(train_loss)

#         if train_loss < best_loss:
#             best_loss = train_loss
#             best_params = deepcopy(model.state_dict())

#             result = {
#                 "loss": train_loss,
#                 "gamma_0": model.get_gammas()[0].item(),
#                 "beta_0": model.get_betas()[0].item(),
#                 "K": model.get_K().item(),
#             }
#             result.update({f"p_{i}": p.item() for i, p in enumerate(model.get_prior())})
#             iterator.set_postfix(result)
    
#     model.load_state_dict(best_params)

#     return model