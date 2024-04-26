import numpy as np
from tqdm import tqdm
from torch.utils.data import Dataset


class HMM:
    num_hidden_states: int
    num_emissions: int
    state_transition_probs: np.ndarray # shape: (num_hidden_states, num_hidden_states)
    emission_probs: np.ndarray # shape: (num_hidden_states, num_emissions)
    start_state_probs: np.ndarray # shape: (num_hidden_states,)

    def __init__(self, num_hidden_states: int, num_emissions: int):
        self.num_hidden_states = num_hidden_states
        self.num_emissions = num_emissions
        self.state_transition_probs = np.random.dirichlet(np.ones(num_hidden_states), size=num_hidden_states)
        self.emission_probs = np.random.dirichlet(np.ones(num_emissions), size=num_hidden_states)
        self.start_state_probs = np.random.dirichlet(np.ones(num_hidden_states), size=1)[0]
    
    def sample(self, length: int):
        state = np.random.choice(self.num_hidden_states, p=self.start_state_probs)
        states, emissions = [], []
        for _ in range(length):
            states.append(state)
            emissions.append(np.random.choice(self.num_emissions, p=self.emission_probs[state]))
            state = np.random.choice(self.num_hidden_states, p=self.state_transition_probs[state])
        return emissions, states
    
    def score(self, emissions: list[int]):
        length = len(emissions)
        scores = np.zeros((length, self.num_hidden_states)) # in log space
        for i in range(self.num_hidden_states):
            scores[0, i] = np.log(self.start_state_probs[i]) + np.log(self.emission_probs[i, emissions[0]])
        for i in range(1, length):
            for j in range(self.num_hidden_states):
                temp = [scores[i - 1, k] + np.log(self.state_transition_probs[k, j]) + np.log(self.emission_probs[j, emissions[i]])
                        for k in range(self.num_hidden_states)]
                scores[i, j] = np.logaddexp.reduce(temp)
        return np.logaddexp.reduce(scores[-1])


class MixtureOfHmms:
    num_hmms: int
    hmms: list[HMM]
    weights: np.ndarray # shape: (num_hmms,)

    def __init__(self, num_hmms: int, num_hidden_states: int, num_emissions: int, uniform_weights: bool=False):
        self.num_hmms = num_hmms
        self.hmms = [HMM(num_hidden_states, num_emissions) for _ in range(num_hmms)]
        if uniform_weights:
            self.weights = np.ones(num_hmms) / num_hmms
        else:
            self.weights = np.random.dirichlet(np.ones(num_hmms), size=1)[0]
    
    def sample(self, length: int):
        hmm = np.random.choice(self.num_hmms, p=self.weights)
        return self.hmms[hmm].sample(length), hmm
    
    def score(self, emissions: list[int]):
        scores = [np.log(self.weights[i]) + self.hmms[i].score(emissions) for i in range(self.num_hmms)]
        return scores
    

class HMMDataset(Dataset):
    def __init__(self, hmms: MixtureOfHmms, num_train_examples: int=10000):
        super(HMMDataset, self).__init__()
        self.hmms = hmms
        self.length = num_train_examples
        self.emissions = []
        self.states = []
        self.hmm = []

        # generate data
        for _ in tqdm(range(self.length)):
            (emissions, states), hmm = self.hmms.sample(1000)
            self.emissions.append(emissions)
            self.states.append(states)
            self.hmm.append(hmm)
    
    def __len__(self):
        return self.length
    
    def __getitem__(self, idx):
        return {
            "input_ids": self.emissions[idx],
            "labels": self.emissions[idx],
            "states": self.states[idx],
            "hmm": self.hmm[idx],
        }