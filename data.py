import numpy as np
from tqdm import tqdm
from torch.utils.data import Dataset
from typing import Union


class XieHMM:
    num_entities: int
    num_properties: int
    entity_transition_probs: np.ndarray # shape: (num_entities, num_entities)
    property_transition_probs: np.ndarray # shape: (num_properties, num_properties)
    start_entity_probs: np.ndarray # shape: (num_entities,)
    start_property_probs: np.ndarray # shape: (num_properties,)

    def __init__(self, num_entities: int, num_properties: int):
        self.num_entities = num_entities
        self.num_properties = num_properties

        # generate T
        # first need 100 random permutation matrices of size (num_properties, num_properties)
        weights = (np.random.random(100) - 0.5) / 0.1
        weights = np.exp(weights) / np.sum(np.exp(weights))
        T = np.zeros((num_properties, num_properties))
        for i in range(100):
            permutation_matrix = np.random.permutation(np.eye(num_properties))
            T += weights[i] * permutation_matrix
        self.property_transition_probs = T

        # start
        T_s = (np.random.random(num_properties) - 0.5) / 10
        T_s = np.exp(T_s) / np.sum(np.exp(T_s))
        self.start_property_probs = T_s

        # generate S
        # same process
        weights = (np.random.random(100) - 0.5) / 0.1
        weights = np.exp(weights) / np.sum(np.exp(weights))
        S = np.zeros((num_entities, num_entities))
        for i in range(100):
            permutation_matrix = np.random.permutation(np.eye(num_entities))
            S += weights[i] * permutation_matrix
        self.entity_transition_probs = S * 0.1 + np.eye(num_entities) * 0.9

        # start
        S_s = (np.random.random(num_entities) - 0.5) / 10
        S_s = np.exp(S_s) / np.sum(np.exp(S_s))
        self.start_entity_probs = S_s
    
    def sample(self, length: int):
        entity = np.random.choice(self.num_entities, p=self.start_entity_probs)
        property = np.random.choice(self.num_properties, p=self.start_property_probs)
        emissions, entities, properties = [], [], []
        for _ in range(length):
            entities.append(entity)
            properties.append(property)
            emissions.append(entity * self.num_properties + property)
            entity = np.random.choice(self.num_entities, p=self.entity_transition_probs[entity])
            property = np.random.choice(self.num_properties, p=self.property_transition_probs[property])
        return emissions, list(zip(entities, properties))


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
    hmms: list[Union[HMM, XieHMM]]
    weights: np.ndarray # shape: (num_hmms,)

    def __init__(self, num_hmms: int, num_hidden_states: int, num_emissions: int,
                 uniform_weights: bool=False, xie: bool=False):
        self.num_hmms = num_hmms

        if xie:
            self.hmms = [XieHMM(num_hidden_states, num_emissions) for _ in range(num_hmms)]
        else:
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