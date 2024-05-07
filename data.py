import numpy as np
from tqdm import tqdm
from torch.utils.data import Dataset
from typing import Union
from collections import defaultdict
from copy import deepcopy


class XieHMM:
    num_entities: int
    num_properties: int
    entity_transition_probs: np.ndarray # shape: (num_entities, num_entities)
    property_transition_probs: np.ndarray # shape: (num_properties, num_properties)
    start_entity_probs: np.ndarray # shape: (num_entities,)
    start_property_probs: np.ndarray # shape: (num_properties,)

    def __init__(
        self, num_entities: int, num_properties: int, vocab_size: int=100,
        entity_transition_probs: np.ndarray=None, emission_mapping: np.ndarray=None,
    ):
        self.num_entities = num_entities
        self.num_properties = num_properties
        self.num_hidden_states = num_entities * num_properties

        # generate property probs
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

        # generate entity probs
        # same process
        if entity_transition_probs is None:
            weights = (np.random.random(100) - 0.5) / 0.1
            weights = np.exp(weights) / np.sum(np.exp(weights))
            S = np.zeros((num_entities, num_entities))
            for i in range(100):
                permutation_matrix = np.random.permutation(np.eye(num_entities))
                S += weights[i] * permutation_matrix
            self.entity_transition_probs = S * 0.1 + np.eye(num_entities) * 0.9
        else:
            self.entity_transition_probs = entity_transition_probs

        S_s = (np.random.random(num_entities) - 0.5) / 10
        S_s = np.exp(S_s) / np.sum(np.exp(S_s))
        self.start_entity_probs = S_s

        # combine entity and property transitions/start probs into one matrix
        transition = np.zeros((num_entities * num_properties, num_entities * num_properties))
        for i in range(num_entities):
            for j in range(num_entities):
                transition[
                    i * num_properties: (i + 1) * num_properties, j * num_properties: (j + 1) * num_properties
                ] = self.property_transition_probs * self.entity_transition_probs[i, j]
        self.state_transition_probs = transition

        start = np.zeros(num_entities * num_properties)
        for i in range(num_entities):
            start[i * num_properties: (i + 1) * num_properties] = self.start_property_probs * self.start_entity_probs[i]
        self.start_state_probs = start

        # emission mapping
        if emission_mapping is None:
            # self.emission_mapping = np.arange(1, num_entities * num_properties + 1).reshape((num_entities, num_properties))
            self.emission_mapping = np.random.randint(low=1, high=vocab_size, size=(num_entities, num_properties))
            self.emission_mapping[:, 0] = 0 # delimiter
            self.emission_mapping = self.emission_mapping.flatten()
        else:
            self.emission_mapping = emission_mapping

        # also store as probs
        self.emission_probs = np.zeros((num_entities * num_properties, vocab_size))
        for i in range(self.num_hidden_states):
            self.emission_probs[i, self.emission_mapping[i]] = 1
    
    def sample(self, length: int, return_argmax: bool=False):
        state = np.random.choice(self.num_hidden_states, p=self.start_state_probs)
        states, emissions = [], []
        argmaxes = []
        for _ in range(length):
            states.append(state)
            emissions.append(self.emission_mapping[state])
            if return_argmax:
                argmax_state = np.argmax(self.state_transition_probs[state])
                argmaxes.append(self.emission_mapping[argmax_state])
            state = np.random.choice(self.num_hidden_states, p=self.state_transition_probs[state])

        if return_argmax:
            return emissions, states, argmaxes
        
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

    def __init__(self, num_hmms: int, num_entities: int, num_properties: int,
                 num_emissions: int=100, uniform_weights: bool=False):
                 
        self.num_hmms = num_hmms
        self.hmms = []
        for i in range(num_hmms):
            if i == 0:
                self.hmms.append(XieHMM(num_entities, num_properties, num_emissions))
            else:
                self.hmms.append(XieHMM(
                    num_entities, num_properties, num_emissions,
                    entity_transition_probs=self.hmms[0].entity_transition_probs,
                    emission_mapping=self.hmms[0].emission_mapping,
                ))
        
        if uniform_weights:
            self.weights = np.ones(num_hmms) / num_hmms
        else:
            self.weights = np.random.dirichlet(np.ones(num_hmms), size=1)[0]
    
    def sample(self, length: int, hmm=None, **kwargs):
        if hmm is None:
            hmm = np.random.choice(self.num_hmms, p=self.weights)
        return self.hmms[hmm].sample(length, **kwargs), hmm
    
    def score(self, emissions: list[int]):
        scores = [np.log(self.weights[i]) + self.hmms[i].score(emissions) for i in range(self.num_hmms)]
        return scores
    

class HMMDataset(Dataset):
    def __init__(self, hmms: MixtureOfHmms, num_train_examples: int=10000, sample_length: int=1000, hmm: int=None):
        super(HMMDataset, self).__init__()
        self.hmms = hmms
        self.emissions = []
        self.states = []
        self.hmm = []
        self.block_size = 1024

        # generate data
        for _ in tqdm(range(num_train_examples)):
            (emissions, states), src_hmm = self.hmms.sample(sample_length, hmm=hmm)
            for start in range(0, sample_length, self.block_size):
                end = min(start + self.block_size, sample_length)
                self.emissions.append(emissions[start:end])
                self.states.append(states[start:end])
                self.hmm.append(src_hmm)

        self.length = len(self.hmm)
    
    def __len__(self):
        return self.length
    
    def __getitem__(self, idx):
        return {
            "input_ids": self.emissions[idx],
            "labels": self.emissions[idx],
            "states": self.states[idx],
            "hmm": self.hmm[idx],
        }
    
    def make_subsets(self):
        datasets = defaultdict(list)
        for i in range(self.length):
            item = self[i]
            datasets[item["hmm"]].append(item)
        return datasets


class HMMInContextDataset(Dataset):
    def __init__(self, hmms: MixtureOfHmms, num_train_examples: int=10000, k: int=10, num_in_context_shots: int=64):
        super(HMMInContextDataset, self).__init__()
        self.hmms = hmms
        self.length = num_train_examples
        self.emissions = []
        self.states = []
        self.labels = []
        self.hmm = []

        # generate data
        for _ in tqdm(range(self.length)):
            hmm = np.random.choice(self.hmms.num_hmms, p=self.hmms.weights)
            emissions_doc = []
            states_doc = []
            labels_doc = []
            for i in range(num_in_context_shots):

                # in-context example
                (emissions, states, labels), _ = self.hmms.sample(k, hmm=hmm, return_argmax=True)

                emissions_doc.extend(emissions)
                states_doc.extend(states)
                labels_doc.extend(labels)

                # delimiter
                if i != num_in_context_shots - 1:
                    emissions_doc.append(0)
                    states_doc.append(0)
                    labels_doc.append(0)
            
            # add doc
            self.hmm.append(hmm)
            self.emissions.append(emissions_doc)
            self.states.append(states_doc)
            self.labels.append(labels_doc)
    
    def __len__(self):
        return self.length
    
    def __getitem__(self, idx):
        return {
            "input_ids": self.emissions[idx],
            "labels": self.labels[idx],
            "states": self.states[idx],
            "hmm": self.hmm[idx],
        }
    
    def make_subsets(self):
        datasets = defaultdict(list)
        for i in range(self.length):
            item = self[i]
            datasets[item["hmm"]].append(item)
        return datasets