"""
Data generation code, largely based on the code for the GINC dataset from
Xie et al. (2022, https://arxiv.org/abs/2111.02080).
"""

import numpy as np
from tqdm import tqdm
from torch.utils.data import Dataset
from typing import Union
from collections import defaultdict
from copy import deepcopy
import contextlib
from hmmlearn.hmm import CategoricalHMM
import random
import matplotlib.pyplot as plt
import torch
from tokenizers import Tokenizer, models, pre_tokenizers, processors
from transformers import PreTrainedTokenizerFast
import datasets
import json


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def softmax(x, temp=1.0, axis=None):
    x /= temp
    if axis is None:
        x -= np.amax(x)
        return np.exp(x) / np.sum(np.exp(x))
    else:
        x -= np.expand_dims(np.amax(x, axis=axis), axis=axis)
        return np.exp(x) / np.expand_dims(np.sum(np.exp(x), axis=axis), axis=axis)


def generate_transmat_block(
        n_components, perm_samples=10, transition_temp=1.0):
    mixing = softmax(np.random.rand(perm_samples) - 0.5, transition_temp)
    mixing = mixing[:, np.newaxis, np.newaxis]
    perm_samples = [np.eye(n_components)[np.random.permutation(n_components)] for i in range(perm_samples)]
    transmat = np.sum(mixing * perm_samples, axis=0)

    return transmat


def combine_transmats(mat1, mat2):
    # combine by tiling mat1 and scaling with mat2
    n = mat1.shape[0]
    m = mat2.shape[0]
    mat = np.zeros((m*n, m*n))
    for i in range(m):
        for j in range(m):
            mat[i*n: (i+1)*n, j*n: (j+1)*n] = mat1 * mat2[i,j]
    return mat


@contextlib.contextmanager
def local_seed(seed):
    state = np.random.get_state()
    np.random.seed(seed)
    try:
        yield
    finally:
        np.random.set_state(state)


def generate_hmm_parameters(
        n_values: int, n_slots: int, n_symbols: int, all_values, perm_samples=10, transition_temp=1.0,
        start_temp=1.0, value_transmat_id_coeff=0.8, value_transmat_seed=1112, prior_values=False
    ):
    n_components = n_values * n_slots

    # generate parameters for HMM
    startprob = softmax(np.random.rand(n_components) - 0.5, start_temp)

    slot_transmat = generate_transmat_block(
            n_slots, perm_samples=n_slots, transition_temp=transition_temp)

    if prior_values:
        value_transmat = generate_transmat_block(
                n_values, perm_samples=n_values, transition_temp=transition_temp)
        # bias the value transmat towards identity
        value_transmat = (1-value_transmat_id_coeff) * value_transmat + value_transmat_id_coeff * np.eye(n_values)
    else:
        with local_seed(value_transmat_seed):
            value_transmat = generate_transmat_block(
                    n_values, perm_samples=n_values, transition_temp=transition_temp)
            # bias the value transmat towards identity
            value_transmat = (1-value_transmat_id_coeff) * value_transmat + value_transmat_id_coeff * np.eye(n_values)

    transmat = combine_transmats(slot_transmat, value_transmat)

    # this is actually the same for all hmms, given all_values
    emissionprob = np.zeros((n_components, n_symbols))
    for i in range(n_components):
        # deterministic given slot and value vector
        slot_idx = i % n_slots
        value_idx = i // n_slots
        emissionprob[i, all_values[value_idx, slot_idx]] = 1

    return startprob, transmat, emissionprob, slot_transmat, value_transmat


class XieHMM:
    num_entities: int
    num_properties: int
    num_hidden_states: int
    entity_transition_probs: np.ndarray # shape: (num_entities, num_entities)
    property_transition_probs: np.ndarray # shape: (num_properties, num_properties)
    start_entity_probs: np.ndarray # shape: (num_entities,)
    start_property_probs: np.ndarray # shape: (num_properties,)

    def __init__(
        self, num_entities: int, num_properties: int, vocab_size: int, all_values: np.ndarray,
        transition_temp: float, start_temp: float, value_transmat_id_coeff: float,
        value_transmat_seed: float,
    ):
        
        # init params
        self.num_entities = num_entities
        self.num_properties = num_properties
        self.vocab_size = vocab_size
        self.num_hidden_states = num_entities * num_properties
        self.num_perm_samples = self.num_hidden_states

        # make hmm params
        startprob, transmat, emissionprob, slot_transmat, value_transmat = generate_hmm_parameters(
            self.num_entities,
            self.num_properties,
            self.vocab_size,
            all_values,
            self.num_perm_samples,
            transition_temp,
            start_temp,
            value_transmat_id_coeff,
            value_transmat_seed,
        )
        self.slot_transmat = slot_transmat
        self.value_transmat = value_transmat

        # make hmm
        self.hmm = CategoricalHMM(n_components=self.num_hidden_states)
        self.hmm.startprob_ = startprob
        self.hmm.transmat_ = transmat
        self.hmm.emissionprob_ = emissionprob


    def sample_from_hmm(self, length: int, seed=None):
        x, h = self.hmm.sample(n_samples=length, random_state=seed)
        return x.T[0], h


    def generate_hiddens_from_state(self, start_state: int, length: int):
        hiddens = [start_state]
        for _ in range(length):
            hiddens.append(np.random.choice(self.hmm.transmat_.shape[1], p=self.hmm.transmat_[hiddens[-1], :]))
        return hiddens
    

    def score(self, emissions: list[int]):
        return self.hmm.score(np.array(emissions).reshape(-1, 1))
    

class MixtureOfHmms:
    num_hmms: int
    hmms: XieHMM
    weights: np.ndarray # shape: (num_hmms,)

    def __init__(
        self, num_hmms: int, num_entities: int, num_properties: int,
        vocab_size: int=100
    ):
        # basic params
        self.num_hmms = num_hmms
        self.hmms = []
        self.num_entities = num_entities # value
        self.num_properties = num_properties # slot
        self.vocab_size = vocab_size
        self.weights = np.ones(num_hmms) / num_hmms

        # seed
        seed = 1111
        np.random.seed(seed)
        random.seed(seed+2)

        # num_values number of num_slots sized lists of vocab words
        self.all_values = np.random.randint(low=1, high=vocab_size, size=(num_entities, num_properties))
        # make sure every row has a delimiter
        self.all_values[:, 0] = 0

        # make hmms
        for _ in range(self.num_hmms):
            hmm = XieHMM(
                num_entities=num_entities, num_properties=num_properties, vocab_size=vocab_size,
                all_values=self.all_values, transition_temp=0.1, start_temp=10.0, value_transmat_id_coeff=0.9,
                value_transmat_seed=seed + 3,
            )
            self.hmms.append(hmm)
        
        # make tokenizer
        vocab = {str(i): i for i in range(vocab_size)}
        vocab["[PAD]"] = vocab_size
        vocab["[BOS]"] = vocab_size + 1
        tokenizer = Tokenizer(models.BPE(vocab=vocab, merges=[]))
        tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
        tokenizer.enable_padding(pad_id=vocab_size)
        # tokenizer.enable_truncation(max_length=10240)

        # save tokenizer to json
        tokenizer.save("tokenizer.json")

        # load tokenizer
        self.tokenizer = PreTrainedTokenizerFast(tokenizer_file="tokenizer.json")
    

    def sample(self, num_samples: int, length: int):
        all_emissions, all_states, all_hmms = [], [], []
        for _ in tqdm(range(num_samples)):
            hmm = np.random.choice(self.num_hmms, p=self.weights)
            emissions, states = self.hmms[hmm].sample_from_hmm(length)
            all_emissions.append(emissions)
            all_states.append(states)
            all_hmms.append(hmm)
        return all_emissions, all_states, all_hmms


    def __getitem__(self, idx):
        return self.hmms[idx]
    

    def score(self, emissions: list[int]):
        scores = np.array([np.log(self.weights[i]) + self.hmms[i].score(emissions) for i in range(self.num_hmms)])
        return scores
    

    def to_label_smoothed(self, alpha: float=0.01):
        # plot transmat
        plt.imshow(self.hmms[0].hmm.transmat_, vmin=0, vmax=1)
        plt.savefig("transmat.png")

        smoothed_mixture = deepcopy(self)
        for hmm in smoothed_mixture.hmms:
            hmm.hmm.transmat_ = (1 - alpha) * hmm.hmm.transmat_ + alpha / hmm.hmm.transmat_.shape[0]

        # plot transmat
        plt.imshow(smoothed_mixture.hmms[0].hmm.transmat_, vmin=0, vmax=1)
        plt.savefig("smoothedtransmat.png")

        return smoothed_mixture
    

    def detokenize(self, emissions: list[int]):
        return " " + " ".join([str(emission) for emission in emissions])


class HMMDataset(Dataset):
    def __init__(
        self, hmms: MixtureOfHmms, num_train_examples: int=10000, sample_length: int=1000, hmm: int=None,
        block_size: int=1024, bos: bool=True
    ):
        super(HMMDataset, self).__init__()
        self.hmms = hmms
        self.emissions = []
        self.states = []
        self.hmm = []
        self.block_size = block_size

        # generate data
        emissions, states, hmm = hmms.sample(num_train_examples, (sample_length - 1) if bos else sample_length)
        if bos:
            emissions = [np.concatenate([[hmms.vocab_size + 1], emission]) for emission in emissions]
            states = [np.concatenate([[-1], state]) for state in states]
        for i in range(num_train_examples):
            assert len(emissions[i]) == sample_length, f"emission length: {len(emissions[i])}, sample_length: {sample_length}: {emissions[i]}"

        # concatenate and make `block_size`-sized chunks
        if self.block_size < sample_length:
            for i in range(0, num_train_examples * sample_length, self.block_size):
                cur_doc = i // sample_length
                cur_pos = i % sample_length
                next_doc = (i + self.block_size) // sample_length
                next_pos = (i + self.block_size) % sample_length
                chunk_emissions, chunk_states = [], []
                hmms = set()
                for j in range(cur_doc, next_doc + 1):
                    if j >= num_train_examples:
                        continue
                    if cur_doc == next_doc:
                        chunk_emissions.extend(emissions[j][cur_pos:next_pos])
                        chunk_states.extend(states[j][cur_pos:next_pos])
                    elif j == cur_doc:
                        chunk_emissions.extend(emissions[j][cur_pos:])
                        chunk_states.extend(states[j][cur_pos:])
                    elif j == next_doc:
                        chunk_emissions.extend(emissions[j][:next_pos])
                        chunk_states.extend(states[j][:next_pos])
                    else:
                        chunk_emissions.extend(emissions[j])
                        chunk_states.extend(states[j])
                    hmms.add(hmm[j])
                self.emissions.append(chunk_emissions)
                self.states.append(chunk_states)
                if len(hmms) == 1:
                    self.hmm.append(hmms.pop())
                else:
                    self.hmm.append(-1)
        else:
            self.emissions = emissions
            self.states = states
            self.hmm = hmm

        # length
        self.length = len(self.emissions)
    
    def __len__(self):
        return self.length
    
    def __getitem__(self, idx):
        return {
            "input_ids": self.emissions[idx],
            "labels": self.emissions[idx],
            # "states": self.states[idx],
            # "hmm": self.hmm[idx] if (len(self.hmm) > 0) else 0,
        }
    
    def make_subsets(self):
        datasets = defaultdict(list)
        for i in range(self.length):
            item = self[i]
            datasets[self.hmm[i]].append(item)
        return datasets


class HMMPreferenceDataset(Dataset):

    @staticmethod
    def make_dataset(
        hmms: MixtureOfHmms, accepted_dist: list[int], rejected_dist: list[int],
        num_train_examples: int=10000, sample_length: int=1000, hmm: int=None,
        block_size: int=1024,
    ):
        assert block_size >= sample_length, "block_size must be greater than or equal to sample_length"

        # make dataset
        dpo_dataset_dict = {}

        # first chosen
        old_weights = hmms.weights
        hmms.weights = np.array(accepted_dist)
        dataset = HMMDataset(hmms, num_train_examples, sample_length, hmm, block_size, bos=False)
        dpo_dataset_dict["chosen"] = dataset.emissions
        dpo_dataset_dict["chosen_input_ids"] = dataset.emissions
        dpo_dataset_dict["chosen_labels"] = dataset.emissions
        dpo_dataset_dict["chosen_attention_mask"] = [[1] * len(emission) for emission in dataset.emissions]

        # then rejected
        hmms.weights = np.array(rejected_dist)
        dataset = HMMDataset(hmms, num_train_examples, sample_length, hmm, block_size, bos=False)
        dpo_dataset_dict["rejected"] = dataset.emissions
        dpo_dataset_dict["rejected_input_ids"] = dataset.emissions
        dpo_dataset_dict["rejected_labels"] = dataset.emissions
        dpo_dataset_dict["rejected_attention_mask"] = [[1] * len(emission) for emission in dataset.emissions]

        # prompts are empty strings
        dpo_dataset_dict["prompt"] = [""] * len(dpo_dataset_dict["chosen_input_ids"])
        dpo_dataset_dict["prompt_input_ids"] = [[hmms.vocab_size + 1,]] * len(dpo_dataset_dict["chosen_input_ids"])
        dpo_dataset_dict["prompt_token_type_ids"] = [[0,]] * len(dpo_dataset_dict["chosen_input_ids"])
        dpo_dataset_dict["prompt_attention_mask"] = [[1,]] * len(dpo_dataset_dict["chosen_input_ids"])

        # reset weights
        hmms.weights = old_weights

        # make dataset
        dataset = datasets.Dataset.from_dict(dpo_dataset_dict)
        return dataset


class HMMInContextDataset(Dataset):
    def __init__(self, hmms: MixtureOfHmms, num_train_examples: int=10000, k: int=10,
                 num_in_context_shots: int=64, use_native_hmm_weights: bool=False):
        super(HMMInContextDataset, self).__init__()
        self.hmms = hmms
        self.length = num_train_examples
        self.emissions = []
        self.properties = []
        self.entities = []
        self.labels = []
        self.hmm = []

        # generate data
        for _ in tqdm(range(self.length)):
            hmm = np.random.choice(self.hmms.num_hmms, p=self.hmms.weights) if use_native_hmm_weights else np.random.choice(list(range(self.hmms.num_hmms)))
            properties = []
            entities = []
            labels = []

            # choose start such that we sample one start slot
            start_property = np.random.randint(low=1, high=self.hmms.num_properties) # slot
            
            for shot in range(num_in_context_shots):
                start_entity = np.random.randint(low=0, high=self.hmms.num_entities) # value
                start_hidden_idx = start_entity * self.hmms.num_properties + start_property
                
                # make cur examples
                h = self.hmms[hmm].generate_hiddens_from_state(start_hidden_idx, length=k - 1)
                cur_properties = [hidden % self.hmms.num_properties for hidden in h]
                cur_entities = [hidden // self.hmms.num_properties for hidden in h]
                properties += cur_properties
                entities += cur_entities

                # labels
                cur_labels = [np.argmax(self.hmms[hmm].hmm.transmat_[hidden, :]) for hidden in h]
                cur_labels = [self.hmms.all_values[hidden // self.hmms.num_properties, hidden % self.hmms.num_properties] for hidden in cur_labels]
                labels += cur_labels

                # print(shot)
                # print("entities:", cur_entities)
                # print("properties:", cur_properties)
                # print("inputs:", [self.hmms.all_values[cur_entities[j], cur_properties[j]] for j in range(len(cur_properties))])
                # print("labels:", cur_labels)
                # input()

                # delimiter
                properties += [0]
                entities += [entities[-1]]
                labels += [0]
            
            # add doc
            prompt = [self.hmms.all_values[entities[j], properties[j]] for j in range(len(properties))]

            self.hmm.append(hmm)
            self.emissions.append(prompt)
            self.properties.append(properties)
            self.entities.append(entities)
            self.labels.append(labels)
    
    def __len__(self):
        return self.length
    
    def __getitem__(self, idx):
        return {
            "input_ids": self.emissions[idx],
            "labels": self.labels[idx],
            # "entities": self.entities[idx],
            # "properties": self.properties[idx],
            # "hmm": self.hmm[idx],
        }
    
    def make_subsets(self):
        datasets = defaultdict(list)
        for i in range(self.length):
            item = self[i]
            datasets[self.hmm[i]].append(item)
        return datasets


if __name__ == "__main__":
    # make hmms
    hmms = MixtureOfHmms(num_hmms=5, num_entities=10, num_properties=10)

    # dataset
    dataset = HMMDataset(hmms, num_train_examples=10, sample_length=100)