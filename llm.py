from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import json
import csv
from copy import deepcopy
from tqdm import tqdm
import random
import math
import pandas as pd
import os
import datasets
import argparse

# info about different models to simplify logit extraction
model_info = {
    "google/gemma-2b-it": {"start_of_turn": 106, "trigger": 2516, "user": 1645, "context_window": 4000},
    "google/gemma-7b-it": {"start_of_turn": 106, "trigger": 2516, "user": 1645, "context_window": 4000},
    "google/gemma-1.1-2b-it": {"start_of_turn": 106, "trigger": 2516, "user": 1645, "context_window": 4000},
    "google/gemma-1.1-7b-it": {"start_of_turn": 106, "trigger": 2516, "user": 1645, "context_window": 4000},
    "google/gemma-2-2b-it": {"start_of_turn": 106, "trigger": 2516, "user": 1645, "context_window": 4000},
    "google/gemma-2-7b-it": {"start_of_turn": 106, "trigger": 2516, "user": 1645, "context_window": 4000},
    "Qwen/Qwen2-0.5B-Instruct": {"start_of_turn": 151644, "trigger": 77091, "user": 872, "context_window": 16384},
    "Qwen/Qwen2-1.5B-Instruct": {"start_of_turn": 151644, "trigger": 77091, "user": 872, "context_window": 16384},
}

dataset_info = {
    "creak": {"end_pos": 2},
    "logiqa": {"end_pos": 2},
    "harmbench": {"end_pos": 4},
    "evals/persona/psychopathy": {"end_pos": 2},
    "evals/persona/machiavellianism": {"end_pos": 2},
    "evals/persona/narcissism": {"end_pos": 2},
}

## DATASET PREPROCESSING AND LOADING
def load_dataset(
    dataset: str="harmbench",
    fixed_response: bool=False
):
    """Load and process the dataset for evaluation."""
    assert dataset in dataset_info, "Invalid dataset"
    qa_pairs = []
    if dataset == "harmbench":
        harmbench_prompts, harmbench_answers = [], []
        with open('datasets/harmbench/harmbench.csv', 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            for row in reader:
                if row[4]:
                    harmbench_prompts.append(row[0] + '\n\n' + row[4])
                else:
                    harmbench_prompts.append(row[0])
        with open('datasets/harmbench/harmbench_wizardlm.csv', 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            for row in reader:
                harmbench_answers.append(('Here is how ' + row[0].split(' USER:')[0].strip().split(' AUSER:')[0].strip(),))
        qa_pairs = list(zip(harmbench_prompts, harmbench_answers))
    elif dataset == "logiqa":
        split = datasets.load_dataset("lucasmccabe/logiqa", split="test")
        def format_prompt(ex):
            prompt = f"Passage: {ex['context']}\nQuestion: {ex['query']}\nChoices:\n"
            for i, choice in enumerate(ex["options"]):
                letter = chr(ord("A") + i)
                prompt += f"{letter}. {choice}\n"
            prompt += "Answer:"
            answer = chr(ord("A") + ex["correct_option"])
            return (prompt, (answer,))
        qa_pairs = [format_prompt(ex) for ex in split]
    elif dataset == "creak":
        filename = "datasets/creak/dev.json"
        qa_pairs = []
        with open(filename, "r") as f:
            for line in f:
                data = json.loads(line)
                qa_pairs.append((data["sentence"], (data["label"],)))
    elif dataset.startswith("evals/persona/"):
        subset = dataset.split("/")[-1]
        filename = f"datasets/evals/persona/{subset}.jsonl"
        qa_pairs = []
        with open(filename, "r") as f:
            for line in f:
                data = json.loads(line)
                if fixed_response:
                    qa_pairs.append((data["question"], (" Yes", " No")))
                else:
                    qa_pairs.append((data["question"], (data["answer_matching_behavior"], data["answer_not_matching_behavior"])))
    return qa_pairs

# COMPUTING LOGITS
def format_token(tokenizer: AutoTokenizer, tok: int) -> str:
    """Format the token for some path patching experiment to show decoding diff"""
    return tokenizer.decode(tok).replace(" ", "_").replace("\n", "\\n")

def top_vals(tokenizer: AutoTokenizer, res: torch.Tensor, n: int=10, return_results: bool=False) -> list | None:
    """Pretty print the top n values of a distribution over the vocabulary"""
    top_values, top_indices = torch.topk(res, n)
    ret = []
    for i, _ in enumerate(top_values):
        tok = format_token(tokenizer, top_indices[i].item())
        ret += [(tok, top_values[i].item())]
        if not return_results:
            print(f"{tok:<20} {top_values[i].item()}")
    if return_results:
        return ret

def get_logits(
    inputs: dict,
    model: AutoModelForCausalLM,
    poses: list | torch.Tensor,
    mode: str="harmbench",
    tokenizer: AutoTokenizer | None=None
) -> list:
    """Get logit values for some prefix of each response in a multi-turn prompt."""
    torch.cuda.empty_cache()
    data = []
    out = model(**inputs)
    offset = dataset_info[mode]["end_pos"]
    for p, pos in enumerate(poses):
        cur_pos = pos[1]
        prob = 0
        end_pos = cur_pos + offset
        for k in range(cur_pos + 1, end_pos):
            next_tok = inputs["input_ids"][0, k + 1]
            # print(format_token(tokenizer, inputs["input_ids"][0, k]), " -> ", format_token(tokenizer, next_tok))
            probs = out.logits[0, k].log_softmax(-1)
            prob -= probs[next_tok]
        data.append({
            "shots": p,
            # "tokens": (cur_pos + 1).item(),
            "nll": prob.item(),
            "model": model.config._name_or_path,
        })
    return data

# MODEL LOADING
device = "cuda:0" if torch.cuda.is_available() else "cpu"

@torch.inference_mode()
def load_model(model_name: str="google/gemma-2b-it") -> tuple[AutoModelForCausalLM, AutoTokenizer]:
    with torch.inference_mode():
        torch.cuda.empty_cache()
    
        # load models
        model_kwargs = {
            "torch_dtype": torch.bfloat16,
            "device_map": device,
            "attn_implementation": "flash_attention_2",
            # "load_in_8bit": True,
        }
        tokenizer = AutoTokenizer.from_pretrained(model_name) # same tokenizer necessary
        model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
        model.eval()
    return model, tokenizer

@torch.inference_mode()
def test_model(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    evals: list=["evals/persona/psychopathy", "evals/persona/machiavellianism", "evals/persona/narcissism"]
) -> list:
    model_name = model.config._name_or_path
    data = []
    for dataset in evals:
        qa_pairs = load_dataset(dataset)
        print(len(qa_pairs))
        
        # compute NLL
        ct = 50
        shots = 1000
        trigger = model_info[model_name]["trigger"]
        user = model_info[model_name]["user"]
        for _ in tqdm(range(ct)):
            random.shuffle(qa_pairs)
            question, answer = qa_pairs[0]
            for hmm in range(len(answer)):
                chat = []
                for j in range(shots):
                    chat.append({"role": "user", "content": qa_pairs[j][0]})
                    chat.append({"role": "assistant", "content": qa_pairs[j][1][hmm]})
                    new_inputs = tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=False)
                    new_inputs = tokenizer(new_inputs, return_tensors="pt").to(device)
                    if new_inputs["input_ids"].shape[-1] > model_info[model_name]["context_window"]:
                        break
                    inputs = new_inputs
                if _ == 0:
                    text = tokenizer.decode(inputs["input_ids"][0])
                    path = f"logs/real-lms/{model_name.replace('/', '_')}___{dataset.replace('/', '_')}___example.txt"
                    with open(path, "w") as f:
                        f.write(text)
                    
                poses = (inputs["input_ids"] == trigger).nonzero()
                more_data = get_logits(inputs, model, poses, mode=dataset, tokenizer=tokenizer)
                for d in more_data:
                    d["hmm"] = hmm
                    d["dataset"] = dataset
                data.extend(more_data)
    return data

@torch.inference_mode()
def main(
    model_name: str="google/gemma-2b-it"
):
    evals = list(dataset_info.keys())
    models = [model_name]
    if model_name == "all":
        models = list(model_info.keys())
    for model_name in models:
        model, tokenizer = None, None
        print(f"Running: {model_name}")
        for dataset in evals:
            # skip if already computed
            m = model_name.replace("/", "_")
            d = dataset.replace("/", "_")
            path = f"logs/real-lms/{m}___{d}___means.csv"
            if os.path.exists(path):
                continue
            print(f"Running: {model_name} on {dataset}")

            # load model and collect data
            if model is None:
                model, tokenizer = load_model(model_name)
            data = test_model(model, tokenizer, evals=[dataset])

            # assemble data
            df = pd.DataFrame(data)
            df["prob"] = df["nll"].map(lambda x: math.exp(-x))
            df["shots"] += 1
            print(len(df))

            # make NLL the actual log expected prob
            df_mean = df.groupby(["dataset", "shots", "model", "hmm"]).mean().reset_index()
            df_mean["nll_avg"] = df_mean["nll"]
            df_mean["nll"] = df_mean["prob"].map(lambda x: -math.log(x))
            df_mean.to_csv(path)
            print("Saved to: ", path)

if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--model", type=str, default="google/gemma-2b-it")
    args = argparser.parse_args()
    main(args.model)