import os
import torch
import transformers
from dataclasses import dataclass, field, asdict
from typing import Optional
from data import HMMDataset, MixtureOfHmms, HMMInContextDataset
from copy import deepcopy
import numpy as np
from collections import defaultdict

import pandas as pd
from plotnine import ggplot, aes, geom_point, facet_wrap, geom_line, stat_summary
from plotnine.scales import scale_y_log10, scale_x_log10

device = "cpu" if not torch.cuda.is_available() else "cuda"

@dataclass
class ModelArguments:
    model_type: str = field(default="gpt2", metadata={"help": "Model architecture."})
    num_hidden_layers: int = field(default=4, metadata={"help": "Number of layers in the transformer."})


@dataclass
class DataArguments:
    num_hmms: int = field(default=5, metadata={"help": "Number of HMMs in the mixture."})
    num_entities: int = field(default=10, metadata={"help": "Number of entities in each HMM."})
    num_properties: int = field(default=10, metadata={"help": "Number of properties in each HMM."})
    num_emissions: int = field(default=50, metadata={"help": "Number of emissions in each HMM."})
    num_train_examples: int = field(default=1000, metadata={"help": "Number of training examples."})
    num_eval_examples: int = field(default=50, metadata={"help": "Number of evaluation examples."})
    # xie: Optional[bool] = field(default=True, metadata={"help": "Whether to use Xie's HMM."})
    k: int = field(default=3, metadata={"help": "Length of each in-context example."})
    num_in_context_examples: int = field(default=1000, metadata={"help": "Number of in-context examples."})
    num_in_context_shots: int = field(default=64, metadata={"help": "Number of in-context shots."})


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    report_to: str = field(default="wandb", metadata={"help": "Where to report metrics."})
    logging_steps: int = field(default=10, metadata={"help": "Log every n steps."})
    logging_strategy: str = field(default="steps", metadata={"help": "Log every n steps or every n epochs."})
    remove_unused_columns: bool = field(default=True)
    wandb_project: str = field(default="toy-alignment")
    wandb_entity: str = field(default="aryamanarora")
    per_device_train_batch_size: int = field(default=8)
    per_device_eval_batch_size: int = field(default=8)
    gradient_accumulation_steps: int = field(default=1)
    torch_compile: bool = field(default=True)
    num_train_epochs: int = field(default=5)
    learning_rate: float = field(default=8e-4)
    warmup_steps: float = field(default=1000)
    data_args: dict = field(default_factory=dict)
    


def in_context_eval(trainer, in_context_dataset, data_args, title="in_context_probs"):
    # in-context eval
    subsets = in_context_dataset.make_subsets()
    in_context_probs = []
    for hmm, subset in subsets.items():
        result = trainer.predict(subset)
        probs = torch.tensor(result.predictions).softmax(-1) # shape: (bs, seq, num_emissions)
        shots = 0
        for i in range(data_args.k - 1, probs.shape[1], data_args.k + 1):
            label = result.label_ids[:, i] # shape: (bs,)
            # get the prob of the correct label for each example
            prob = probs[torch.arange(probs.shape[0]), i, label]
            argmax = probs[torch.arange(probs.shape[0]), i].argmax(-1)
            for j in range(len(prob)):
                in_context_probs.append({
                    "k": shots,
                    "prob": prob[j].item(),
                    "acc": 1 if (argmax[j] == label[j]).item() else 0,
                    "nll": -torch.log(prob[j]).item(),
                    "hmm": str(hmm)
                })
            shots += 1
    
    # plot each
    df = pd.DataFrame(in_context_probs)
    plot = (
        ggplot(df, aes(x="k", y="acc")) +
        # geom_point(alpha=0.1, stroke=0) +
        facet_wrap("~hmm") +
        stat_summary(group="k", geom="line")
    )
    plot.save(f"{trainer.args.output_dir}/{title}.png")
    plot = (
        ggplot(df, aes(x="k", y="prob")) +
        geom_point(alpha=0.1, stroke=0) +
        facet_wrap("~hmm") +
        stat_summary(group="k", geom="line")
    )
    plot.save(f"{trainer.args.output_dir}/{title}_p.png")
    plot = (
        ggplot(df, aes(x="k", y="nll")) +
        # geom_point(alpha=0.1, stroke=0) +
        facet_wrap("~hmm") +
        stat_summary(group="k", geom="line") +
        scale_y_log10() + scale_x_log10()
    )
    plot.save(f"{trainer.args.output_dir}/{title}_nll.png")
    plot = (
        ggplot(df, aes(x="k", y="acc")) +
        # geom_point(alpha=0.1, stroke=0) +
        stat_summary(group="k", geom="line")
    )
    plot.save(f"{trainer.args.output_dir}/{title}_all.png")


def train():
    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    training_args.data_args = asdict(data_args)

    # wandb setup
    os.environ['WANDB_ENTITY'] = training_args.wandb_entity
    os.environ['WANDB_PROJECT'] = training_args.wandb_project

    # set up model
    config = transformers.CONFIG_MAPPING[model_args.model_type](
        vocab_size=data_args.num_emissions,
        num_hidden_layers=model_args.num_hidden_layers,
        num_attention_heads=12,
        num_key_value_heads=12,
        hidden_size=768,
        max_position_embeddings=1024,
    )
    if model_args.model_type == "llama":
        config.intermediate_size = 4 * 768
        config.tie_word_embeddings = True
    print(config)
    model = transformers.AutoModelForCausalLM.from_config(config)
    n_params = sum({p.data_ptr(): p.numel() for p in model.parameters()}.values())
    print(f"Training new model from scratch - Total size={n_params/(10**6):.2f}M params")

    # set up data
    hmms = MixtureOfHmms(
        num_hmms=data_args.num_hmms, num_entities=data_args.num_entities, num_properties=data_args.num_properties,
        vocab_size=data_args.num_emissions,
    )
    train_dataset = HMMDataset(hmms=hmms, num_train_examples=data_args.num_train_examples, sample_length=10240)
    eval_dataset = HMMDataset(hmms=hmms, num_train_examples=data_args.num_eval_examples, sample_length=1024)
    in_context_dataset = HMMInContextDataset(
        hmms=hmms, num_train_examples=data_args.num_in_context_examples,
        k=data_args.k, num_in_context_shots=data_args.num_in_context_shots
    )
    print("train_dataset:", len(train_dataset))
    print("eval_dataset:", len(eval_dataset))
    print("in_context_dataset:", len(in_context_dataset))

    # print(torch.tensor(hmms.score(train_dataset[0]["input_ids"][:20])).softmax(-1))
    # input()

    # set up trainer
    trainer = transformers.Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
    )
    trainer.train()

    # # eval
    # subsets = eval_dataset.make_subsets()
    # results = defaultdict(dict)
    # for hmm, subset in subsets.items():
    #     res = trainer.evaluate(eval_dataset=subset, metric_key_prefix=f'eval_{hmm}')
    #     results[hmm]["base"] = res[f'eval_{hmm}_loss']

    # in-context eval
    in_context_eval(trainer, in_context_dataset, data_args, title="in_context_probs")

    # sft
    # old_weights = deepcopy(hmms.weights)
    hmms.weights = np.zeros(hmms.num_hmms)
    hmms.weights[0] = 1
    sft_dataset = HMMDataset(hmms=hmms, num_train_examples=data_args.num_train_examples // 100, sample_length=10240)

    # train
    trainer.train_dataset = sft_dataset
    trainer.train()

    # # eval
    # subsets = eval_dataset.make_subsets()
    # for hmm, subset in subsets.items():
    #     res = trainer.evaluate(eval_dataset=subset, metric_key_prefix=f'sft_eval_{hmm}')
    #     results[hmm]["sft"] = res[f'sft_eval_{hmm}_loss']
    
    # # print
    # for hmm in sorted(list(results.keys())):
    #     print(f"{hmm}: {results[hmm]['base']:.4f} -> {results[hmm]['sft']:.4f}")

    # in-context eval
    in_context_eval(trainer, in_context_dataset, data_args, title="in_context_probs_sft")


if __name__ == "__main__":
    train()