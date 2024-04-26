import torch
import transformers
from dataclasses import dataclass, field
from typing import Optional
from data import HMMDataset, MixtureOfHmms

device = "cpu"

@dataclass
class ModelArguments:
    model_type: str = field(default="llama", metadata={"help": "Model architecture."})
    num_hidden_layers: int = field(default=4, metadata={"help": "Number of layers in the transformer."})


@dataclass
class DataArguments:
    num_hmms: int = field(default=5, metadata={"help": "Number of HMMs in the mixture."})
    num_hidden_states: int = field(default=5, metadata={"help": "Number of hidden states in each HMM."})
    num_emissions: int = field(default=5, metadata={"help": "Number of emissions in each HMM."})
    uniform_weights: Optional[bool] = field(default=True, metadata={"help": "Whether to use uniform weights for the mixture."})
    num_train_examples: int = field(default=100, metadata={"help": "Number of training examples."})
    num_eval_examples: int = field(default=10, metadata={"help": "Number of evaluation examples."})


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    report_to: str = field(default="none", metadata={"help": "Where to report metrics."})


def train():
    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # set up model
    config = transformers.CONFIG_MAPPING[model_args.model_type](
        vocab_size=data_args.num_emissions,
        num_hidden_layers=model_args.num_hidden_layers,
        num_attention_heads=4, # 12
        num_key_value_heads=4, # 12
        hidden_size=64, # 768
    )
    model = transformers.AutoModelForCausalLM.from_config(config)
    transformers.LlamaForCausalLM
    n_params = sum({p.data_ptr(): p.numel() for p in model.parameters()}.values())
    print(f"Training new model from scratch - Total size={n_params/2**20:.2f}M params")

    # set up data
    hmms = MixtureOfHmms(num_hmms=data_args.num_hmms, num_hidden_states=data_args.num_hidden_states,
                         num_emissions=data_args.num_emissions, uniform_weights=data_args.uniform_weights)
    train_dataset = HMMDataset(hmms=hmms, num_train_examples=data_args.num_train_examples)
    eval_dataset = HMMDataset(hmms=hmms, num_train_examples=data_args.num_eval_examples)

    # set up trainer
    trainer = transformers.Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
    )
    trainer.train()


if __name__ == "__main__":
    train()