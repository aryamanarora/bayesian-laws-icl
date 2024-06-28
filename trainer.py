import torch
import transformers
from dataclasses import dataclass, field
from typing import List


@dataclass
class ModelArguments:
    model_type: str = field(default="gpt2", metadata={"help": "Model architecture."})
    num_hidden_layers: int = field(default=4, metadata={"help": "Number of layers in the transformer."})
    ideal: bool = field(default=False)


@dataclass
class DataArguments:
    num_hmms: int = field(default=5, metadata={"help": "Number of HMMs in the mixture."})
    num_entities: int = field(default=10, metadata={"help": "Number of entities in each HMM."})
    num_properties: int = field(default=10, metadata={"help": "Number of properties in each HMM."})
    num_emissions: int = field(default=50, metadata={"help": "Number of emissions in each HMM."})
    num_train_examples: int = field(default=1000, metadata={"help": "Number of training examples."})
    num_eval_examples: int = field(default=50, metadata={"help": "Number of evaluation examples."})
    num_sft_examples: str | None = field(default=None)
    # xie: Optional[bool] = field(default=True, metadata={"help": "Whether to use Xie's HMM."})
    k: int = field(default=3, metadata={"help": "Length of each in-context example."})
    num_in_context_examples: int = field(default=1000, metadata={"help": "Number of in-context examples."})
    pretrain_dist: str = field(default="1,1,1,1,1", metadata={"help": "Pretrain distribution."})
    sft_dist: str = field(default="1,0,0,0,0", metadata={"help": "SFT distribution."})


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    report_to: List[str] | None = field(default="wandb", metadata={"help": "Where to report metrics."})
    logging_steps: float = field(default=10, metadata={"help": "Log every n steps."})
    logging_strategy: str = field(default="steps", metadata={"help": "Log every n steps or every n epochs."})
    evaluation_strategy: str = field(default="epoch", metadata={"help": "Evaluate every n steps or every n epochs."})
    remove_unused_columns: bool | None = field(default=True)
    wandb_project: str = field(default="toy-alignment")
    wandb_entity: str = field(default="aryamanarora")
    per_device_train_batch_size: int = field(default=8)
    per_device_eval_batch_size: int = field(default=8)
    gradient_accumulation_steps: int = field(default=1)
    torch_compile: bool = field(default=True)
    num_train_epochs: float = field(default=5.0)
    learning_rate: float = field(default=8e-4)
    warmup_steps: int = field(default=1000)
    data_args: dict = field(default_factory=dict)
    model_args: dict = field(default_factory=dict)
    save_strategy: str = field(default="no")


def in_context_eval(trainer: transformers.Trainer, in_context_dataset, k: int):
    """Evaluate the model on in-context examples (OOD from pretraining)."""

    # what does the data look like? e.g. k = 3
    # h1 a1 b1 / h2 a2 b2 / h3 a3 b3
    # 0  1  2  3 4  5  6  7 8  9  10
    #    ^          ^          ^
    # we eval at ^
    # start at k - 2, step is k + 1

    subsets = in_context_dataset.make_subsets()
    in_context_probs = []
    for hmm, subset in subsets.items():
        result = trainer.predict(subset)
        probs = torch.tensor(result.predictions).softmax(-1) # shape: (bs, seq, num_emissions)
        for i in range(k - 2, probs.shape[1], k + 1):
            shots = i // (k + 1)
            label = result.label_ids[:, i] # shape: (bs,)
            # get the prob of the correct label for each example
            prob = probs[torch.arange(probs.shape[0]), i, label]
            argmax = probs[torch.arange(probs.shape[0]), i].argmax(-1)
            for j in range(len(prob)):
                in_context_probs.append({
                    "shots": shots,
                    "prob": prob[j].item(),
                    "acc": 1 if (argmax[j] == label[j]).item() else 0,
                    "nll": -torch.log(prob[j]).item(),
                    "hmm": str(hmm)
                })

    return in_context_probs


class SFTTrainer(transformers.Trainer):
    """Supervised finetuning trainer."""

    def evaluate(self, eval_dataset=None, ignore_keys=None, metric_key_prefix="eval", old=False):
        # create data member var if not exists
        if not hasattr(self, "data"):
            self.data = []
        if eval_dataset is None:
            eval_dataset = self.eval_dataset

        # eval
        if old:
            return super().evaluate(eval_dataset, ignore_keys, metric_key_prefix)
        else:
            for k, in_context_dataset in eval_dataset.items():
                step = self.state.global_step
                more = in_context_eval(self, in_context_dataset, k)
                for m in more:
                    m["k"] = k
                    m["sft"] = step
                    m["sft_amount"] = self.sft_amount
                self.data.extend(more)
            return {}