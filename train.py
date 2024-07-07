
import numpy as np
import os
import pandas as pd
import torch
import transformers
from collections import defaultdict
from copy import deepcopy
from data import HMMDataset, MixtureOfHmms, HMMInContextDataset, softmax
from dataclasses import asdict
from plotnine import ggplot, aes, geom_point, facet_wrap, facet_grid, geom_line, stat_summary, ylim
from plotnine.scales import scale_y_log10, scale_x_log10
from tqdm import tqdm
from trainer import ModelArguments, DataArguments, TrainingArguments, SFTTrainer, in_context_eval

device = "cpu" if not torch.cuda.is_available() else "cuda"

K = [3, 5, 8, 10]


def group_data(df: list) -> pd.DataFrame:
    return pd.DataFrame(df).groupby(["shots", "k", "hmm", "sft", "sft_amount"]).mean().reset_index()


def train():
    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # args
    if data_args.num_sft_examples is None:
        data_args.num_sft_examples = max(1, data_args.num_train_examples // 20)
    else:
        data_args.num_sft_examples = [int(x) for x in data_args.num_sft_examples.split(",")]
    training_args.data_args = asdict(data_args)
    training_args.model_args = asdict(model_args)
    training_args.output_dir = f"logs/{training_args.output_dir}"

    # wandb setup
    os.environ['WANDB_ENTITY'] = training_args.wandb_entity
    os.environ['WANDB_PROJECT'] = training_args.wandb_project

    # set up data
    hmms = MixtureOfHmms(
        num_hmms=data_args.num_hmms, num_entities=data_args.num_entities, num_properties=data_args.num_properties,
        vocab_size=data_args.num_emissions,
    )
    hmms.weights = np.array([float(x) for x in data_args.pretrain_dist.split(",")])
    hmms.weights /= hmms.weights.sum()
    data = []

    if model_args.ideal:

        # make directory
        if not os.path.exists(training_args.output_dir):
            os.makedirs(training_args.output_dir)

        # data
        in_context_datasets = {}
        for k in K:
            in_context_datasets[k] = HMMInContextDataset(
                hmms=hmms, num_train_examples=data_args.num_in_context_examples // 50,
                k=k, num_in_context_shots=1024 // (k + 1)
            )
            print(f"in_context_datasets[{k}]:", len(in_context_datasets[k]))

        # label smooth
        hmms_smoothed = hmms.to_label_smoothed(alpha=0.95)

        # in-context eval
        for k, in_context_dataset in in_context_datasets.items():
            subsets = in_context_dataset.make_subsets()
            for hmm, subset in subsets.items():
                for sequence in tqdm(subset):
                    seq = sequence['input_ids']
                    scores = np.zeros(len(subsets))
                    for i in range(k - 2, len(seq), k + 1):
                        subseq = seq[i - (k - 2):i + 2]
                        shots = i // (k + 1)
                        score = hmms_smoothed.score(subseq)
                        scores += score
                        data.append({
                            "shots": shots,
                            "prob": softmax(scores)[hmm],
                            "acc": 1 if (np.argmax(scores) == hmm) else 0,
                            "nll": 0,
                            "hmm": str(hmm),
                            "sft": False,
                            "k": k
                        })

    else:

        # data
        train_dataset = HMMDataset(hmms=hmms, num_train_examples=data_args.num_train_examples, sample_length=10240)
        eval_dataset = HMMDataset(hmms=hmms, num_train_examples=data_args.num_eval_examples, sample_length=1024)
        in_context_datasets = {}
        old_weights = deepcopy(hmms.weights)
        hmms.weights = np.array([1.0 for _ in data_args.pretrain_dist.split(",")])
        hmms.weights /= hmms.weights.sum()
        for k in K:
            in_context_datasets[k] = HMMInContextDataset(
                hmms=hmms, num_train_examples=data_args.num_in_context_examples,
                k=k, num_in_context_shots=1024 // (k + 1)
            )
            print(f"in_context_datasets[{k}]:", len(in_context_datasets[k]))
        hmms.weights = old_weights

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

        # set up trainer
        trainer = transformers.Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
        )
        trainer.train()

        # evaluate
        trainer.evaluate()

        # per-HMM eval
        subsets = eval_dataset.make_subsets()
        results = defaultdict(dict)
        for hmm, subset in subsets.items():
            res = trainer.evaluate(eval_dataset=subset, metric_key_prefix=f'eval_{hmm}')
            results[hmm]["base"] = res[f'eval_{hmm}_loss']

        # in-context eval
        for k, in_context_dataset in in_context_datasets.items():
            more = in_context_eval(trainer, in_context_dataset, k)
            for m in more:
                m["sft"] = False
                m["sft_amount"] = 0
                m["k"] = k
            data.append(group_data(more))
        print("Length of data after pretraining:", len(data), data[-1])

        # save model
        trainer.save_model()
        trainer.state.save_to_json(f"{training_args.output_dir}/trainer_state.json")

        # delete stuff to save memory
        del model
        del trainer
        
        # SFT
        for sft_amount in data_args.num_sft_examples:
            # make dataset
            if sft_amount == 0:
                continue
            hmms.weights = np.array([float(x) for x in data_args.sft_dist.split(",")])
            hmms.weights /= hmms.weights.sum()
            sft_dataset = HMMDataset(hmms=hmms, num_train_examples=sft_amount, sample_length=10240)

            # load model
            sft_model = transformers.AutoModelForCausalLM.from_pretrained(training_args.output_dir).to(device)
            sft_trainer = SFTTrainer(
                model=sft_model,
                args=deepcopy(training_args),
                train_dataset=sft_dataset,
                eval_dataset=in_context_datasets,
            )
            sft_trainer.sft_amount = sft_amount
            sft_trainer.train_dataset = sft_dataset
            sft_trainer.args.num_train_epochs = 1
            sft_trainer.args.warmup_steps = 0
            sft_trainer.args.warmup_ratio = 0.1
            sft_trainer.args.set_evaluate(strategy="steps", steps=sft_amount // 5, delay=0)
            sft_trainer.train()

            # eval
            data.append(group_data(sft_trainer.data))
            print("Length of data after SFT training:", len(data), data[-1])
            subsets = eval_dataset.make_subsets()
            for hmm, subset in subsets.items():
                res = sft_trainer.evaluate(eval_dataset=subset, metric_key_prefix=f'eval_sft_{hmm}', old=True)
                results[hmm]["sft"] = res[f'eval_sft_{hmm}_loss']

            # print
            for hmm in sorted(list(results.keys())):
                print(f"{hmm}: {results[hmm]['base']:.4f} -> {results[hmm]['sft']:.4f}")

            # in-context eval
            for k, in_context_dataset in in_context_datasets.items():
                more = in_context_eval(sft_trainer, in_context_dataset, k)
                for m in more:
                    m["sft"] = True
                    m["k"] = k
                    m["sft_amount"] = sft_amount
                data.append(group_data(more))
            print("Length of data after SFT eval:", len(data), data[-1])
            
            # delete stuff to save memory
            del sft_dataset
            del sft_model
            del sft_trainer

    # plot each
    title = "in_context_probs"

    # concat the dfs in data
    df = pd.concat(data)
    df = df.groupby(["shots", "k", "hmm", "sft", "sft_amount"]).mean().reset_index()

    # save df
    df.to_csv(f"{training_args.output_dir}/{title}.csv")

    # dump trainer state
    # trainer.save_model()
    # trainer.state.save_to_json(f"{training_args.output_dir}/trainer_state.json")

    # save training args
    with open(f"{training_args.output_dir}/training_args.json", "w") as f:
        f.write(training_args.to_json_string())


if __name__ == "__main__":
    train()
