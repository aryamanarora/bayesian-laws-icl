
import numpy as np
import os
import pandas as pd
import torch
import transformers
import trl
from collections import defaultdict
from copy import deepcopy
from data import HMMDataset, MixtureOfHmms, HMMInContextDataset, HMMPreferenceDataset, softmax
from dataclasses import asdict
from tqdm import tqdm
from trainer import ModelArguments, DataArguments, TrainingArguments, SFTTrainer, DPOTrainer, DataCollator, in_context_eval
from calflops import calculate_flops

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
    run_name = training_args.output_dir.split("/")[-1]
    if training_args.load_dir is None:
        training_args.load_dir = training_args.output_dir

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
        train_dataset = None
        if model_args.do_pretrain:
            # set up model
            config = transformers.CONFIG_MAPPING[model_args.model_type](
                vocab_size=data_args.num_emissions + 2,
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

            # calculate flops
            eff_bs = training_args.per_device_train_batch_size * training_args.gradient_accumulation_steps
            flops, macs, params = calculate_flops(
                model=model,
                input_shape=(eff_bs, 1024),
                transformer_tokenizer=hmms.tokenizer,
                output_as_string=False,
            )
            print(f"Single forward pass FLOPs: {flops / (10**12):.3f}T")
            total_flops = float(flops) * training_args.num_train_epochs * (data_args.num_train_examples // eff_bs)
            print(f"Total forward pass FLOPs: {total_flops / (10**12):.3f}T")
            print(f"Total forward + backward pass FLOPs: {total_flops * 3 / (10**12):.3f}T")
            print(f"Training new model from scratch - Total size={params / (10**6):.2f}M params")

            # make datasets
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
        results = defaultdict(dict)
        
        if model_args.do_pretrain:
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

            # save results
            with open(f"{training_args.output_dir}/results.json", "w") as f:
                f.write(str(results))

            # delete stuff to save memory
            del model
            del trainer
        
        # SFT
        if training_args.sft_method == "sft":
            for sft_amount in data_args.num_sft_examples:
                # make dataset
                sft_dataset = None
                if sft_amount > 0:
                    hmms.weights = np.array([float(x) for x in data_args.sft_dist.split(",")])
                    hmms.weights /= hmms.weights.sum()
                    sft_dataset = HMMDataset(hmms=hmms, num_train_examples=sft_amount, sample_length=10240)

                # load model
                sft_model = transformers.AutoModelForCausalLM.from_pretrained(training_args.load_dir).to(device)
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
                sft_trainer.args.report_to = []
                # sft_trainer.args.run_name = f"{run_name}_sft-{sft_amount}"
                if sft_amount > 0:
                    total_steps = (len(sft_dataset) * sft_trainer.args.num_train_epochs) // (sft_trainer.args.per_device_train_batch_size * sft_trainer.args.gradient_accumulation_steps)
                    print(f"Training for {total_steps} steps")
                    sft_trainer.args.set_evaluate(strategy="steps", steps=total_steps // 5, delay=0)
                    sft_trainer.train()
                else:
                    sft_trainer.data = []

                # eval
                if sft_amount > 0:
                    data.append(group_data(sft_trainer.data))
                print("Length of data after SFT training:", len(data), data[-1] if len(data) > 0 else "empty")
                subsets = eval_dataset.make_subsets()
                for hmm, subset in subsets.items():
                    res = sft_trainer.evaluate(eval_dataset=subset, metric_key_prefix=f'eval_sft_{hmm}', old=True)
                    results[hmm]["sft"] = res[f'eval_sft_{hmm}_loss']

                # print
                if model_args.do_pretrain:
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

                # save trainer logs locally
                sft_trainer.state.save_to_json(f"{training_args.output_dir}/trainer_state_sft.json")

                # save results
                with open(f"{training_args.output_dir}/results_sft.json", "w") as f:
                    f.write(str(results))
                
                # delete stuff to save memory
                del sft_dataset
                del sft_model
                del sft_trainer
        elif training_args.sft_method == "dpo":
            for sft_amount in data_args.num_sft_examples:
                # make dataset
                if sft_amount == 0:
                    continue
                accepted_weights = np.array([float(x) for x in data_args.sft_dist.split(",")])
                rejected_weights = 1 - accepted_weights
                accepted_weights /= accepted_weights.sum()
                rejected_weights /= rejected_weights.sum()
                dpo_model = transformers.AutoModelForCausalLM.from_pretrained(training_args.load_dir).to(device)
                dpo_dataset = HMMPreferenceDataset.make_dataset(
                    hmms=hmms, accepted_dist=accepted_weights, rejected_dist=rejected_weights,
                    num_train_examples=sft_amount, sample_length=1024
                )
                dpo_eval_dataset = HMMPreferenceDataset.make_dataset(
                    hmms=hmms, accepted_dist=accepted_weights, rejected_dist=rejected_weights,
                    num_train_examples=data_args.num_eval_examples, sample_length=1024
                )

                # load model
                dpo_trainer = DPOTrainer(
                    dpo_model,
                    None,
                    args=deepcopy(training_args),
                    train_dataset=dpo_dataset,
                    eval_dataset=dpo_eval_dataset,
                    tokenizer=hmms.tokenizer,
                    data_collator=DataCollator(),
                )
                dpo_trainer.sft_amount = sft_amount
                dpo_trainer.args.num_train_epochs = 1
                dpo_trainer.args.warmup_steps = 0
                dpo_trainer.args.warmup_ratio = 0.1
                dpo_trainer.args.report_to = []
                dpo_trainer.icl_dataset = in_context_datasets
                # dpo_trainer.args.run_name = f"{run_name}_dpo-{sft_amount}"
                total_steps = (len(dpo_dataset) * dpo_trainer.args.num_train_epochs) // (dpo_trainer.args.per_device_train_batch_size * dpo_trainer.args.gradient_accumulation_steps)
                print(f"Training for {total_steps} steps")
                dpo_trainer.args.set_evaluate(strategy="steps", steps=total_steps // 5, delay=0)
                dpo_trainer.train()
                data.append(group_data(dpo_trainer.data))

                # set up trainer
                sft_trainer = SFTTrainer(
                    model=dpo_model,
                    args=deepcopy(training_args),
                    train_dataset=None,
                    eval_dataset=in_context_datasets,
                )
                sft_trainer.sft_amount = sft_amount
                print("Length of data after DPO training:", len(data))
                subsets = eval_dataset.make_subsets()
                for hmm, subset in subsets.items():
                    res = sft_trainer.evaluate(eval_dataset=subset, metric_key_prefix=f'eval_sft_{hmm}', old=True)
                    results[hmm]["sft"] = res[f'eval_sft_{hmm}_loss']

                # print
                if model_args.do_pretrain:
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
                print("Length of data after SFT eval:", len(data))

                # save trainer logs locally
                dpo_trainer.state.save_to_json(f"{training_args.output_dir}/trainer_state_dpo.json")

                # save results
                with open(f"{training_args.output_dir}/results_dpo.json", "w") as f:
                    f.write(str(results))
                
                # delete stuff to save memory
                del dpo_dataset
                del dpo_model
                del dpo_trainer


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
