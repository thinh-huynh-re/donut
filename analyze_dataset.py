import datetime
import os
from os.path import basename
from typing import Optional

from tap import Tap

from config import Config
from donut import DonutDataset
from donut.model import DonutModel
from lightning_module import DonutDataPLModule, DonutModelPLModule


def analyze_dataset(config: Config):
    # pl.utilities.seed.seed_everything(config.get("seed", 42), workers=True)

    model_module = DonutModelPLModule(config)
    data_module = DonutDataPLModule(config)

    # add datasets to data_module
    datasets = {"train": [], "validation": []}
    for i, dataset_name_or_path in enumerate(config.dataset_name_or_paths):
        task_name = os.path.basename(
            dataset_name_or_path
        )  # e.g., cord-v2, docvqa, rvlcdip, ...

        for split in ["train", "validation"]:
            datasets[split].append(
                DonutDataset(
                    dataset_name_or_path=dataset_name_or_path,
                    donut_model=model_module.model,
                    max_length=config.max_length,
                    split=split,
                    task_start_token=config.task_start_tokens[i]
                    if config.get("task_start_tokens", None)
                    else f"<s_{task_name}>",
                    prompt_end_token="<s_answer>"
                    if "docvqa" in dataset_name_or_path
                    else f"<s_{task_name}>",
                    sort_json_key=config.sort_json_key,
                )
            )
            # prompt_end_token is used for ignoring a given prompt in a loss function
            # for docvqa task, i.e., {"question": {used as a prompt}, "answer": {prediction target}},
            # set prompt_end_token to "<s_answer>"
    data_module.train_datasets = datasets["train"]
    data_module.val_datasets = datasets["validation"]

    model: DonutModel = model_module.model
    tokenizer = model.decoder.tokenizer

    for dataset in datasets["train"]:
        for input_tensor, input_ids, labels in dataset:
            print(tokenizer.convert_ids_to_tokens(input_ids))
            break
        break


class ArgumentParser(Tap):
    config: str
    exp_version: Optional[str] = None


if __name__ == "__main__":
    parser = ArgumentParser()
    args, left_argv = parser.parse_known_args()
    args: ArgumentParser

    config = Config(args.config)
    config.argv_update(left_argv)

    config.exp_name = basename(args.config).split(".")[0]
    config.exp_version = (
        datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        if not args.exp_version
        else args.exp_version
    )

    analyze_dataset(config)
