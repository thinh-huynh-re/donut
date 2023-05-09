import datetime
import json
import os
import shutil
from os.path import basename
from typing import List, Optional

from datasets import load_dataset
from PIL import Image
from tap import Tap
from tqdm import tqdm

from config import Config
from donut.model import DonutModel
from lightning_module import DonutModelPLModule


def extract_dataset(config: Config):
    model_module = DonutModelPLModule(config)

    dataset_name_or_path = "naver-clova-ix/cord-v2"
    extracted_dataset_dir = "dataset/extracted_datasets/cord-v2"
    split = "train"

    dir_path = os.path.join(extracted_dataset_dir, split)

    shutil.rmtree(dir_path, ignore_errors=True)
    os.makedirs(dir_path, exist_ok=True)

    task_name = os.path.basename(
        dataset_name_or_path
    )  # e.g., cord-v2, docvqa, rvlcdip, ...

    dataset_name_or_path = dataset_name_or_path
    donut_model: DonutModel = model_module.model
    max_length = config.max_length
    split = split
    task_start_token = f"<s_{task_name}>"
    prompt_end_token = f"<s_{task_name}>"
    sort_json_key = config.sort_json_key
    # prompt_end_token is used for ignoring a given prompt in a loss function
    # for docvqa task, i.e., {"question": {used as a prompt}, "answer": {prediction target}},
    # set prompt_end_token to "<s_answer>"

    dataset = load_dataset(
        dataset_name_or_path,
        split=split,
        cache_dir=os.path.join("dataset", dataset_name_or_path),
        use_auth_token=True,
    )

    for i, sample in tqdm(enumerate(dataset), total=len(dataset)):
        ground_truth = json.loads(sample["ground_truth"])
        if (
            "gt_parses" in ground_truth
        ):  # when multiple ground truths are available, e.g., docvqa
            assert isinstance(ground_truth["gt_parses"], list)
            gt_jsons = ground_truth["gt_parses"]
        else:
            assert "gt_parse" in ground_truth and isinstance(
                ground_truth["gt_parse"], dict
            )
            gt_jsons = [ground_truth["gt_parse"]]

        gt_token_sequences: List[str] = [
            task_start_token
            + donut_model.json2tokenv2(
                gt_json,
                update_special_tokens_for_json_key=split == "train",
                sort_json_key=sort_json_key,
            )
            + donut_model.decoder.tokenizer.eos_token
            for gt_json in gt_jsons  # load json from list of json
        ]
        assert len(gt_token_sequences) == 1, f"len = {len(gt_token_sequences)}"

        gt_token_sequence = gt_token_sequences[0]
        with open(os.path.join(dir_path, f"{i}.html"), "w", encoding="utf8") as f:
            f.write(gt_token_sequence)

        with open(
            os.path.join(dir_path, f"{i}.json"), "w", encoding="utf8"
        ) as json_file:
            json.dump(ground_truth, json_file, ensure_ascii=False, indent=4)

        image: Image.Image = sample["image"]
        image.save(os.path.join(dir_path, f"{i}.png"))

    donut_model.decoder.add_special_tokens([task_start_token, prompt_end_token])
    prompt_end_token_id = donut_model.decoder.tokenizer.convert_tokens_to_ids(
        prompt_end_token
    )
    tokenizer = donut_model.decoder.tokenizer
    print("Done", tokenizer)


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

    extract_dataset(config)
