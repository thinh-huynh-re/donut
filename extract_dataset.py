import datetime
import json
import os
import shutil
from os.path import basename
from typing import Dict, List, Optional

from pprint import pprint
from datasets import load_dataset
from PIL import Image
from tap import Tap
from tqdm import tqdm

from config import Config
from donut.model import DonutModel
from lightning_module import DonutModelPLModule

import numpy as np
import threading

"""
Extract HuggingFace (HF) datasets into directories/files structure format

dataset
|- extracted_datasets
    |- cord-v2
        |- train.json
        |- train
            |- x.png
        |- valiation.json
        |- validation
            |- x.png
        |- test.json
        |- test
            |- x.png
        

*Note*: Internet connection is required!

Conventionally, the format of HF datasets is as follows:
1, `split` can be 'train', 'validation', 'test'
2, each record is a dictionary with the following keys
    {
        "image": PIL.Image.Image,
        "ground_truth": str
    }

We convert these with the following criteria:
1, Each image is located in a separated file
2, A metadata specifies details about the dataset:
    {
        # before processing all records in the dataset
        "before": {
            "vocab_size": int,
            "special_tokens": List[str],
        },
        "data": [
            {
                "gt_token_sequences": ["str",...], # each image can contain multiple ground truth
                "image_path": str
            },
            ...
        ],
        # after processing all records in the dataset, 
        # there are some new additional special tokens
        "after": {
            "vocab_size": int,
            "special_tokens": List[str],
        },
    }
"""


def ground_truth_to_gt_json(sample: Dict[str, str]) -> List[Dict]:
    ground_truth = json.loads(sample["ground_truth"])
    if (
        "gt_parses" in ground_truth
    ):  # when multiple ground truths are available, e.g., docvqa
        assert isinstance(ground_truth["gt_parses"], list)
        gt_jsons = ground_truth["gt_parses"]
    else:
        assert "gt_parse" in ground_truth and isinstance(ground_truth["gt_parse"], dict)
        gt_jsons = [ground_truth["gt_parse"]]
    return gt_jsons


def thread_fn(
    data: List[Dict],
    dataset,
    indices: List[int],
    dir_path: str,
    donut_model: DonutModel,
    sort_json_key: bool,
    split: str,
):
    for i in tqdm(indices):
        sample = dataset[int(i)]
        gt_jsons = ground_truth_to_gt_json(sample)

        image_path = os.path.join(dir_path, f"{i}.png")

        image: Image.Image = sample["image"]
        image.save(image_path)

        data.append(
            {
                "gt_token_sequences": [
                    donut_model.json2token_v2(
                        gt_json,
                        # update_special_tokens_for_json_key=split == "train",
                        update_special_tokens_for_json_key=True,
                        sort_json_key=sort_json_key,
                    )
                    for gt_json in gt_jsons
                ],
                "image_path": f"{split}/{i}.png",
            }
        )


NUM_THREADS = 6


def extract_dataset(config: Config):
    for i in range(len(config.dataset_name_or_paths)):
        # "naver-clova-ix/cord-v2"
        dataset_name_or_path = config.dataset_name_or_paths[i]

        # "dataset/extracted_datasets/cord-v2"
        extracted_dataset_dir = os.path.join(
            "dataset/extracted_datasets", os.path.basename(dataset_name_or_path)
        )

        model_module = DonutModelPLModule(config)
        donut_model: DonutModel = model_module.model

        init_vocab = donut_model.decoder.tokenizer.get_vocab()
        metadata = {
            "before": {
                "vocab_size": len(donut_model.decoder.tokenizer),
                # "special_tokens": donut_model.decoder.tokenizer.all_special_tokens,
            },
            "splits": {
                # "train": {
                #   "data": [{
                #       "gt_token_sequences": List[str],
                #       "image_path": str,
                #   }],
                #   "length": int
                # }
            },
            "after": {
                "vocab_size": None,
                # "special_tokens": None,
                "additional_special_tokens": [],
            },
            # "added_tokens": [],
        }

        for split in config.splits[i]:
            dir_path = os.path.join(extracted_dataset_dir, split)

            shutil.rmtree(dir_path, ignore_errors=True)
            os.makedirs(dir_path, exist_ok=True)

            sort_json_key = config.sort_json_key

            dataset = load_dataset(
                dataset_name_or_path,
                split=split,
                cache_dir=os.path.join("dataset", dataset_name_or_path),
                use_auth_token=not config.local_files_only,
            )
            print(type(dataset))

            data = []

            threads = []
            for i, indices in enumerate(
                np.array_split(list(range(len(dataset))), NUM_THREADS)
            ):
                thread = threading.Thread(
                    target=thread_fn,
                    args=(
                        data,
                        dataset,
                        indices,
                        dir_path,
                        donut_model,
                        sort_json_key,
                        split,
                    ),
                )
                threads.append(thread)
                thread.start()

            for thread in threads:
                thread.join()

            metadata["splits"][split] = {
                "data": data,
                "length": len(data),
            }

        metadata["after"]["vocab_size"] = len(donut_model.decoder.tokenizer)
        # metadata["after"][
        #     "special_tokens"
        # ] = donut_model.decoder.tokenizer.all_special_tokens

        additional_special_tokens = list(donut_model.additional_special_tokens)
        metadata["after"]["additional_special_tokens"] = {
            "items": additional_special_tokens,
            "length": len(additional_special_tokens),
        }
        # added_tokens = list(
        #     set(donut_model.decoder.tokenizer.get_vocab()).difference(init_vocab)
        # )
        # metadata["added_tokens"] = {
        #     "items": added_tokens,
        #     "length": len(added_tokens),
        # }

        with open(
            os.path.join(extracted_dataset_dir, f"metadata.json"),
            "w",
            encoding="utf8",
        ) as json_file:
            json.dump(metadata, json_file, ensure_ascii=False, indent=4)


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

    # Validation
    assert len(config.dataset_name_or_paths) == len(config.splits)
    pprint(config.asdict())

    extract_dataset(config)
