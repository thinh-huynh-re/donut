"""
Donut
Copyright (c) 2022-present NAVER Corp.
MIT License
"""
import json
import os
from typing import Optional

import numpy as np
import torch
from tap import Tap
from tqdm import tqdm
from config import Config

import json
from donut import DonutModel, JSONParseEvaluator, save_json
from donut.util import DonutDataset
from lightning_module import DonutModelPLModule


class ArgumentParser(Tap):
    """
    result_dir: path to experiment result directory
    which has the following structure:
    ├── artifacts.ckpt
    ├── config.yaml
    ├── pytorch_model.bin
    """

    result_dir: str
    # pretrained_model_name_or_path: str
    # dataset_name_or_path: str
    split: Optional[str] = "test"
    task_name: Optional[str] = None  # equals to dataset_name_or_path if None
    save_path: Optional[str] = None


def load_dataset(
    result_dir: str,
    dataset_name_or_path: str,
    model_module: DonutModelPLModule,
    split: str = "test",
):
    task_name = os.path.basename(
        dataset_name_or_path
    )  # e.g., cord-v2, docvqa, rvlcdip, ...

    with open(os.path.join(result_dir, "added_tokens.json"), "r") as f:
        # Reading from json file
        json_object: dict = json.load(f)

    special_tokens = [k for k in json_object.keys()]
    model_module.model.decoder.add_special_tokens(special_tokens)

    return DonutDataset(
        dataset_name_or_path=dataset_name_or_path,
        donut_model=model_module.model,
        max_length=config.max_length,
        split=split,
        task_start_token=f"<s_{task_name}>",
        prompt_end_token=f"<s_{task_name}>",
        sort_json_key=config.sort_json_key,
        local_files_only=config.local_files_only,
    )


def load_checkpoint(path: str):
    checkpoint = torch.load(os.path.join(path, "artifacts.ckpt"), map_location="cpu")
    state_dict = torch.load(os.path.join(path, "pytorch_model.bin"), map_location="cpu")
    checkpoint["state_dict"] = {
        "model." + key: value for key, value in state_dict.items()
    }
    return checkpoint


def test(args: ArgumentParser, config: Config):
    model_module = DonutModelPLModule(config)
    dataset = load_dataset(
        args.result_dir,
        config.dataset_name_or_paths[0],
        model_module,
        split=args.split,
    )
    checkpoint = load_checkpoint(args.result_dir)
    model_module.load_state_dict(checkpoint["state_dict"])
    pretrained_model = model_module.model
    # pretrained_model = DonutModel.from_pretrained(args.pretrained_model_name_or_path)

    if torch.cuda.is_available():
        pretrained_model.half()
        pretrained_model.to("cuda")

    pretrained_model.eval()

    if args.save_path:
        os.makedirs(os.path.dirname(args.save_path), exist_ok=True)

    predictions = []
    ground_truths = []
    accs = []

    evaluator = JSONParseEvaluator()

    for idx, sample in tqdm(enumerate(dataset), total=len(dataset)):
        ground_truth = json.loads(sample["ground_truth"])

        if args.task_name == "docvqa":
            output = pretrained_model.inference(
                image=sample["image"],
                prompt=f"<s_{args.task_name}><s_question>{ground_truth['gt_parses'][0]['question'].lower()}</s_question><s_answer>",
            )["predictions"][0]
        else:
            output = pretrained_model.inference(
                image=sample["image"], prompt=f"<s_{args.task_name}>"
            )["predictions"][0]

        if args.task_name == "rvlcdip":
            gt = ground_truth["gt_parse"]
            score = float(output["class"] == gt["class"])
        elif args.task_name == "docvqa":
            # Note: we evaluated the model on the official website.
            # In this script, an exact-match based score will be returned instead
            gt = ground_truth["gt_parses"]
            answers = set([qa_parse["answer"] for qa_parse in gt])
            score = float(output["answer"] in answers)
        else:
            gt = ground_truth["gt_parse"]
            score = evaluator.cal_acc(output, gt)

        accs.append(score)

        predictions.append(output)
        ground_truths.append(gt)

    scores = {
        "ted_accuracies": accs,
        "ted_accuracy": np.mean(accs),
        "f1_accuracy": evaluator.cal_f1(predictions, ground_truths),
    }
    print(
        f"Total number of samples: {len(accs)}, Tree Edit Distance (TED) based accuracy score: {scores['ted_accuracy']}, F1 accuracy score: {scores['f1_accuracy']}"
    )

    if args.save_path:
        scores["predictions"] = predictions
        scores["ground_truths"] = ground_truths
        save_json(args.save_path, scores)

    return predictions


if __name__ == "__main__":
    # parser = argparse.ArgumentParser()
    # parser.add_argument("--pretrained_model_name_or_path", type=str)
    # parser.add_argument("--dataset_name_or_path", type=str)
    # parser.add_argument("--split", type=str, default="test")
    # parser.add_argument("--task_name", type=str, default=None)
    # parser.add_argument("--save_path", type=str, default=None)
    parser = ArgumentParser()
    args, _ = parser.parse_known_args()
    args: ArgumentParser

    config = Config(os.path.join(args.result_dir, "config.yaml"))

    if args.task_name is None:
        args.task_name = os.path.basename(config.dataset_name_or_paths[0])

    predictions = test(args, config)
