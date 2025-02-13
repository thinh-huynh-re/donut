"""
Donut
Copyright (c) 2022-present NAVER Corp.
MIT License
"""
import json
import os
import random
from typing import Any, Dict, List, Tuple, Union

import torch
import zss
from datasets import load_dataset
from nltk import edit_distance
from torch.utils.data import Dataset
from zss import Node
import itertools
import glob
from tqdm import tqdm
from PIL import Image
import copy
from PIL import ImageFile

from donut.model import DonutModel


def save_json(write_path: Union[str, bytes, os.PathLike], save_obj: Any):
    with open(write_path, "w") as f:
        json.dump(save_obj, f)


def load_json(json_path: Union[str, bytes, os.PathLike]):
    with open(json_path, mode="r", encoding="utf-8") as f:
        return json.load(f)


class DonutRawDatasetV1(Dataset):
    """
    Directory format:
        train
        - <name>.<jpg|png|jpeg>
        - <name>.json
        ...
    """

    def __init__(
        self,
        dataset_name_or_path: str = "dataset/DocumentUnderstanding/receipts",
        split: str = "train",
    ):
        ImageFile.LOAD_TRUNCATED_IMAGES = True
        exts = ["jpg", "png", "jpeg"]

        dataset_dir = os.path.join("dataset", dataset_name_or_path, split)

        image_paths = list(
            itertools.chain(*[glob.glob(f"{dataset_dir}/*.{ext}") for ext in exts])
        )
        json_paths = glob.glob(f"{dataset_dir}/*.json")

        self.metadata: List[Dict[str, Any]] = []

        for image_path in tqdm(image_paths):
            name = os.path.basename(image_path).split(".")[0]
            json_path = f"{dataset_dir}/{name}.json"

            # Make sure each image has exactly one corresponding json file
            if json_path not in json_paths:
                raise Exception(f"JSON not found: {json_path}")

            data = load_json(json_path)
            del data["shapes"]
            ground_truth = json.dumps(data, ensure_ascii=False)

            self.metadata.append(
                dict(image=Image.open(image_path), ground_truth=ground_truth)
            )

    def __len__(self) -> int:
        return len(self.metadata)

    def __getitem__(self, idx: int) -> Tuple[Dict[str, Any]]:
        return self.metadata[idx]


class DonutRawDatasetV2(Dataset):
    """
    Directory format:
        metadata.json
        train
        - <name>.<jpg|png|jpeg>
        validation
        - <name>.<jpg|png|jpeg>
        test
        - <name>.<jpg|png|jpeg>
    """

    def __init__(
        self,
        task_start_token: str,
        eos_token: str,
        dataset_name_or_path: str = "dataset/DocumentUnderstanding/receipts",
        split: str = "train",
        preload: bool = False,
    ):
        self.dataset_dir = os.path.join("dataset", dataset_name_or_path)

        raw_metadata = load_json(os.path.join(self.dataset_dir, "metadata.json"))

        """
        [
            {
                "gt_token_sequences": [
                    "<s_menu><s_nm>Nasi Campur Bali</s_nm><s_cnt>1 x</s_cnt><s_price>75,000</s_price><sep/><s_nm>Bbk Bengil Nasi</s_nm><s_cnt>1 x</s_cnt><s_price>125,000</s_price><sep/><s_nm>MilkShake Starwb</s_nm><s_cnt>1 x</s_cnt><s_price>37,000</s_price><sep/><s_nm>Ice Lemon Tea</s_nm><s_cnt>1 x</s_cnt><s_price>24,000</s_price><sep/><s_nm>Nasi Ayam Dewata</s_nm><s_cnt>1 x</s_cnt><s_price>70,000</s_price><sep/><s_nm>Free Ice Tea</s_nm><s_cnt>3 x</s_cnt><s_price>0</s_price><sep/><s_nm>Organic Green Sa</s_nm><s_cnt>1 x</s_cnt><s_price>65,000</s_price><sep/><s_nm>Ice Tea</s_nm><s_cnt>1 x</s_cnt><s_price>18,000</s_price><sep/><s_nm>Ice Orange</s_nm><s_cnt>1 x</s_cnt><s_price>29,000</s_price><sep/><s_nm>Ayam Suir Bali</s_nm><s_cnt>1 x</s_cnt><s_price>85,000</s_price><sep/><s_nm>Tahu Goreng</s_nm><s_cnt>2 x</s_cnt><s_price>36,000</s_price><sep/><s_nm>Tempe Goreng</s_nm><s_cnt>2 x</s_cnt><s_price>36,000</s_price><sep/><s_nm>Tahu Telor Asin</s_nm><s_cnt>1 x</s_cnt><s_price>40,000.</s_price><sep/><s_nm>Nasi Goreng Samb</s_nm><s_cnt>1 x</s_cnt><s_price>70,000</s_price><sep/><s_nm>Bbk Panggang Sam</s_nm><s_cnt>3 x</s_cnt><s_price>366,000</s_price><sep/><s_nm>Ayam Sambal Hija</s_nm><s_cnt>1 x</s_cnt><s_price>92,000</s_price><sep/><s_nm>Hot Tea</s_nm><s_cnt>2 x</s_cnt><s_price>44,000</s_price><sep/><s_nm>Ice Kopi</s_nm><s_cnt>1 x</s_cnt><s_price>32,000</s_price><sep/><s_nm>Tahu Telor Asin</s_nm><s_cnt>1 x</s_cnt><s_price>40,000</s_price><sep/><s_nm>Free Ice Tea</s_nm><s_cnt>1 x</s_cnt><s_price>0</s_price><sep/><s_nm>Bebek Street</s_nm><s_cnt>1 x</s_cnt><s_price>44,000</s_price><sep/><s_nm>Ice Tea Tawar</s_nm><s_cnt>1 x</s_cnt><s_price>18,000</s_price></s_menu><s_sub_total><s_subtotal_price>1,346,000</s_subtotal_price><s_service_price>100,950</s_service_price><s_tax_price>144,695</s_tax_price><s_etc>-45</s_etc></s_sub_total><s_total><s_total_price>1,591,600</s_total_price></s_total>"
                ],
                "image_path": "train/0.png"
            },
            ...
        ]
        """
        self.metadata: List[Dict[str, Any]] = raw_metadata["splits"][split]["data"]

        """
        [
            "<s_subtotal_price>",
            "<s_sub_total>",
            "<s_itemsubtotal>",
            ...
        ]
        """
        self.additional_special_tokens: List[str] = raw_metadata["after"][
            "additional_special_tokens"
        ]["items"]

        # Add <task_start_token> at the beginning of the sentence
        # and add <eos_token> at the end of the sentence
        for d in tqdm(self.metadata):
            d["gt_token_sequences"] = [
                task_start_token + gt_token_sequence + eos_token
                for gt_token_sequence in d["gt_token_sequences"]
            ]

        # Preload images in-memory. Make sure you have enough RAM.when preload=True
        self.preload = preload
        if preload:
            for d in self.metadata:
                d["image"] = Image.open(os.path.join(self.dataset_dir, d["image_path"]))

    def __len__(self) -> int:
        return len(self.metadata)

    def __getitem__(self, idx: int) -> Tuple[Dict[str, Any]]:
        """
        {
            "image": PIL.Image.Image,
            "image_path": str,
            "gt_token_sequences": List[str]
        }
        """
        if self.preload:
            return self.metadata[idx]
        else:
            data = copy.deepcopy(self.metadata[idx])
            data["image"] = Image.open(
                os.path.join(self.dataset_dir, data["image_path"])
            )
            return data


class DonutDataset(Dataset):
    """
    DonutDataset which is saved in huggingface datasets format. (see details in https://huggingface.co/docs/datasets)
    Each row, consists of image path(png/jpg/jpeg) and gt data (json/jsonl/txt),
    and it will be converted into input_tensor(vectorized image) and input_ids(tokenized string)

    Args:
        dataset_name_or_path: name of dataset (available at huggingface.co/datasets) or the path containing image files and metadata.jsonl
        ignore_id: ignore_index for torch.nn.CrossEntropyLoss
        task_start_token: the special token to be fed to the decoder to conduct the target task
    """

    def __init__(
        self,
        dataset_name_or_path: str,
        donut_model: DonutModel,
        max_length: int,
        split: str = "train",
        ignore_id: int = -100,
        task_start_token: str = "<s>",
        prompt_end_token: str = None,
        sort_json_key: bool = True,
        local_files_only: bool = False,
    ):
        super().__init__()

        self.donut_model: DonutModel = donut_model
        self.max_length = max_length
        self.split = split
        self.ignore_id = ignore_id
        self.task_start_token = task_start_token
        self.prompt_end_token = (
            prompt_end_token if prompt_end_token else task_start_token
        )
        self.sort_json_key = sort_json_key

        if local_files_only:
            self.dataset = DonutRawDatasetV1(dataset_name_or_path, split)
        else:
            self.dataset = load_dataset(
                dataset_name_or_path,
                split=self.split,
                cache_dir=None
                if dataset_name_or_path is None
                else os.path.join("dataset", dataset_name_or_path),
                use_auth_token=not local_files_only,
            )
        self.dataset_length = len(self.dataset)

        self.gt_token_sequences = []
        self.gt_jsons_list = []  # TODO: remove this
        for i, sample in enumerate(tqdm(self.dataset)):
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

            self.gt_jsons_list.append(gt_jsons)  # TODO: remove this

            self.gt_token_sequences.append(
                [
                    task_start_token
                    + self.donut_model.json2token(
                        gt_json,
                        update_special_tokens_for_json_key=self.split == "train",
                        sort_json_key=self.sort_json_key,
                    )
                    + self.donut_model.decoder.tokenizer.eos_token
                    for gt_json in gt_jsons  # load json from list of json
                ]
            )

        self.donut_model.decoder.add_special_tokens(
            [self.task_start_token, self.prompt_end_token]
        )
        self.prompt_end_token_id = (
            self.donut_model.decoder.tokenizer.convert_tokens_to_ids(
                self.prompt_end_token
            )
        )

    def __len__(self) -> int:
        return self.dataset_length

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Load image from image_path of given dataset_path and convert into input_tensor and labels.
        Convert gt data into input_ids (tokenized string)

        Returns:
            input_tensor : preprocessed image
            input_ids : tokenized gt_data
            labels : masked labels (model doesn't need to predict prompt and pad token)
        """
        sample = self.dataset[idx]

        # input_tensor
        input_tensor = self.donut_model.encoder.prepare_input(
            sample["image"], random_padding=self.split == "train"
        )

        # input_ids
        processed_parse = random.choice(
            self.gt_token_sequences[idx]
        )  # can be more than one, e.g., DocVQA Task 1
        input_ids = self.donut_model.decoder.tokenizer(
            processed_parse,
            add_special_tokens=False,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )["input_ids"].squeeze(0)

        if self.split == "train":
            labels = input_ids.clone()
            labels[
                labels == self.donut_model.decoder.tokenizer.pad_token_id
            ] = self.ignore_id  # model doesn't need to predict pad token
            labels[
                : torch.nonzero(labels == self.prompt_end_token_id).sum() + 1
            ] = self.ignore_id  # model doesn't need to predict prompt (for VQA)
            return input_tensor, input_ids, labels
        else:
            prompt_end_index = torch.nonzero(
                input_ids == self.prompt_end_token_id
            ).sum()  # return prompt end index instead of target output labels
            return input_tensor, input_ids, prompt_end_index, processed_parse


class DonutDatasetV2(Dataset):
    """
    DonutDataset which is saved in huggingface datasets format. (see details in https://huggingface.co/docs/datasets)
    Each row, consists of image path(png/jpg/jpeg) and gt data (json/jsonl/txt),
    and it will be converted into input_tensor(vectorized image) and input_ids(tokenized string)

    Args:
        dataset_name_or_path: name of dataset (available at huggingface.co/datasets) or the path containing image files and metadata.jsonl
        ignore_id: ignore_index for torch.nn.CrossEntropyLoss
        task_start_token: the special token to be fed to the decoder to conduct the target task
    """

    def __init__(
        self,
        dataset_name_or_path: str,
        donut_model: DonutModel,
        max_length: int,
        split: str = "train",
        ignore_id: int = -100,
        task_start_token: str = "<s>",
        prompt_end_token: str = None,
        sort_json_key: bool = True,
        preload: bool = False,
        debug_mode: bool = False,
        max_samples: int = -1,
        data_augmentation: bool = False,
    ):
        super().__init__()
        self.data_augmentation = data_augmentation

        self.debug_mode = debug_mode
        self.donut_model: DonutModel = donut_model
        self.max_length = max_length
        self.split = split
        self.ignore_id = ignore_id
        self.task_start_token = task_start_token
        self.prompt_end_token = (
            prompt_end_token if prompt_end_token else task_start_token
        )
        self.sort_json_key = sort_json_key

        self.dataset = DonutRawDatasetV2(
            task_start_token=task_start_token,
            eos_token=self.donut_model.decoder.tokenizer.eos_token,
            dataset_name_or_path=dataset_name_or_path,
            split=split,
            preload=preload,
        )
        self.dataset_length = len(self.dataset) if max_samples < 0 else max_samples

        self.donut_model.decoder.add_special_tokens(
            self.dataset.additional_special_tokens
            + [self.task_start_token, self.prompt_end_token]
        )
        self.prompt_end_token_id = (
            self.donut_model.decoder.tokenizer.convert_tokens_to_ids(
                self.prompt_end_token
            )
        )

    def __len__(self) -> int:
        return self.dataset_length

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Load image from image_path of given dataset_path and convert into input_tensor and labels.
        Convert gt data into input_ids (tokenized string)

        Returns:
            input_tensor : preprocessed image
            input_ids : tokenized gt_data
            labels : masked labels (model doesn't need to predict prompt and pad token)
        """
        sample = self.dataset[idx]

        # input_tensor
        input_tensor = self.donut_model.encoder.prepare_input(
            sample["image"],
            random_padding=self.split == "train",
            data_augmentation=self.data_augmentation,
            debug_mode=self.debug_mode,
        )

        # input_ids
        processed_parse = random.choice(
            sample["gt_token_sequences"]
        )  # can be more than one, e.g., DocVQA Task 1
        input_ids = self.donut_model.decoder.tokenizer(
            processed_parse,
            add_special_tokens=False,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )["input_ids"].squeeze(0)

        if self.split == "train":
            labels = input_ids.clone()
            labels[
                labels == self.donut_model.decoder.tokenizer.pad_token_id
            ] = self.ignore_id  # model doesn't need to predict pad token
            labels[
                : torch.nonzero(labels == self.prompt_end_token_id).sum() + 1
            ] = self.ignore_id  # model doesn't need to predict prompt (for VQA)
            return input_tensor, input_ids, labels
        else:
            prompt_end_index = torch.nonzero(
                input_ids == self.prompt_end_token_id
            ).sum()  # return prompt end index instead of target output labels
            return input_tensor, input_ids, prompt_end_index, processed_parse


class JSONParseEvaluator:
    """
    Calculate n-TED(Normalized Tree Edit Distance) based accuracy and F1 accuracy score
    """

    @staticmethod
    def flatten(data: dict):
        """
        Convert Dictionary into Non-nested Dictionary
        Example:
            input(dict)
                {
                    "menu": [
                        {"name" : ["cake"], "count" : ["2"]},
                        {"name" : ["juice"], "count" : ["1"]},
                    ]
                }
            output(list)
                [
                    ("menu.name", "cake"),
                    ("menu.count", "2"),
                    ("menu.name", "juice"),
                    ("menu.count", "1"),
                ]
        """
        flatten_data = list()

        def _flatten(value, key=""):
            if type(value) is dict:
                for child_key, child_value in value.items():
                    _flatten(child_value, f"{key}.{child_key}" if key else child_key)
            elif type(value) is list:
                for value_item in value:
                    _flatten(value_item, key)
            else:
                flatten_data.append((key, value))

        _flatten(data)
        return flatten_data

    @staticmethod
    def update_cost(node1: Node, node2: Node):
        """
        Update cost for tree edit distance.
        If both are leaf node, calculate string edit distance between two labels (special token '<leaf>' will be ignored).
        If one of them is leaf node, cost is length of string in leaf node + 1.
        If neither are leaf node, cost is 0 if label1 is same with label2 othewise 1
        """
        label1 = node1.label
        label2 = node2.label
        label1_leaf = "<leaf>" in label1
        label2_leaf = "<leaf>" in label2
        if label1_leaf == True and label2_leaf == True:
            return edit_distance(
                label1.replace("<leaf>", ""), label2.replace("<leaf>", "")
            )
        elif label1_leaf == False and label2_leaf == True:
            return 1 + len(label2.replace("<leaf>", ""))
        elif label1_leaf == True and label2_leaf == False:
            return 1 + len(label1.replace("<leaf>", ""))
        else:
            return int(label1 != label2)

    @staticmethod
    def insert_and_remove_cost(node: Node):
        """
        Insert and remove cost for tree edit distance.
        If leaf node, cost is length of label name.
        Otherwise, 1
        """
        label = node.label
        if "<leaf>" in label:
            return len(label.replace("<leaf>", ""))
        else:
            return 1

    def normalize_dict(self, data: Union[Dict, List, Any]):
        """
        Sort by value, while iterate over element if data is list
        """
        if not data:
            return {}

        if isinstance(data, dict):
            new_data = dict()
            for key in sorted(data.keys(), key=lambda k: (len(k), k)):
                value = self.normalize_dict(data[key])
                if value:
                    if not isinstance(value, list):
                        value = [value]
                    new_data[key] = value

        elif isinstance(data, list):
            if all(isinstance(item, dict) for item in data):
                new_data = []
                for item in data:
                    item = self.normalize_dict(item)
                    if item:
                        new_data.append(item)
            else:
                new_data = [
                    str(item).strip()
                    for item in data
                    if type(item) in {str, int, float} and str(item).strip()
                ]
        else:
            new_data = [str(data).strip()]

        return new_data

    def cal_f1(self, preds: List[dict], answers: List[dict]):
        """
        Calculate global F1 accuracy score (field-level, micro-averaged) by counting all true positives, false negatives and false positives
        """
        total_tp, total_fn_or_fp = 0, 0
        for pred, answer in zip(preds, answers):
            pred, answer = self.flatten(self.normalize_dict(pred)), self.flatten(
                self.normalize_dict(answer)
            )
            for field in pred:
                if field in answer:
                    total_tp += 1
                    answer.remove(field)
                else:
                    total_fn_or_fp += 1
            total_fn_or_fp += len(answer)
        return total_tp / (total_tp + total_fn_or_fp / 2)

    def construct_tree_from_dict(self, data: Union[Dict, List], node_name: str = None):
        """
        Convert Dictionary into Tree

        Example:
            input(dict)

                {
                    "menu": [
                        {"name" : ["cake"], "count" : ["2"]},
                        {"name" : ["juice"], "count" : ["1"]},
                    ]
                }

            output(tree)
                                     <root>
                                       |
                                     menu
                                    /    \
                             <subtree>  <subtree>
                            /      |     |      \
                         name    count  name    count
                        /         |     |         \
                  <leaf>cake  <leaf>2  <leaf>juice  <leaf>1
         """
        if node_name is None:
            node_name = "<root>"

        node = Node(node_name)

        if isinstance(data, dict):
            for key, value in data.items():
                kid_node = self.construct_tree_from_dict(value, key)
                node.addkid(kid_node)
        elif isinstance(data, list):
            if all(isinstance(item, dict) for item in data):
                for item in data:
                    kid_node = self.construct_tree_from_dict(
                        item,
                        "<subtree>",
                    )
                    node.addkid(kid_node)
            else:
                for item in data:
                    node.addkid(Node(f"<leaf>{item}"))
        else:
            raise Exception(data, node_name)
        return node

    def cal_acc(self, pred: dict, answer: dict):
        """
        Calculate normalized tree edit distance(nTED) based accuracy.
        1) Construct tree from dict,
        2) Get tree distance with insert/remove/update cost,
        3) Divide distance with GT tree size (i.e., nTED),
        4) Calculate nTED based accuracy. (= max(1 - nTED, 0 ).
        """
        pred = self.construct_tree_from_dict(self.normalize_dict(pred))
        answer = self.construct_tree_from_dict(self.normalize_dict(answer))
        return max(
            0,
            1
            - (
                zss.distance(
                    pred,
                    answer,
                    get_children=zss.Node.get_children,
                    insert_cost=self.insert_and_remove_cost,
                    remove_cost=self.insert_and_remove_cost,
                    update_cost=self.update_cost,
                    return_operations=False,
                )
                / zss.distance(
                    self.construct_tree_from_dict(self.normalize_dict({})),
                    answer,
                    get_children=zss.Node.get_children,
                    insert_cost=self.insert_and_remove_cost,
                    remove_cost=self.insert_and_remove_cost,
                    update_cost=self.update_cost,
                    return_operations=False,
                )
            ),
        )
