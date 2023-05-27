"""
Donut
Copyright (c) 2022-present NAVER Corp.
MIT License
"""
import datetime
import os
from logging import Logger
from os.path import basename
from pathlib import Path
from typing import Any, Dict, List, Optional

import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.loggers.tensorboard import TensorBoardLogger
from pytorch_lightning.plugins import CheckpointIO
from pytorch_lightning.utilities import rank_zero_only
from tap import Tap

from config import Config
from donut import DonutDataset
from donut.util import DonutDatasetV2
from lightning_module import DonutDataPLModule, DonutModelPLModule
from lightning_fabric.utilities.types import _PATH

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512"


class CustomCheckpointIO(CheckpointIO):
    def save_checkpoint(
        self,
        checkpoint: Dict[str, Any],
        path: _PATH,
        storage_options: Optional[Any] = None,
    ):
        del checkpoint["state_dict"]
        torch.save(checkpoint, path)

    def load_checkpoint(self, path: _PATH, map_location: Optional[Any] = None):
        checkpoint = torch.load(path + "artifacts.ckpt")
        state_dict = torch.load(path + "pytorch_model.bin")
        checkpoint["state_dict"] = {
            "model." + key: value for key, value in state_dict.items()
        }
        return checkpoint

    def remove_checkpoint(self, path: _PATH) -> None:
        return super().remove_checkpoint(path)


@rank_zero_only
def save_config_file(config: Config, path: str) -> None:
    if not Path(path).exists():
        os.makedirs(path)
    save_path = Path(path) / "config.yaml"
    print(config.dumps())
    with open(save_path, "w") as f:
        f.write(config.dumps(modified_color=None, quote_str=True))
        print(f"Config is saved at {save_path}")


def train(config: Config):
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
                DonutDatasetV2(
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
                    preload=config.preload,
                )
            )
            # prompt_end_token is used for ignoring a given prompt in a loss function
            # for docvqa task, i.e., {"question": {used as a prompt}, "answer": {prediction target}},
            # set prompt_end_token to "<s_answer>"
    data_module.train_datasets = datasets["train"]
    data_module.val_datasets = datasets["validation"]

    print("Num. tokens", len(model_module.model.decoder.tokenizer))

    loggers: List[Logger] = []
    tb_logger = TensorBoardLogger(
        save_dir=config.result_path,
        name=config.exp_name,
        version=config.exp_version,
        default_hp_metric=False,
    )
    loggers.append(tb_logger)

    if config.get("wandb", False):
        wb_logger = WandbLogger(
            project=config.exp_name,
            name=config.exp_version,
            save_dir=config.result_path,
            config=config.asdict(),
            log_model=True,
        )
        loggers.append(wb_logger)

    lr_callback = LearningRateMonitor(logging_interval="step")

    checkpoint_callback = ModelCheckpoint(
        monitor="val_metric",
        dirpath=Path(config.result_path) / config.exp_name / config.exp_version,
        filename="artifacts",
        save_top_k=1,
        save_last=False,
        mode="min",
    )

    custom_ckpt = CustomCheckpointIO()
    trainer = pl.Trainer(
        resume_from_checkpoint=config.get("resume_from_checkpoint_path", None),
        # num_nodes=config.get("num_nodes", 1),
        # gpus=torch.cuda.device_count(),
        strategy="ddp",
        accelerator="gpu",
        plugins=custom_ckpt,
        max_epochs=config.max_epochs,
        max_steps=config.max_steps,
        devices=config.devices,
        val_check_interval=config.val_check_interval,
        check_val_every_n_epoch=config.check_val_every_n_epoch,
        gradient_clip_val=config.gradient_clip_val,
        precision=16,
        num_sanity_val_steps=0,
        logger=loggers,
        callbacks=[lr_callback, checkpoint_callback],
    )

    trainer.fit(model_module, data_module)


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

    assert (
        len(config.dataset_name_or_paths)
        == len(config.splits)
        == len(config.task_start_tokens)
    )

    save_config_file(
        config, Path(config.result_path) / config.exp_name / config.exp_version
    )
    train(config)
