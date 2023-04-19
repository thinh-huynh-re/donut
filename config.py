from typing import Optional

from sconf import Config as SConfig


class Config(SConfig):
    resume_from_checkpoint_path: Optional[bool]
    result_path: str
    pretrained_model_name_or_path: str
    dataset_name_or_paths: list[str]
    sort_json_key: bool
    train_batch_sizes: list[int]
    val_batch_sizes: list[int]
    input_size: list[int]
    max_length: int
    align_long_axis: bool
    num_nodes: int
    seed: int
    lr: float | str
    warmup_steps: int
    num_training_samples_per_epoch: int
    max_epochs: int
    max_steps: int
    num_workers: int
    val_check_interval: float
    check_val_every_n_epoch: int
    gradient_clip_val: float
    verbose: bool

    wandb: bool
    exp_name: str
    exp_version: str
