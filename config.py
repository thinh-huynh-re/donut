from typing import Any, Dict, List, Optional, Union

from sconf import Config as SConfig
import yaml


class Config:
    def __init__(self, yaml_path: str):
        self.resume_from_checkpoint_path: Optional[bool]
        self.result_path: str
        self.tokenizer_name_or_path: str  # path to tokenizer
        self.pretrained_model_name_or_path: str  # path to model
        self.dataset_name_or_paths: List[str]
        self.sort_json_key: bool
        self.train_batch_sizes: List[int]
        self.val_batch_sizes: List[int]
        self.input_size: List[int]
        self.max_length: int
        self.align_long_axis: bool
        self.num_nodes: int
        self.splits: List[List[int]]
        self.devices: Union[List[int], str, int]
        self.seed: int
        self.lr: float
        self.warmup_steps: int
        self.num_training_samples_per_epoch: int
        self.max_epochs: int
        self.max_steps: int
        self.num_workers: int
        self.val_check_interval: float
        self.check_val_every_n_epoch: int
        self.accumulate_grad_batches: int
        self.gradient_clip_val: float
        self.verbose: bool
        self.local_files_only: bool
        self.task_start_tokens: List[int]
        self.preload: bool = False
        self.debug_mode: bool = False

        self.wandb: bool
        self.exp_name: str
        self.exp_version: str

        self.max_samples: int

        # set positive number to limit the number of samples (for debug)
        self.train_max_samples: int = -1
        self.validation_max_samples: int = -1

        self.load_yaml(yaml_path)

    def load_yaml(self, yaml_path: str):
        with open(yaml_path, "r") as f:
            data = yaml.safe_load(f)
        self.argv_update(data)

    def argv_update(self, data: Dict[str, Any]):
        self.__dict__.update(data)

    def dumps(self) -> str:
        return yaml.dump(self.asdict())

    def get(self, key: str, default_value: Any) -> Optional[Any]:
        if key in self.__dict__:
            return self.__dict__[key]
        return default_value

    def asdict(self) -> Dict[str, Any]:
        d = dict()
        for k, v in self.__dict__.items():
            if not k.startswith("_"):
                d[k] = v
        return d
