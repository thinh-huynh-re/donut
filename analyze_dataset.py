import os
from PIL import Image
from datasets import load_dataset
import json

dataset_name_or_path = "naver-clova-ix/cord-v2"
split = "train"

dataset = load_dataset(
    dataset_name_or_path,
    split=split,
    cache_dir=os.path.join("dataset", dataset_name_or_path),
)

sample = dataset[0]
image: Image.Image = sample["image"]
ground_truth: str = sample["ground_truth"]
ground_truth_json = json.loads(ground_truth)


print()
