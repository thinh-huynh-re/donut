import os
from PIL import Image
from datasets import load_dataset
import json, shutil

extracted_dataset_dir = "dataset/extracted_datasets/cord-v2"
dataset_name_or_path = "naver-clova-ix/cord-v2"
split = "train"

dataset = load_dataset(
    dataset_name_or_path,
    split=split,
    cache_dir=os.path.join("dataset", dataset_name_or_path),
)

dir_path = os.path.join(extracted_dataset_dir, split)

shutil.rmtree(dir_path, ignore_errors=True)
os.makedirs(dir_path, exist_ok=True)

for i, sample in enumerate(dataset):
    image: Image.Image = sample["image"]
    ground_truth: str = sample["ground_truth"]
    ground_truth_json = json.loads(ground_truth)
    image.save(os.path.join(dir_path, f"{i}.png"))
    with open(os.path.join(dir_path, f"{i}.json"), "w", encoding="utf8") as json_file:
        json.dump(ground_truth_json, json_file, ensure_ascii=False, indent=4)
