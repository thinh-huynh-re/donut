# How to setup training Receipts

## Prepare dataset

```
├── dataset
│   ├── DocumentUnderstanding
│   │   ├── receipts
│   │   │   ├── test
│   │   │   │   ├── *.[jpg|png|jpeg]
│   │   │   │   ├── *.json
│   │   │   ├── train
│   │   │   │   ├── *.[jpg|png|jpeg]
│   │   │   │   ├── *.json
│   │   │   ├── validation
│   │   │   │   ├── *.[jpg|png|jpeg]
│   │   │   │   ├── *.json
```

## Prepare model

Just run this command for the first time. It will download and cache all the required models into `model` directory

```
python analyze_dataset.py --config config/train_receipts.yaml
```

For example, if the model is using pretrained weights from `naver-clova-ix/donut-base` and tokenizer `xlm-roberta-base`. The directory structure of `model` will look like this:

```
├── model
│   ├── naver-clova-ix
│   ├── xlm-roberta-base
```

Note: No need to understand what inside the directory `naver-clova-ix` or `xlm-roberta-base` since it is the format of HuggingFace.

## Make sure cached models and cached datasets are available

Run this command again but with different config file `train_receipts_local.yaml`

```
python analyze_dataset.py --config config/train_receipts_local.yaml
```

## Set up Python environments

### Virtual environment

```
python3.10 -m venv env
source env/bin/activate
pip install -r requirements.txt
```

### Conda environment

```
conda create -n <env_name> python==3.10
conda activate <env_name>
pip install -r requirements.txt
```

## Train

### Normal training (internet connection required)

```
python train.py --config config/train_receipts.yaml
```

### Train using cached datasets and cached models only (no internet connection required)

```
python train.py --config config/train_receipts_local.yaml --devices [2]
```

### Sync wandb

```
wandb sync <log_dir>
```

## Pretrain

```
python pretrain.py --config pretrain/v1.0.1-pr.yaml
```