# How to run

## Extract dataset

```bash
python extract_dataset.py --config config/pretrain/v1.0.1.yaml
```

## Pretrain

```bash
WANDB_MODE=offline WANDB_CACHE_DIR=wandb_cache python trainv2.py --config config/pretrain/v1.0.1-pr.yaml
```
