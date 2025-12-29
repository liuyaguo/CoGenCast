# CAFM

Research codebase for time-series forecasting. The core model is in `models/My_Model.py`, supporting LLM backbones (Qwen2.5/Qwen3) or a pure Transformer encoder, with optional text-summary features for training and inference.

## Requirements

- Python 3.9+ (recommended)
- PyTorch 2.2.2
- Other dependencies in `requirements.txt`

Install:

```bash
pip install -r requirements.txt
pip install transformers
```

Note: the code uses `transformers` (`AutoTokenizer`/`Qwen3Model`) and expects a local model checkpoint directory (see `--llm_path`).

## Data

Default data root is `./datasets` (set by `--root_path`), and the file name is set by `--data_path`.

For text-enhanced datasets (`Dataset_Custom1`), the CSV should contain at least:

- `date` (timestamp)
- feature columns (multivariate) or the target column (univariate)
- `json_summary` (text summary field used as external text features)

For univariate forecasting (`--features S`), extra columns are required:

- `prior_history_avg`
- `start_date`
- `end_date`

Example scripts use text-augmented filenames such as `ETTh1_text.csv`.

## Quick Start (finetune + inference)

Example with ETTh1:

```bash
python -u run.py \
  --task_name finetune \
  --downstream_task forecast \
  --model My_Model \
  --data ETTh1 \
  --root_path ./datasets/ETT-small/ \
  --data_path ETTh1_text.csv \
  --input_len 96 \
  --pred_len 12 \
  --patch_len 6 \
  --stride 6 \
  --batch_size 4 \
  --learning_rate 6e-5 \
  --backbone Qwen3-0.6B \
  --llm_path /path/to/Qwen3-0.6B \
  --text 1
```

Scripts are available under `scripts/`, for example:

```bash
bash scripts/ETTh1.sh
```

## Pretraining

The model supports a `pretrain` task (`Model.forward` includes a `pretrain` branch). If you need a standalone pretraining flow, call `Exp_My_Model.pretrain()` in `exp/exp_my_model.py`, or add a corresponding entry in `run.py`.

## Outputs

- Logs: `./outputs/logs/`
- Finetuned checkpoints: `./outputs/checkpoints/<setting>/checkpoint.pth`
- Pretrained checkpoints: `./outputs/pretrain_checkpoints/<data>/ckpt*.pth`
- Test results: `./outputs/test_results/<data>/`

## Project Structure

- `run.py`: entrypoint and argument config
- `exp/`: training/validation/testing flows
- `models/`: model implementations
- `layers/`: components and modules
- `data_provider/`: data loading and processing
- `scripts/`: training scripts
- `utils/`: metrics, visualization, utilities

## Common Arguments

- `--task_name`: `pretrain` or `finetune`
- `--downstream_task`: `forecast` or `classification`
- `--backbone`: `Qwen3-0.6B` / `Qwen2.5-0.5B` / `Transformer`
- `--llm_path`: local model checkpoint path
- `--text`: enable text features (1 to enable)

For multi-GPU training, launch with `torchrun` and set `LOCAL_RANK/RANK/WORLD_SIZE`; the script will auto-enable DDP.

