# CoGenCast


Research codebase for time-series forecasting. The core model is in `models/My_Model.py`, supporting LLM backbones Qwen3  for training and inference.



## Requirements

- Python 3.10 (recommended)
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

For text-enhanced datasets , the CSV should contain at least:

- `date` (timestamp)
- feature columns (multivariate) or the target column (univariate)
- `json_summary` (text summary field used as external text features)



Example scripts use text-augmented filenames such as `ETTh1_text.csv`.

## Quick Start (finetune + inference)

Example with ETTh1:


Scripts are available under `scripts/`, for example:

```bash
bash scripts/ETTh1.sh
```





