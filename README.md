# CoGenCast: A Coupled Autoregressive–Flow Generative Framework for Time Series Forecasting


The core model is in `models/My_Model.py`, supporting LLM backbones Qwen3  for training and inference.



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

Default data root is `./datasets` (set by `--root_path`).

For all datasets , the CSV should contain at least:

- `date` (timestamp)
- feature columns (multivariate) or the target column (univariate)
- `json_summary` (text summary field used as  context features)





## Usage

Example with ETTh1:


Scripts are available under `scripts/`, for example:

```bash
bash scripts/ETTh1.sh
```





