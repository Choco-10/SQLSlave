# SQLAgent

You ask a question in plain English, describe your database tables and columns, and SQLAgent generates the SQL query for you. It uses a DeepSeek-Coder-6.7B-Instruct model fine-tuned on the Spider text-to-SQL dataset with QLoRA.

## Setup

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

## Data

```powershell
python -m cli.download_spider
```

Downloads the Spider dataset and converts it into:

- `spider\tables.json`
- `spider\train_spider.json`
- `spider\dev.json`
- `converted_spider\train.jsonl`
- `converted_spider\validation.jsonl`

## Train

```powershell
python -m cli.train
```

Runs QLoRA fine-tuning for 1 epoch and saves the adapter to `artifacts\qlora_adapter`.

Optional flags: `--num-train-epochs`, `--learning-rate`, `--eval-steps`.

## Evaluate

```powershell
python -m cli.eval_accuracy
```

Runs execution and exact-match accuracy on the validation set.

## Serve

```powershell
uvicorn api.app:app --host 0.0.0.0 --port 8000
```

Open another terminal:

```powershell
streamlit run ui/streamlit_app.py
```

Open `http://127.0.0.1:8501` in your browser. Enter your question, define the schema (tables, columns, PKs, FKs), and generate SQL.

Set `$env:ADAPTER_PATH` to use a custom adapter path.

## Check

```powershell
curl -X POST http://127.0.0.1:8000/generate -H "Content-Type: application/json" -d "{\"question\":\"Show total sales by region from orders table\",\"schema\":[{\"name\":\"orders\",\"columns\":{\"region\":{},\"amount\":{}}}]}"
```

## Docker

### Prerequisites

- [Docker Desktop](https://www.docker.com/products/docker-desktop/)
- [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html) for GPU passthrough

### Build

```powershell
docker compose build
```

### Serve API (FastAPI)

```powershell
docker compose up
```

The API is available at `http://localhost:8000`. Open `http://localhost:8000/docs` for the interactive Swagger UI.

### Serve UI (Streamlit)

```powershell
docker compose run --service-ports app streamlit run ui/streamlit_app.py --server.port 8501
```

Open `http://localhost:8501` in your browser.

### Train (inside container)

```powershell
docker compose run app python cli/train.py
```

The QLoRA adapter is saved to `artifacts/qlora_adapter/` — this directory is **volume-mounted** (`./artifacts:/app/artifacts`), so the trained weights persist on your host machine and are immediately available to the API without rebuilding the image.

### Override model / adapter

Set environment variables to use a different base model or adapter path:

```powershell
docker compose run -e BASE_MODEL_ID="deepseek-ai/deepseek-coder-6.7b-instruct" -e ADAPTER_PATH="artifacts/qlora_adapter" app
```
