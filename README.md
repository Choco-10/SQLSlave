# SQL Slave

This project fine-tunes CodeLlama-7B-Instruct with QLoRA on Spider data,
then serves inference through FastAPI and Streamlit. The dataset download,
conversion, and training paths are fixed by the code so the commands stay short.

## Setup

Create the local virtual environment and install dependencies:

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

## Data

Download Spider from the official release and convert it into the fixed local
folders:

```powershell
python download_spider.py
```

This creates:

- `spider\tables.json`
- `spider\train_spider.json`
- `spider\dev.json`
- `converted_spider\train.jsonl`
- `converted_spider\validation.jsonl`

## Train

Fine-tune with QLoRA using the fixed local Spider files and write the adapter
to `artifacts\qlora_adapter`:

```powershell
python train.py
```

During training, the terminal shows:

- `loss` for the optimizer steps
- `eval_loss` and `perplexity` for validation quality
- `exact_match` for SQL string matching after normalization
- runtime and throughput for each evaluation pass

You can tune the exact-match check with:

- `--exact-match-samples 32` to control the small validation sample used during training
- `--exact-match-samples 0` to evaluate the full validation set during training
- `--exact-match-max-new-tokens 128` to limit generation length during the exact-match check

## Serve

Start the API:

```powershell
uvicorn app:app --host 0.0.0.0 --port 8000
```

Start the Streamlit UI in another terminal:

```powershell
streamlit run streamlit_app.py
```

## Check

Run a quick local request against the API:

```powershell
curl -X POST http://127.0.0.1:8000/generate -H "Content-Type: application/json" -d '{"question":"Show total sales by region from orders table"}'
```

Or validate model loading with the test script:

```powershell
python infer_test.py
```

## Docker

Build the containers:

```powershell
docker compose build
```

Start the API and UI:

```powershell
docker compose up api streamlit
```

The API listens on `http://127.0.0.1:8000` and Streamlit on `http://127.0.0.1:8501`.

If you want to point the API at a different adapter directory, set
`ADAPTER_PATH` before starting it.

## Notes

- The API returns only SQL.
- Questions should include a table hint such as `from orders table`.
- `download_spider.py` and `train.py` use the fixed local folders by default.
- `train.py` prints a metric guide at startup so the terminal output is easier to read.
