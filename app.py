from __future__ import annotations

import os

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

try:
    from .model import load_sql_generator
except ImportError:  # pragma: no cover - allows direct script execution
    from model import load_sql_generator


class GenerateRequest(BaseModel):
    question: str


class GenerateResponse(BaseModel):
    sql: str


app = FastAPI(title="Text-to-SQL Generator")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def startup_event() -> None:
    base_model_id = os.getenv("BASE_MODEL_ID", "codellama/CodeLlama-7b-Instruct-hf")
    adapter_path = os.getenv("ADAPTER_PATH", "artifacts/qlora_adapter")
    if adapter_path and os.path.exists(adapter_path):
        app.state.generator = load_sql_generator(base_model_id=base_model_id, adapter_path=adapter_path)
    else:
        app.state.generator = load_sql_generator(base_model_id=base_model_id)


@app.post("/generate", response_model=GenerateResponse)
def generate(request: GenerateRequest) -> GenerateResponse:
    sql = app.state.generator.generate(request.question)
    return GenerateResponse(sql=sql)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}
