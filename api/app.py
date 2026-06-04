from __future__ import annotations

import os
from typing import Any

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from core.model import load_sql_generator


class GenerateRequest(BaseModel):
    question: str
    schema: list[dict[str, Any]] = Field(default_factory=list)


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
def startup_event():
    base_model_id = os.getenv("BASE_MODEL_ID", "codellama/CodeLlama-7b-Instruct-hf")
    adapter_path = os.getenv("ADAPTER_PATH", "artifacts/qlora_adapter")

    if adapter_path and os.path.exists(adapter_path):
        app.state.generator = load_sql_generator(base_model_id, adapter_path)
    else:
        app.state.generator = load_sql_generator(base_model_id)


@app.post("/generate", response_model=GenerateResponse)
def generate(request: GenerateRequest) -> GenerateResponse:
    sql = app.state.generator.generate(
        question=request.question,
        schema=request.schema
    )
    return GenerateResponse(sql=sql)


@app.get("/health")
def health():
    return {"status": "ok"}