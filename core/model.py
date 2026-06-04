from __future__ import annotations

from collections.abc import Mapping, Sequence
import re
from dataclasses import dataclass
from typing import Any

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from core.prompt import build_prompt


@dataclass
class GeneratorConfig:
    base_model_id: str = "codellama/CodeLlama-7b-Instruct-hf"
    adapter_path: str | None = None
    max_new_tokens: int = 128
    temperature: float = 0.1
    num_beams: int = 1


class SQLGenerator:
    def __init__(self, model, tokenizer, config: GeneratorConfig) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.config = config

    @classmethod
    def load(cls, config: GeneratorConfig) -> "SQLGenerator":
        tokenizer = AutoTokenizer.from_pretrained(config.base_model_id, use_fast=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.float16,
        )

        model = AutoModelForCausalLM.from_pretrained(
            config.base_model_id,
            quantization_config=quantization_config,
            device_map="auto",
            torch_dtype=torch.float16,
        )

        if config.adapter_path:
            model = PeftModel.from_pretrained(model, config.adapter_path)

        model.eval()
        return cls(model=model, tokenizer=tokenizer, config=config)

    # -----------------------------
    # IMPORTANT FIX: SCHEMA NORMALIZATION
    # -----------------------------
    @staticmethod
    def normalize_schema(schema: Sequence[Mapping[str, Any]] | None):
        if not schema:
            return []

        normalized = []

        for table in schema:
            table_name = (table.get("name") or "").strip()
            columns = table.get("columns") or {}

            norm_cols = {}

            for col_name, meta in columns.items():
                col_name = str(col_name).strip()
                if not col_name:
                    continue

                if isinstance(meta, dict):
                    norm_cols[col_name] = {
                        "pk": bool(meta.get("pk", False)),
                        "fk": meta.get("fk", None),
                    }
                else:
                    norm_cols[col_name] = {"pk": False, "fk": None}

            normalized.append({
                "name": table_name,
                "columns": norm_cols
            })

        return normalized

    def generate(
        self,
        question: str,
        schema: Sequence[Mapping[str, object]] | None = None,
        prompt: str | None = None,
    ) -> str:

        schema = self.normalize_schema(schema)

        if prompt is None:
            prompt = build_prompt(question, schema=schema)

        inputs = self.tokenizer(prompt, return_tensors="pt")
        device = next(self.model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.inference_mode():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=self.config.max_new_tokens,
                num_beams=max(1, self.config.num_beams),
                early_stopping=True,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )

        generated = self.tokenizer.decode(
            output_ids[0][inputs["input_ids"].shape[1]:],
            skip_special_tokens=True
        )

        return self._postprocess_sql(generated)

    @staticmethod
    def _postprocess_sql(text: str) -> str:
        cleaned = text.strip()
        cleaned = re.sub(r"^```(?:sql)?", "", cleaned, flags=re.IGNORECASE).strip()
        cleaned = re.sub(r"```$", "", cleaned).strip()
        cleaned = cleaned.split("\n\n")[0].strip()
        cleaned = cleaned.rstrip(".")

        match = re.search(r"(?is)\b(with|select)\b", cleaned)
        if match:
            cleaned = cleaned[match.start():].strip()

        if ";" in cleaned:
            cleaned = cleaned.split(";", 1)[0].strip()

        if cleaned and not cleaned.endswith(";"):
            cleaned += ";"

        return cleaned


def load_sql_generator(base_model_id: str, adapter_path: str | None = None):
    return SQLGenerator.load(
        GeneratorConfig(base_model_id=base_model_id, adapter_path=adapter_path)
    )