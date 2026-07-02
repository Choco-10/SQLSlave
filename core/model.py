from __future__ import annotations
import re
from dataclasses import dataclass
from typing import Any
from collections.abc import Mapping, Sequence
import os
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from tokenizers import Tokenizer
from huggingface_hub import hf_hub_download
from core.prompt import build_prompt

def load_corrected_tokenizer(model_id_or_path: str) -> AutoTokenizer:
    tokenizer = AutoTokenizer.from_pretrained(model_id_or_path, use_fast=True)
    tokenizer_json_path = None
    if os.path.isdir(model_id_or_path):
        candidate = os.path.join(model_id_or_path, "tokenizer.json")
        if os.path.exists(candidate):
            tokenizer_json_path = candidate
    else:
        try:
            tokenizer_json_path = hf_hub_download(
                repo_id=model_id_or_path,
                filename="tokenizer.json",
                local_files_only=True
            )
        except Exception:
            try:
                tokenizer_json_path = hf_hub_download(
                    repo_id=model_id_or_path,
                    filename="tokenizer.json"
                )
            except Exception:
                pass
    if tokenizer_json_path and os.path.exists(tokenizer_json_path):
        try:
            raw_tokenizer = Tokenizer.from_file(str(tokenizer_json_path))
            tokenizer._tokenizer = raw_tokenizer
        except Exception as e:
            print(f"Warning: Failed to override tokenizer with raw tokenizer.json: {e}")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer

@dataclass
class GeneratorConfig:
    base_model_id: str = "deepseek-ai/deepseek-coder-6.7b-instruct"
    adapter_path: str | None = None
    max_new_tokens: int = 170
    temperature: float = 0.1
    num_beams: int = 1

class SQLGenerator:
    def __init__(self, model, tokenizer, config: GeneratorConfig) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.config = config
        self._device = next(model.parameters()).device

    @classmethod
    def load(cls, config: GeneratorConfig) -> "SQLGenerator":
        tokenizer = load_corrected_tokenizer(config.base_model_id)
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
        device_map = {"": 0} if torch.cuda.is_available() else "auto"
        model = AutoModelForCausalLM.from_pretrained(
            config.base_model_id,
            quantization_config=quantization_config,
            device_map=device_map,
            dtype=torch.bfloat16,
            attn_implementation="sdpa",
        )
        model.config.pad_token_id = tokenizer.pad_token_id
        model.config.bos_token_id = tokenizer.bos_token_id
        model.config.eos_token_id = tokenizer.eos_token_id
        model.config.use_cache = True
        if config.adapter_path:
            model = PeftModel.from_pretrained(model, config.adapter_path)
        model.eval()
        return cls(model=model, tokenizer=tokenizer, config=config)

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
                    col_info = {}
                    if meta.get("pk"):
                        col_info["pk"] = True
                    if meta.get("fk"):
                        col_info["fk"] = meta["fk"]
                    if meta.get("type"):
                        col_info["type"] = meta["type"]
                    norm_cols[col_name] = col_info
                else:
                    norm_cols[col_name] = {}
            normalized.append({"name": table_name, "columns": norm_cols})
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
        inputs = {k: v.to(self._device) for k, v in inputs.items()}
        with torch.inference_mode():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=self.config.max_new_tokens,
                num_beams=max(1, self.config.num_beams),
                early_stopping=(self.config.num_beams > 1),
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                stop_strings=[";"],
                tokenizer=self.tokenizer,
            )
        generated = self.tokenizer.decode(
            output_ids[0][inputs["input_ids"].shape[1]:],
            skip_special_tokens=True,
        )
        return self._postprocess_sql(generated, question)

    def generate_batch(
        self,
        questions: list[str],
        schemas: list[Sequence[Mapping[str, object]] | None] | None = None,
    ) -> list[str]:
        if schemas is None:
            schemas = [None] * len(questions)
        prompts = []
        for q, s in zip(questions, schemas):
            norm_s = self.normalize_schema(s)
            prompts.append(build_prompt(q, schema=norm_s))
        orig_padding_side = self.tokenizer.padding_side
        self.tokenizer.padding_side = "left"
        inputs = self.tokenizer(prompts, return_tensors="pt", padding=True)
        inputs = {k: v.to(self._device) for k, v in inputs.items()}
        in_len = inputs["input_ids"].shape[1]
        with torch.inference_mode():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=self.config.max_new_tokens,
                num_beams=max(1, self.config.num_beams),
                early_stopping=(self.config.num_beams > 1),
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                stop_strings=[";"],
                tokenizer=self.tokenizer,
            )
        self.tokenizer.padding_side = orig_padding_side
        decoded_outputs = []
        for i in range(len(questions)):
            generated_slice = output_ids[i][in_len:]
            gen_text = self.tokenizer.decode(generated_slice, skip_special_tokens=True)
            decoded_outputs.append(self._postprocess_sql(gen_text, questions[i]))
        return decoded_outputs

    @staticmethod
    def _postprocess_sql(text: str, question: str = "") -> str:
        cleaned = text.replace("<|EOT|>", "").strip()
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
        if question:
            cleaned = SQLGenerator.align_literal_casing(cleaned, question)
        return cleaned

    @staticmethod
    def align_literal_casing(sql: str, question: str) -> str:
        pattern = r"('(?:[^'\\]|\\.)*'|\"(?:[^\"\\]|\\.)*\")"
        parts = re.split(pattern, sql)
        new_parts = []
        q_lower = question.lower()
        for part in parts:
            if (part.startswith("'") and part.endswith("'")) or (part.startswith('"') and part.endswith('"')):
                quote_char = part[0]
                val = part[1:-1]
                val_lower = val.lower()
                if val == val_lower:
                    new_parts.append(part)
                    continue
                if len(val) > 1:
                    idx = q_lower.find(val_lower)
                    if idx != -1:
                        matched_val = question[idx : idx + len(val)]
                        new_parts.append(f"{quote_char}{matched_val}{quote_char}")
                    else:
                        new_parts.append(part)
                elif len(val) == 1:
                    pattern_val = r"\b" + re.escape(val_lower) + r"\b"
                    m = re.search(pattern_val, q_lower)
                    if m:
                        matched_val = question[m.start():m.end()]
                        new_parts.append(f"{quote_char}{matched_val}{quote_char}")
                    else:
                        new_parts.append(part)
                else:
                    new_parts.append(part)
            else:
                new_parts.append(part)
        return "".join(new_parts)

def load_sql_generator(base_model_id: str, adapter_path: str | None = None):
    return SQLGenerator.load(
        GeneratorConfig(base_model_id=base_model_id, adapter_path=adapter_path)
    )