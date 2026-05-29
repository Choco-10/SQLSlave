from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

try:
    from .prompt import build_prompt
except ImportError:  # pragma: no cover - allows direct script execution
    from prompt import build_prompt


@dataclass
class GeneratorConfig:
    base_model_id: str = "codellama/CodeLlama-7b-Instruct-hf"
    adapter_path: str | None = None
    max_new_tokens: int = 128
    temperature: float = 0.1


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

    def generate(self, question: str) -> str:
        prompt = build_prompt(question)
        inputs = self.tokenizer(prompt, return_tensors="pt")
        device = next(self.model.parameters()).device
        inputs = {name: tensor.to(device) for name, tensor in inputs.items()}

        with torch.inference_mode():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=self.config.max_new_tokens,
                do_sample=False,
                temperature=self.config.temperature,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )

        generated = self.tokenizer.decode(output_ids[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True)
        return self._clean_sql(generated)

    @staticmethod
    def _clean_sql(text: str) -> str:
        cleaned = text.strip()
        cleaned = re.sub(r"^```(?:sql)?", "", cleaned, flags=re.IGNORECASE).strip()
        cleaned = re.sub(r"```$", "", cleaned).strip()
        cleaned = cleaned.split("\n\n")[0].strip()
        cleaned = cleaned.rstrip(".")
        sql_match = re.search(r"(?is)(select\s+.*)", cleaned)
        if sql_match:
            cleaned = sql_match.group(1).strip()
        if cleaned and not cleaned.endswith(";"):
            cleaned = f"{cleaned};"
        return cleaned


def load_sql_generator(
    base_model_id: str = "codellama/CodeLlama-7b-Instruct-hf",
    adapter_path: str | None = None,
) -> SQLGenerator:
    return SQLGenerator.load(
        GeneratorConfig(base_model_id=base_model_id, adapter_path=adapter_path)
    )


def load_adapter_path(output_dir: str | Path) -> str:
    output_dir = Path(output_dir)
    if (output_dir / "adapter_config.json").exists():
        return str(output_dir)
    raise FileNotFoundError(f"No PEFT adapter found in {output_dir}")
