from __future__ import annotations

import argparse
import os
import math
import re
import sys


if not sys.flags.utf8_mode and os.environ.get("SQLSLAVE_UTF8_BOOTSTRAPPED") != "1":
    os.environ["SQLSLAVE_UTF8_BOOTSTRAPPED"] = "1"
    os.execv(sys.executable, [sys.executable, "-X", "utf8", os.path.abspath(__file__), *sys.argv[1:]])

import torch
from datasets import load_dataset
from peft import LoraConfig, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainerCallback,
)
from trl import SFTConfig, SFTTrainer

try:
    from .dataset import convert_spider_dataset
except ImportError:  # pragma: no cover - allows direct script execution
    from dataset import convert_spider_dataset


_PROMPT_MARKER = "SQL:[/INST]"

DEFAULT_LOG_METRICS = [
    "loss",
    "eval_loss",
    "epoch",
    "progress",
    "runtime",
    "samples_per_second",
    "steps_per_second",
    "entropy",
    "num_tokens",
    "mean_token_accuracy",
    "step",
    "event",
    "exact_match",
    "exact_match_detail",
]


def _format_float(value: float | int | None, precision: int = 4) -> str:
    if value is None:
        return "-"
    return f"{value:.{precision}f}"


def _format_percent(value: float | None) -> str:
    if value is None:
        return "-"
    return f"{value * 100:.1f}%"


def _format_fraction(numerator: int | None, denominator: int | None) -> str:
    if numerator is None or denominator in (None, 0):
        return "-"
    return f"{numerator}/{denominator}"


def _normalize_log_metrics(raw_metrics: str | None) -> list[str]:
    if not raw_metrics:
        return DEFAULT_LOG_METRICS.copy()

    metrics: list[str] = []
    for metric in raw_metrics.split(","):
        cleaned = metric.strip()
        if cleaned and cleaned not in metrics:
            metrics.append(cleaned)

    return metrics or DEFAULT_LOG_METRICS.copy()


def _metric_legend() -> dict[str, str]:
    return {
        "loss": "Training error; lower means fewer mistakes on the batch.",
        "eval_loss": "Validation error; like a quiz after practice.",
        "perplexity": "Eval loss translated into how many choices the model feels it has.",
        "learning_rate": "Current step size; like how big the steering correction is.",
        "grad_norm": "Update strength; like how hard the model is being pushed.",
        "epoch": "How many passes through the dataset are done.",
        "progress": "Step count and completion percent.",
        "runtime": "How long evaluation took.",
        "samples_per_second": "Validation rows processed per second.",
        "steps_per_second": "Evaluation steps processed per second.",
        "entropy": "Average predictive entropy of the model's token distribution; higher means more uncertainty.",
        "num_tokens": "Number of tokens processed in the logged interval; used to compute throughput.",
        "mean_token_accuracy": "Token-level accuracy: fraction of tokens where the top prediction equals the gold token.",
        "step": "Global optimizer/update step count.",
        "event": "Type of log row (train, eval, exact) for human-friendly tables.",
        "exact_match": "Exact SQL match rate after normalization on sampled validation rows.",
        "exact_match_detail": "Raw matched count over the sampled validation set.",
    }


def print_metric_legend(selected_metrics: list[str]) -> None:
    legend = _metric_legend()
    print("Metric guide:")
    for metric in selected_metrics:
        description = legend.get(metric)
        if description is not None:
            print(f"  {metric}: {description}")


def _build_table_row(columns: list[tuple[str, str]]) -> str:
    widths = [max(len(header), len(value)) for header, value in columns]
    header = " | ".join(f"{name:<{width}}" for (name, _), width in zip(columns, widths, strict=True))
    values = " | ".join(f"{value:<{width}}" for (_, value), width in zip(columns, widths, strict=True))
    separator = "-+-".join("-" * width for width in widths)
    return "\n".join([header, separator, values])


def _build_event_columns(
    *,
    event_type: str,
    step: int,
    total_steps: int,
    logs: dict[str, float] | None = None,
    metrics: dict[str, float] | None = None,
    exact_match: float | None = None,
    exact_matches: int | None = None,
    exact_match_total: int | None = None,
    selected_metrics: list[str],
) -> list[tuple[str, str]]:
    data = logs or metrics or {}
    columns: list[tuple[str, str]] = [
        ("event", event_type),
        ("step", str(step)),
    ]

    # Helper to fetch metric value; returns None when no displayable value is present
    def _get_val(name: str):
        if name == "epoch":
            return _format_float(data.get("epoch"), 2) if data.get("epoch") is not None else None
        if name == "loss":
            return _format_float(data.get("loss"), 4) if data.get("loss") is not None else None
        if name == "eval_loss":
            return _format_float(data.get("eval_loss"), 4) if data.get("eval_loss") is not None else None
        if name == "perplexity":
            eval_loss = data.get("eval_loss")
            if eval_loss is None:
                return None
            try:
                return f"{math.exp(min(eval_loss, 20.0)):.2f}"
            except OverflowError:
                return "inf"
        if name == "learning_rate":
            return f"{data.get('learning_rate'):.2e}" if data.get("learning_rate") is not None else None
        if name == "grad_norm":
            return _format_float(data.get("grad_norm"), 4) if data.get("grad_norm") is not None else None
        if name == "progress":
            return f"{step}/{total_steps} ({step / total_steps * 100:.1f}%)" if total_steps else str(step)
        if name == "runtime":
            runtime = data.get("eval_runtime") if metrics is not None else data.get("runtime")
            return f"{runtime:.1f}s" if runtime is not None else None
        if name == "samples_per_second":
            samples_per_second = data.get("eval_samples_per_second") if metrics is not None else data.get("samples_per_second")
            return f"{samples_per_second:.2f}" if samples_per_second is not None else None
        if name == "steps_per_second":
            steps_per_second = data.get("eval_steps_per_second") if metrics is not None else data.get("steps_per_second")
            return f"{steps_per_second:.2f}" if steps_per_second is not None else None
        if name == "exact_match":
            return _format_percent(exact_match) if exact_match is not None else None
        if name == "exact_match_detail":
            return _format_fraction(exact_matches, exact_match_total) if exact_matches is not None and exact_match_total is not None else None

        # Generic fallback: show any training-specific scalar present in `data`
        val = data.get(name)
        if val is None:
            return None
        # Format floats neatly
        if isinstance(val, float):
            return _format_float(val, 4)
        return str(val)

    # Build columns preserving requested order but skipping missing values
    for metric in selected_metrics:
        val = _get_val(metric)
        if val is not None:
            # Friendly header names for some metrics
            header = metric
            if metric == "learning_rate":
                header = "lr"
            if metric == "samples_per_second":
                header = "samples/s"
            if metric == "steps_per_second":
                header = "steps/s"
            columns.append((header, val))

    # Also, for train events, include any extra scalars present in logs that weren't requested explicitly
    if logs is not None:
        known_headers = {h for h, _ in columns}
        for k, v in logs.items():
            if k in ("loss", "eval_loss", "epoch", "learning_rate", "grad_norm", "progress"):
                continue
            if k in ("eval_runtime", "eval_samples_per_second", "eval_steps_per_second"):
                continue
            if str(k) in known_headers:
                continue
            # Skip if value is None
            if v is None:
                continue
            # Format and append
            if isinstance(v, float):
                columns.append((k, _format_float(v, 4)))
            else:
                columns.append((k, str(v)))

    return columns


def _extract_prompt_and_sql(text: str) -> tuple[str, str]:
    if _PROMPT_MARKER not in text:
        return text.strip(), ""

    prompt, sql = text.split(_PROMPT_MARKER, 1)
    return f"{prompt}{_PROMPT_MARKER}", sql.strip()


def _normalize_sql_for_exact_match(text: str) -> str:
    cleaned = text.strip()
    cleaned = re.sub(r"^```(?:sql)?", "", cleaned, flags=re.IGNORECASE).strip()
    cleaned = re.sub(r"```$", "", cleaned).strip()
    cleaned = cleaned.rstrip(";")
    cleaned = cleaned.lower()
    cleaned = re.sub(r"\s+", " ", cleaned)
    cleaned = re.sub(r"\s*([(),.=<>+\-/*])\s*", r"\1", cleaned)
    return cleaned.strip()


def evaluate_exact_match(model, tokenizer, dataset, sample_count: int, max_new_tokens: int) -> tuple[float, int, int]:
    if sample_count <= 0:
        sample_count = len(dataset)
    sample_count = min(sample_count, len(dataset))
    if sample_count == 0:
        return 0.0, 0, 0

    device = next(model.parameters()).device
    matches = 0

    model.eval()
    for index in range(sample_count):
        example = dataset[index]
        prompt, gold_sql = _extract_prompt_and_sql(example["text"])

        inputs = tokenizer(prompt, return_tensors="pt")
        inputs = {name: tensor.to(device) for name, tensor in inputs.items()}

        with torch.inference_mode():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                temperature=0.0,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )

        predicted_sql = tokenizer.decode(output_ids[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True)
        if _normalize_sql_for_exact_match(predicted_sql) == _normalize_sql_for_exact_match(gold_sql):
            matches += 1

    return matches / sample_count, matches, sample_count


class TerminalAnalyticsCallback(TrainerCallback):
    def __init__(
        self,
        tokenizer,
        eval_dataset,
        exact_match_samples: int,
        exact_match_max_new_tokens: int,
        log_metrics: list[str],
    ) -> None:
        self.tokenizer = tokenizer
        self.eval_dataset = eval_dataset
        self.exact_match_samples = exact_match_samples
        self.exact_match_max_new_tokens = exact_match_max_new_tokens
        self.log_metrics = log_metrics

    def on_train_begin(self, args, state, control, **kwargs):
        total_steps = state.max_steps or 0
        print_metric_legend(self.log_metrics)
        print(
            f"Training started: {args.num_train_epochs} epochs, "
            f"about {total_steps} optimizer steps, "
            f"logging every {args.logging_steps} steps, eval every {args.eval_steps} steps."
        )

    def on_log(self, args, state, control, logs=None, **kwargs):
        if not logs:
            return

        step = state.global_step
        total_steps = state.max_steps or 0
        columns = _build_event_columns(
            event_type="train",
            step=step,
            total_steps=total_steps,
            logs=logs,
            selected_metrics=self.log_metrics,
        )
        print(_build_table_row(columns))

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if not metrics:
            return

        eval_loss = metrics.get("eval_loss")
        perplexity = None
        if eval_loss is not None:
            try:
                perplexity = math.exp(min(eval_loss, 20.0))
            except OverflowError:
                perplexity = float("inf")

        columns = _build_event_columns(
            event_type="eval",
            step=state.global_step,
            total_steps=state.max_steps or 0,
            metrics=metrics,
            exact_match=None,
            exact_matches=None,
            exact_match_total=None,
            selected_metrics=self.log_metrics,
        )
        if perplexity is not None and "perplexity" in self.log_metrics:
            columns = [(name, (f"{perplexity:.2f}" if name == "perplexity" else value)) for name, value in columns]

        print(_build_table_row(columns))

        model = kwargs.get("model")
        if model is not None:
            exact_match, exact_matches, exact_match_total = evaluate_exact_match(
                model,
                self.tokenizer,
                self.eval_dataset,
                self.exact_match_samples,
                self.exact_match_max_new_tokens,
            )
            exact_match_columns = _build_event_columns(
                event_type="exact",
                step=state.global_step,
                total_steps=state.max_steps or 0,
                exact_match=exact_match,
                exact_matches=exact_matches,
                exact_match_total=exact_match_total,
                selected_metrics=self.log_metrics,
            )
            print(_build_table_row(exact_match_columns))

    def on_train_end(self, args, state, control, **kwargs):
        total_steps = state.max_steps or 0
        print(f"Training finished at step {state.global_step}/{total_steps}.")


def build_tokenizer(model_id: str, token: str | None):
    tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True, token=token)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    return tokenizer


def build_model(model_id: str, token: str | None):
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for training. Install a CUDA-enabled PyTorch build and NVIDIA driver first.")

    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.float16,
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=quantization_config,
        device_map="auto",
        torch_dtype=torch.float16,
        token=token,
    )
    model = prepare_model_for_kbit_training(model)
    return model


def main() -> None:
    parser = argparse.ArgumentParser(description="Fine-tune CodeLlama with QLoRA for text-to-SQL.")
    parser.add_argument("--model-id", default="codellama/CodeLlama-7b-Instruct-hf")
    parser.add_argument("--spider-root", default="spider", help="Path to the Spider dataset root.")
    parser.add_argument("--train-file", default=None, help="Path to a converted train JSONL file.")
    parser.add_argument("--validation-file", default=None, help="Path to a converted validation JSONL file.")
    parser.add_argument("--converted-dir", default="converted_spider")
    parser.add_argument("--output-dir", default="artifacts/qlora_adapter")
    parser.add_argument("--max-seq-length", type=int, default=512)
    parser.add_argument("--num-train-epochs", type=float, default=1.0)
    parser.add_argument("--per-device-train-batch-size", type=int, default=1)
    parser.add_argument("--per-device-eval-batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--logging-steps", type=int, default=10)
    parser.add_argument("--save-steps", type=int, default=100)
    parser.add_argument("--eval-steps", type=int, default=100)
    parser.add_argument(
        "--log-metrics",
        default=",".join(DEFAULT_LOG_METRICS),
        help="Comma-separated metrics to show in the training table. Available: "
        "loss, eval_loss, perplexity, learning_rate, grad_norm, epoch, progress, runtime, samples_per_second, "
        "steps_per_second, exact_match, exact_match_detail.",
    )
    parser.add_argument(
        "--hf-token",
        default=os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN"),
        help="Hugging Face access token for authenticated model downloads.",
    )
    parser.add_argument(
        "--exact-match-samples",
        type=int,
        default=32,
        help="Number of validation examples to use for the exact-match metric. Use 0 for the full validation set.",
    )
    parser.add_argument(
        "--exact-match-max-new-tokens",
        type=int,
        default=128,
        help="Maximum number of new tokens to generate for each exact-match validation example.",
    )
    args = parser.parse_args()

    train_file = args.train_file
    validation_file = args.validation_file

    if args.spider_root and not (train_file and validation_file):
        train_path, validation_path = convert_spider_dataset(args.spider_root, args.converted_dir)
        train_file = str(train_path)
        validation_file = str(validation_path)

    if not train_file or not validation_file:
        raise ValueError("Provide either --spider-root or both --train-file and --validation-file.")

    dataset = load_dataset("json", data_files={"train": train_file, "validation": validation_file})
    print(
        f"Loaded dataset: {len(dataset['train'])} train rows, {len(dataset['validation'])} validation rows."
    )

    hf_token = args.hf_token

    tokenizer = build_tokenizer(args.model_id, hf_token)
    model = build_model(args.model_id, hf_token)

    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )

    training_args = SFTConfig(
        output_dir=args.output_dir,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        logging_steps=args.logging_steps,
        logging_strategy="steps",
        logging_first_step=True,
        save_steps=args.save_steps,
        eval_steps=args.eval_steps,
        eval_strategy="steps",
        save_strategy="steps",
        disable_tqdm=False,
        bf16=torch.cuda.is_available() and torch.cuda.is_bf16_supported(),
        fp16=torch.cuda.is_available() and not torch.cuda.is_bf16_supported(),
        gradient_checkpointing=True,
        optim="paged_adamw_8bit",
        report_to="none",
        remove_unused_columns=False,
        load_best_model_at_end=False,
        max_length=args.max_seq_length,
        packing=False,
    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        peft_config=lora_config,
        processing_class=tokenizer,
    )
    trainer.add_callback(
        TerminalAnalyticsCallback(
            tokenizer=tokenizer,
            eval_dataset=dataset["validation"],
            exact_match_samples=args.exact_match_samples,
            exact_match_max_new_tokens=args.exact_match_max_new_tokens,
            log_metrics=_normalize_log_metrics(args.log_metrics),
        )
    )

    train_result = trainer.train()
    if getattr(train_result, "metrics", None):
        print("Training summary:")
        for key in ["train_loss", "train_runtime", "train_samples_per_second", "train_steps_per_second", "epoch"]:
            value = train_result.metrics.get(key)
            if value is None:
                continue
            if key == "train_loss":
                print(f"  {key}={value:.4f}")
            elif key == "epoch":
                print(f"  {key}={value:.2f}")
            else:
                print(f"  {key}={value:.2f}")

    exact_match, exact_matches, exact_match_total = evaluate_exact_match(
        trainer.model,
        tokenizer,
        dataset["validation"],
        0,
        args.exact_match_max_new_tokens,
    )
    print(
        f"Final validation exact match: {exact_matches}/{exact_match_total} = {exact_match * 100:.1f}% "
        f"(normalized SQL string match)."
    )

    trainer.model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)


if __name__ == "__main__":
    main()
