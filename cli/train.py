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
    PrinterCallback,
    TrainerCallback,
)
from trl import SFTConfig, SFTTrainer

from core.dataset import convert_spider_dataset


_PROMPT_MARKER = "SQL:[/INST]"


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


# =========================================================
# TRAINING CALLBACK WITH TABULAR LOGGING
# =========================================================
class TerminalAnalyticsCallback(TrainerCallback):
    def __init__(
        self,
        tokenizer,
        eval_dataset,
        exact_match_max_new_tokens: int,
        log_metrics: list[str],
    ) -> None:
        self.tokenizer = tokenizer
        self.eval_dataset = eval_dataset
        self.exact_match_max_new_tokens = exact_match_max_new_tokens
        self.log_metrics = log_metrics
        self._start_time = 0.0
        self._last_header_step = -100

    @staticmethod
    def _progress_bar(pct: float, width: int = 20) -> str:
        filled = int(width * pct / 100.0)
        filled = max(0, min(filled, width))
        bar = "█" * filled + "░" * (width - filled)
        return f"{bar} {pct:5.1f}%"

    @staticmethod
    def _format_eta(seconds: float) -> str:
        if seconds <= 0 or not math.isfinite(seconds):
            return ""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        if hours > 0:
            return f"{hours}h{minutes:02d}m{secs:02d}s"
        elif minutes > 0:
            return f"{minutes}m{secs:02d}s"
        else:
            return f"{secs}s"

    def on_train_begin(self, args, state, control, **kwargs):
        import time
        self._start_time = time.time()
        total_steps = state.max_steps or 0
        print(f"\nTraining started: {args.num_train_epochs} epochs, ~{total_steps} optimizer steps")

    def _print_train_header(self, step: int):
        print("\n" + "=" * 36)
        print(f"{'Step':<6} | {'Event':<7} | {'Loss':<8}")
        print("-" * 36)
        self._last_header_step = step

    def on_log(self, args, state, control, logs=None, **kwargs):
        if not logs:
            return

        step = logs.get("step", state.global_step)

        # Skip logging eval entries here — they are handled by on_evaluate
        if "eval_loss" in logs:
            return

        # ── TRAIN log entry ──
        # Re-print table header every 10 steps so it stays visible
        if step - self._last_header_step >= 10:
            self._print_train_header(step)

        loss = logs.get("loss", logs.get("train_loss"))
        loss_str = f"{loss:.4f}" if loss is not None else ""
        print(f"{str(step):<6} | {'TRAIN':<7} | {loss_str:<8}")

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if not metrics:
            return

        import time
        step = state.global_step
        max_steps = state.max_steps or 1
        pct = min(step / max_steps * 100, 100.0)

        # Compute ETA
        elapsed = time.time() - self._start_time
        steps_done = step
        steps_remaining = max_steps - step
        if steps_done > 0:
            eta_seconds = (elapsed / steps_done) * steps_remaining
            eta_str = f"ETA {self._format_eta(eta_seconds)}"
        else:
            eta_str = ""

        print("\n" + "=" * 80)
        print(f"{'Step':<6} | {'Event':<7} | {'Value':<22} | {'Progress':<24}")
        print("-" * 80)

        # EVAL row
        eval_loss = metrics.get("eval_loss")
        eval_loss_str = f"{eval_loss:.4f}" if eval_loss is not None else ""
        print(f"{str(step):<6} | {'EVAL':<7} | {eval_loss_str:<22} | {self._progress_bar(pct)}")

        print("=" * 80)

        # Reset so next train log re-prints its header
        self._last_header_step = -100

    def on_train_end(self, args, state, control, **kwargs):
        import time
        step = state.global_step
        max_steps = state.max_steps or 1
        pct = 100.0

        print(f"\n  Training finished at step {step}/{max_steps or 'N/A'}")
        print(f"\n  Running final exact match on full validation dataset ({len(self.eval_dataset)} samples)...")

        model = kwargs.get("model")
        if model is not None:
            total_samples = len(self.eval_dataset)
            device = next(model.parameters()).device
            matches = 0

            model.eval()
            start_time = time.time()
            for i, example in enumerate(self.eval_dataset, 1):
                prompt, gold_sql = _extract_prompt_and_sql(example["text"])

                inputs = self.tokenizer(prompt, return_tensors="pt")
                inputs = {name: tensor.to(device) for name, tensor in inputs.items()}

                with torch.inference_mode():
                    output_ids = model.generate(
                        **inputs,
                        max_new_tokens=self.exact_match_max_new_tokens,
                        do_sample=False,
                        temperature=0.0,
                        pad_token_id=self.tokenizer.eos_token_id,
                        eos_token_id=self.tokenizer.eos_token_id,
                    )

                predicted_sql = self.tokenizer.decode(
                    output_ids[0][inputs["input_ids"].shape[1]:],
                    skip_special_tokens=True
                )

                match = _normalize_sql_for_exact_match(predicted_sql) == _normalize_sql_for_exact_match(gold_sql)
                if match:
                    matches += 1

                # Speed and ETA
                elapsed = time.time() - start_time
                samples_per_sec = i / elapsed if elapsed > 0 else 0
                remaining = total_samples - i
                eta_seconds = remaining / samples_per_sec if samples_per_sec > 0 else 0
                eta_str = self._format_eta(eta_seconds)

                # Live per-sample output
                icon = "✅" if match else "❌"
                display_sql = predicted_sql.replace("\n", " ").strip()
                if len(display_sql) > 50:
                    display_sql = display_sql[:47] + "..."
                speed_str = f"{samples_per_sec:.1f} samp/s"
                print(f"[{i:4d}/{total_samples}] {icon} {display_sql:<54} | {speed_str:<14} | ETA {eta_str}")

            accuracy_pct = (matches / total_samples * 100) if total_samples > 0 else 0.0
            print(f"\n  ✅ Final Exact Match: {matches}/{total_samples} ({accuracy_pct:.1f}%)\n")
        else:
            print("  ⚠️  No model available for final exact match evaluation.\n")


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument("--model-id", default="codellama/CodeLlama-7b-Instruct-hf")
    parser.add_argument("--spider-root", default="spider")
    parser.add_argument("--train-file", default=None)
    parser.add_argument("--validation-file", default=None)
    parser.add_argument("--converted-dir", default="converted_spider")
    parser.add_argument("--output-dir", default="artifacts/qlora_adapter")

    # Increased to 512 to prevent truncation of Spider dataset schemas + queries
    parser.add_argument("--max-seq-length", type=int, default=512)
    parser.add_argument("--num-train-epochs", type=float, default=1.0)

    # unchanged training params
    parser.add_argument("--per-device-train-batch-size", type=int, default=1)
    parser.add_argument("--per-device-eval-batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=2e-4)

    parser.add_argument("--logging-steps", type=int, default=10)
    parser.add_argument("--save-steps", type=int, default=500)
    parser.add_argument("--eval-steps", type=int, default=200)

    # ✅ FIXED DEFAULT
    parser.add_argument("--exact-match-max-new-tokens", type=int, default=128)

    args = parser.parse_args()

    if args.spider_root and not (args.train_file and args.validation_file):
        train_path, val_path = convert_spider_dataset(args.spider_root, args.converted_dir)
        args.train_file = str(train_path)
        args.validation_file = str(val_path)

    dataset = load_dataset(
        "json",
        data_files={
            "train": args.train_file,
            "validation": args.validation_file,
        },
    )

    print(f"Loaded dataset: {len(dataset['train'])} train, {len(dataset['validation'])} val")

    tokenizer = AutoTokenizer.from_pretrained(args.model_id, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # ✅ 1. Enable 4-bit Quantization (QLoRA) to drastically reduce VRAM and speed up training
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        quantization_config=quantization_config,
        device_map="auto",
        torch_dtype=torch.bfloat16,
    )

    # ✅ 2. Prepare model for k-bit training (required for LoRA/QLoRA)
    model = prepare_model_for_kbit_training(model)

    # ✅ 3. Add LoRA configuration
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )

    # ✅ 4. Optimized SFTConfig for maximum training speed
    training_args = SFTConfig(
        output_dir=args.output_dir,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        save_total_limit=2,
        eval_strategy="steps",
        eval_steps=args.eval_steps,
        
        # Speed & Memory Optimizations:
        bf16=True,                                  # Use bfloat16 mixed precision (2x faster, stable on Ampere+)
        gradient_checkpointing=True,                # Save VRAM (critical for 6GB GPUs like RTX 4050)
        optim="paged_adamw_32bit",                  # Memory-efficient optimizer, prevents OOM spikes
        dataloader_num_workers=0,                   # Set to 0 on Windows to avoid 'spawn' multiprocessing overhead/hangs
        dataloader_pin_memory=False,                # Pin memory is less beneficial with num_workers=0 on Windows
        max_length=args.max_seq_length,         # Prevent truncation warnings
        dataset_text_field="text",                  # Required for SFTTrainer
        packing=False,                              # Set to True if you want to pack multiple short sequences
    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        tokenizer=tokenizer,
        peft_config=lora_config,                    # Apply LoRA config
    )

    # Remove default PrinterCallback to suppress ugly HF dictionary logs
    # Use try/except for backward compat with newer transformers versions
    try:
        trainer.pop_callback(PrinterCallback)
    except Exception:
        pass

    trainer.add_callback(
        TerminalAnalyticsCallback(
            tokenizer,
            dataset["validation"],
            args.exact_match_max_new_tokens,
            [],
        )
    )

    trainer.train()


if __name__ == "__main__":
    main()