from __future__ import annotations

import argparse
import os
import math
import sys

if not sys.flags.utf8_mode and os.environ.get("SQLSLAVE_UTF8_BOOTSTRAPPED") != "1":
    import subprocess
    os.environ["SQLSLAVE_UTF8_BOOTSTRAPPED"] = "1"
    sys.exit(subprocess.call([sys.executable, "-X", "utf8", "-m", "cli.train", *sys.argv[1:]]))


os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "garbage_collection_threshold:0.6,max_split_size_mb:128"

import torch
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cuda.enable_math_sdp(False)

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
from transformers.data.data_collator import DataCollatorMixin

from core.dataset import convert_spider_dataset
from core.model import load_corrected_tokenizer


class CompletionOnlyDataCollator(DataCollatorMixin):
    def __init__(self, tokenizer, response_template="### Response:\n", pad_to_multiple_of=8):
        self.tokenizer = tokenizer
        self.response_template = response_template
        self.pad_to_multiple_of = pad_to_multiple_of
        self.response_token_ids = tokenizer.encode(response_template, add_special_tokens=False)
        self.return_tensors = "pt"

    def torch_call(self, examples):
        input_ids = [torch.tensor(ex["input_ids"]) for ex in examples]
        labels = [ids.clone() for ids in input_ids]
        
        for i in range(len(examples)):
            ids = input_ids[i].tolist()
            lbl = labels[i]
            
            found = False
            n_template = len(self.response_token_ids)
            for idx in range(len(ids) - n_template + 1):
                if ids[idx : idx + n_template] == self.response_token_ids:
                    lbl[: idx + n_template] = -100
                    found = True
                    break
            
        max_len = max(len(ids) for ids in input_ids)
        if self.pad_to_multiple_of is not None:
            max_len = ((max_len + self.pad_to_multiple_of - 1) // self.pad_to_multiple_of) * self.pad_to_multiple_of
            
        padded_input_ids = []
        padded_labels = []
        attention_masks = []
        
        for ids, lbl in zip(input_ids, labels):
            pad_len = max_len - len(ids)
            padded_input_ids.append(torch.cat([ids, torch.full((pad_len,), self.tokenizer.pad_token_id, dtype=torch.long)]))
            padded_labels.append(torch.cat([lbl, torch.full((pad_len,), -100, dtype=torch.long)]))
            mask = torch.cat([torch.ones(len(ids), dtype=torch.long), torch.zeros(pad_len, dtype=torch.long)])
            attention_masks.append(mask)
            
        return {
            "input_ids": torch.stack(padded_input_ids),
            "labels": torch.stack(padded_labels),
            "attention_mask": torch.stack(attention_masks),
        }


class TerminalAnalyticsCallback(TrainerCallback):
    def __init__(self) -> None:
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

        if "eval_loss" in logs:
            return

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

        eval_loss = metrics.get("eval_loss")
        eval_loss_str = f"{eval_loss:.4f}" if eval_loss is not None else ""
        print(f"{str(step):<6} | {'EVAL':<7} | {eval_loss_str:<22} | {self._progress_bar(pct)}")

        print("=" * 80)

        self._last_header_step = -100

    def on_train_end(self, args, state, control, **kwargs):
        step = state.global_step
        max_steps = state.max_steps or 1
        print(f"\n  Training finished at step {step}/{max_steps or 'N/A'}")


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument("--model-id", default="deepseek-ai/deepseek-coder-6.7b-instruct")
    parser.add_argument("--spider-root", default="spider")
    parser.add_argument("--train-file", default=None)
    parser.add_argument("--validation-file", default=None)
    parser.add_argument("--converted-dir", default="converted_spider")
    parser.add_argument("--output-dir", default="artifacts/qlora_adapter")

    parser.add_argument("--max-seq-length", type=int, default=1024)
    parser.add_argument("--num-train-epochs", type=float, default=1.0)
    parser.add_argument("--max-schema-tables", type=int, default=8)

    parser.add_argument("--per-device-train-batch-size", type=int, default=1)
    parser.add_argument("--per-device-eval-batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=2e-4)

    parser.add_argument("--lora-r", type=int, default=16, help="LoRA rank (lower rank saves VRAM and avoids paging)")
    parser.add_argument("--lora-alpha", type=int, default=32, help="LoRA alpha parameter")

    parser.add_argument("--logging-steps", type=int, default=10)
    parser.add_argument("--save-steps", type=int, default=500)
    parser.add_argument("--eval-steps", type=int, default=200)

    args = parser.parse_args()

    if args.spider_root and not (args.train_file and args.validation_file):
        train_path, val_path = convert_spider_dataset(
            args.spider_root,
            args.converted_dir,
            max_schema_tables=args.max_schema_tables,
        )
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

    tokenizer = load_corrected_tokenizer(args.model_id)
    tokenizer.padding_side = "right"

    print("Pre-tokenizing datasets on the main thread to preserve tokenizer spaces...")
    def tokenize_fn(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            max_length=args.max_seq_length,
        )

    tokenized_dataset = dataset.map(
        tokenize_fn,
        batched=True,
        num_proc=None,
        remove_columns=dataset["train"].column_names,
    )

    print("Using standard QLoRA setup...")
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    device_map = {"": 0} if torch.cuda.is_available() else "auto"

    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        quantization_config=quantization_config,
        device_map=device_map,
        dtype=torch.bfloat16,
        attn_implementation="sdpa",
    )
    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.bos_token_id = tokenizer.bos_token_id
    model.config.eos_token_id = tokenizer.eos_token_id
    model.config.use_cache = False

    model = prepare_model_for_kbit_training(
        model,
        use_gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
    )

    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )

    training_args = SFTConfig(
        output_dir=args.output_dir,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        logging_steps=args.logging_steps,
        save_strategy="no",
        eval_strategy="epoch" if args.num_train_epochs >= 0.1 else "no",
        bf16=True,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        optim="paged_adamw_8bit",
        dataloader_num_workers=0,
        dataloader_pin_memory=True,
        max_length=args.max_seq_length,
        packing=False,
        torch_empty_cache_steps=1,
    )

    response_template = "### Response:\n"
    collator = CompletionOnlyDataCollator(
        response_template=response_template,
        tokenizer=tokenizer,
        pad_to_multiple_of=8,
    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["validation"],
        processing_class=tokenizer,
        peft_config=lora_config,
        data_collator=collator,
    )

    try:
        trainer.pop_callback(PrinterCallback)
    except Exception:
        pass

    trainer.add_callback(TerminalAnalyticsCallback())

    trainer.train()

    print(f"\nSaving final model and tokenizer to {args.output_dir}...")
    trainer.save_model(args.output_dir)


if __name__ == "__main__":
    main()