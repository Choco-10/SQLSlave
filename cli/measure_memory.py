"""Measure GPU memory footprint: FP16 vs 4-bit (NF4) quantization.

Compares the peak VRAM usage of loading CodeLlama-7B in full FP16
precision versus 4-bit NF4 quantization with double quantization.

Usage:
    python -m cli.measure_memory
"""
from __future__ import annotations

import gc
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


MODEL_ID = "codellama/CodeLlama-7b-Instruct-hf"


def measure_peak_vram(func):
    """Run *func* and return peak GPU memory allocated (bytes)."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    result = func()

    gc.collect()
    if torch.cuda.is_available():
        peak = torch.cuda.max_memory_allocated()
    else:
        peak = 0

    return peak, result


def load_fp16():
    """Load model in FP16 and return (model, tokenizer)."""
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float16,
        device_map="auto",
    )
    model.eval()
    return model, tokenizer


def load_4bit():
    """Load model in 4-bit NF4 and return (model, tokenizer)."""
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.float16,
    )

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        quantization_config=quantization_config,
        device_map="auto",
        torch_dtype=torch.float16,
    )
    model.eval()
    return model, tokenizer


def format_bytes(b: int) -> str:
    """Format bytes into human-readable string."""
    gb = b / (1024 ** 3)
    mb = b / (1024 ** 2)
    if gb >= 1:
        return f"{gb:.2f} GB"
    return f"{mb:.1f} MB"


def main():
    if not torch.cuda.is_available():
        print("ERROR: No CUDA GPU available. This script requires a GPU.")
        return

    gpu_name = torch.cuda.get_device_name(0)
    gpu_total = torch.cuda.get_device_properties(0).total_memory
    print(f"GPU: {gpu_name}")
    print(f"Total GPU Memory: {format_bytes(gpu_total)}")
    print("=" * 60)

    # --- Measure FP16 ---
    print("\nLoading model in FP16...")
    peak_fp16, (model_fp16, tok_fp16) = measure_peak_vram(load_fp16)
    print(f"Peak VRAM (FP16): {format_bytes(peak_fp16)}")
    del model_fp16, tok_fp16
    gc.collect()
    torch.cuda.empty_cache()

    # --- Measure 4-bit ---
    print("\nLoading model in 4-bit (NF4)...")
    peak_4bit, (model_4bit, tok_4bit) = measure_peak_vram(load_4bit)
    print(f"Peak VRAM (4-bit): {format_bytes(peak_4bit)}")
    del model_4bit, tok_4bit
    gc.collect()
    torch.cuda.empty_cache()

    # --- Summary ---
    reduction_bytes = peak_fp16 - peak_4bit
    reduction_pct = (reduction_bytes / peak_fp16) * 100 if peak_fp16 > 0 else 0

    print("\n" + "=" * 60)
    print("MEMORY COMPARISON RESULTS")
    print("=" * 60)
    print(f"FP16 Peak VRAM:     {format_bytes(peak_fp16)}")
    print(f"4-bit Peak VRAM:    {format_bytes(peak_4bit)}")
    print(f"Memory Saved:       {format_bytes(reduction_bytes)}")
    print(f"Reduction:          {reduction_pct:.1f}%")
    print("=" * 60)


if __name__ == "__main__":
    main()