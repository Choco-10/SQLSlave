from __future__ import annotations

import argparse
import os
import time

try:
    from .model import load_sql_generator
except ImportError:
    from model import load_sql_generator


def main() -> None:
    parser = argparse.ArgumentParser(description="Quick inference smoke test for the SQL generator.")
    parser.add_argument("--adapter-path", default=os.getenv("ADAPTER_PATH", "artifacts/qlora_adapter"))
    parser.add_argument(
        "--model-id", default=os.getenv("BASE_MODEL_ID", "codellama/CodeLlama-7b-Instruct-hf")
    )
    parser.add_argument(
        "--question",
        default="Show total sales by region from orders table",
        help="Natural language question that includes table hint",
    )
    args = parser.parse_args()

    print("Loading model (this may take a minute)...")
    start = time.time()
    gen = load_sql_generator(base_model_id=args.model_id, adapter_path=args.adapter_path)
    print(f"Loaded in {time.time()-start:.1f}s")

    print("Generating SQL for question:")
    print("  ", args.question)
    start = time.time()
    sql = gen.generate(args.question)
    print(f"Done ({time.time()-start:.2f}s). Generated SQL:\n")
    print(sql)


if __name__ == "__main__":
    main()
