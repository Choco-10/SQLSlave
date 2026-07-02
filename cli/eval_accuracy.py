from __future__ import annotations
import json
import re
import sqlite3
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Optional
import torch

torch.backends.cuda.enable_math_sdp(False)
from core.model import load_sql_generator

CONVERTED_VAL = Path("converted_spider/validation.jsonl")
SPIDER_DEV    = Path("spider/dev.json")
SPIDER_ROOT   = Path("spider")
DB_DIR        = SPIDER_ROOT / "database"
MODEL_ID      = "deepseek-ai/deepseek-coder-6.7b-instruct"
ADAPTER_PATH  = "artifacts/qlora_adapter"
MAX_ROWS      = 5000
DB_EXEC_WORKERS = 8

_CODE_BLOCK_RE = re.compile(r"```.*?```", re.S)
_SQL_START_RE  = re.compile(r"(?is)\b(select|with)\b")
_LIMIT_RE      = re.compile(r"\bLIMIT\b", re.IGNORECASE)

def _normalize_structure(sql: str) -> str:
    sql = sql.strip().rstrip(";")
    parts = re.split(r"('(?:[^'\\]|\\.)*'|\"(?:[^\"\\]|\\.)*\")", sql)
    normalised = []
    for part in parts:
        if (part.startswith("'") and part.endswith("'")) or (part.startswith('"') and part.endswith('"')):
            normalised.append(part.strip())
        else:
            p = part.lower()
            p = re.sub(r"\s+", " ", p)
            p = re.sub(r"\s*,\s*", ",", p)
            p = re.sub(r"\s*\(\s*", "(", p)
            p = re.sub(r"\s*\)\s*", ")", p)
            p = re.sub(r"\s*=\s*", "=", p)
            p = re.sub(r"\s*>\s*", ">", p)
            p = re.sub(r"\s*<\s*", "<", p)
            p = re.sub(r"\s*!=\s*", "!=", p)
            p = re.sub(r"\s*>=\s*", ">=", p)
            p = re.sub(r"\s*<=\s*", "<=", p)
            normalised.append(p)
    return "".join(normalised).strip()

def extract_sql_raw(text: str) -> str:
    text = _CODE_BLOCK_RE.sub("", text).lstrip()
    m = _SQL_START_RE.search(text)
    if not m:
        return "SELECT 1"
    sql = text[m.start():]
    sc = sql.find(";")
    if sc != -1:
        sql = sql[:sc]
    return sql.strip()

def validate_sql(sql: str) -> str:
    sql = re.sub(r"\s+", " ", sql).strip()
    if not sql.endswith(";"):
        sql += ";"
    if not re.search(r"(?i)\b(select|with)\b", sql):
        return "SELECT 1;"
    return sql

def _load_dev_db_ids(dev_path: Path) -> list[str]:
    with dev_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    return [ex["db_id"] for ex in data]

def load_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            yield json.loads(line)

def _build_db_path_map(db_ids: list[str]) -> dict[str, Optional[Path]]:
    mapping = {}
    for db_id in set(db_ids):
        candidate = DB_DIR / db_id / f"{db_id}.sqlite"
        if candidate.exists():
            mapping[db_id] = candidate
        elif (DB_DIR / db_id).exists():
            files = list((DB_DIR / db_id).glob("*.sqlite"))
            mapping[db_id] = files[0] if files else None
        else:
            mapping[db_id] = None
    return mapping

class _ConnectionPool:
    def __init__(self) -> None:
        self._conns = {}

    def get(self, db_path: Path) -> sqlite3.Connection:
        if db_path not in self._conns:
            conn = sqlite3.connect(str(db_path), check_same_thread=False)
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA query_only=ON")
            self._conns[db_path] = conn
        return self._conns[db_path]

    def close_all(self) -> None:
        for conn in self._conns.values():
            try:
                conn.close()
            except Exception:
                pass
        self._conns.clear()

_pool = _ConnectionPool()

def execute_sql(db_path: Path, sql: str, timeout: int = 30, sort: bool = True) -> list[tuple] | None:
    if not _LIMIT_RE.search(sql.rstrip().rstrip(";")):
        sql = sql.rstrip().rstrip(";") + f" LIMIT {MAX_ROWS};"
    try:
        with sqlite3.connect(str(db_path), timeout=timeout) as conn:
            conn.execute("PRAGMA query_only=ON")
            cursor = conn.execute(sql)
            rows = cursor.fetchall()
        if sort:
            return sorted(rows, key=lambda r: tuple(("" if v is None else str(v)) for v in r))
        return rows
    except Exception:
        return None

def result_sets_match(pred_rows: list[tuple], gold_rows: list[tuple], ordered: bool = False) -> bool:
    if len(pred_rows) != len(gold_rows):
        return False
    def sort_key(row):
        return tuple("" if v is None else str(v) for v in row)
    rows_iter = zip(pred_rows, gold_rows) if ordered else zip(sorted(pred_rows, key=sort_key), sorted(gold_rows, key=sort_key))
    for p_row, g_row in rows_iter:
        if len(p_row) != len(g_row):
            return False
        for p_val, g_val in zip(p_row, g_row):
            if str(p_val) != str(g_val):
                return False
    return True

def _evaluate_one(idx: int, pred_validated: str, gold_validated: str, db_path: Optional[Path], gold_cache: dict) -> dict:
    if db_path is None or not db_path.exists():
        return {"idx": idx, "exec_attempted": False}
    cache_key = (str(db_path), gold_validated)
    if cache_key in gold_cache:
        gold_rows = gold_cache[cache_key]
    else:
        gold_rows = execute_sql(db_path, gold_validated, sort=True)
        gold_cache[cache_key] = gold_rows
    pred_rows = execute_sql(db_path, pred_validated, sort=True)
    exec_match = False
    pred_failed = pred_rows is None
    gold_failed = gold_rows is None
    if pred_rows is not None and gold_rows is not None:
        exec_match = result_sets_match(pred_rows, gold_rows)
    return {
        "idx": idx,
        "exec_attempted": True,
        "exec_match": exec_match,
        "pred_failed": pred_failed,
        "gold_failed": gold_failed,
    }

def main():
    if not CONVERTED_VAL.exists():
        print(f"ERROR: Validation file not found at {CONVERTED_VAL}")
        return
    if not SPIDER_DEV.exists():
        print(f"ERROR: Spider dev.json not found at {SPIDER_DEV}")
        return
    db_available = DB_DIR.exists() and any(DB_DIR.iterdir())
    if not db_available:
        print(f"WARNING: Spider databases not found at {DB_DIR}")

    db_ids = _load_dev_db_ids(SPIDER_DEV)
    samples = list(load_jsonl(CONVERTED_VAL))
    if len(samples) != len(db_ids):
        print("WARNING: sample count mismatch. Using min.")
    total = min(len(samples), len(db_ids))
    import os
    if "EVAL_LIMIT" in os.environ:
        try:
            total = min(total, int(os.environ["EVAL_LIMIT"]))
            print(f"Limiting evaluation to first {total} samples via EVAL_LIMIT")
        except ValueError:
            pass

    samples = samples[:total]
    db_ids = db_ids[:total]
    original_indices = list(range(len(samples)))
    db_path_map = _build_db_path_map(db_ids) if db_available else {}

    print("Loading model …\n")
    generator = load_sql_generator(base_model_id=MODEL_ID, adapter_path=ADAPTER_PATH)

    print(f"Phase 1 — Inference on {total} samples …\n")
    phase1_start = time.time()
    BATCH_SIZE = 2
    predictions = []
    latencies = []

    for batch_idx in range(0, total, BATCH_SIZE):
        batch_samples = samples[batch_idx : batch_idx + BATCH_SIZE]
        batch_db_ids = db_ids[batch_idx : batch_idx + BATCH_SIZE]
        t0 = time.time()
        questions = [s["question"] for s in batch_samples]
        schemas = [s.get("schema", []) for s in batch_samples]
        raw_outputs = generator.generate_batch(questions, schemas)
        batch_latency = (time.time() - t0) / len(batch_samples)
        for _ in range(len(batch_samples)):
            latencies.append(batch_latency)
        for sample, db_id, raw_output in zip(batch_samples, batch_db_ids, raw_outputs):
            gold_sql = sample["sql"]
            pred_extracted = extract_sql_raw(raw_output)
            pred_validated = validate_sql(pred_extracted)
            pred_clean = _normalize_structure(pred_extracted)
            gold_extracted = extract_sql_raw(gold_sql)
            gold_validated = validate_sql(gold_extracted)
            gold_clean = _normalize_structure(gold_sql)
            predictions.append({
                "pred_validated": pred_validated,
                "pred_clean": pred_clean,
                "gold_validated": gold_validated,
                "gold_clean": gold_clean,
                "db_id": db_id,
            })
        current_processed = min(batch_idx + BATCH_SIZE, total)
        avg_lat = sum(latencies) / len(latencies)
        remaining = avg_lat * (total - current_processed)
        eta_m, eta_s = divmod(int(remaining), 60)
        print(f"  [{current_processed:4d}/{total}]  Avg:{avg_lat:.3f}s/sample  ETA:{eta_m}m{eta_s:02d}s")

    phase1_time = time.time() - phase1_start
    print(f"\nPhase 1 done in {phase1_time:.1f}s\n")

    em_correct = 0
    exec_correct = 0
    exec_attempted = 0
    exec_pred_fail = 0
    exec_gold_fail = 0

    em_results = [p["pred_clean"] == p["gold_clean"] for p in predictions]
    em_correct = sum(em_results)

    if db_available:
        print(f"Phase 2 — DB evaluation ({DB_EXEC_WORKERS} threads) …\n")
        phase2_start = time.time()
        gold_cache = {}
        exec_results = {}
        with ThreadPoolExecutor(max_workers=DB_EXEC_WORKERS) as executor:
            futures = {
                executor.submit(
                    _evaluate_one,
                    original_indices[i],
                    predictions[i]["pred_validated"],
                    predictions[i]["gold_validated"],
                    db_path_map.get(predictions[i]["db_id"]),
                    gold_cache,
                ): i
                for i in range(total)
            }
            completed = 0
            running_exec_ok = 0
            running_exec_at = 0
            for future in as_completed(futures):
                result = future.result()
                exec_results[result["idx"]] = result
                completed += 1
                if result.get("exec_attempted"):
                    running_exec_at += 1
                    if result.get("exec_match"):
                        running_exec_ok += 1
                if completed % 10 == 0 or completed == total:
                    seen_i_indices = [futures[f] for f in futures if f.done()]
                    em_so_far = sum(em_results[idx] for idx in seen_i_indices)
                    em_pct = em_so_far / completed * 100
                    ex_pct = (running_exec_ok / running_exec_at * 100 if running_exec_at > 0 else 0.0)
                    print(f"  [{completed:4d}/{total}] EM:{em_pct:5.1f}%  EX:{ex_pct:5.1f}%  (out-of-order)")

        detailed_results = []
        for i in range(total):
            r = exec_results.get(original_indices[i], {})
            p = predictions[i]
            sample = samples[i]
            exec_match = r.get("exec_match", False)
            em_match = em_results[i]
            detailed_results.append({
                "idx": original_indices[i],
                "question": sample["question"],
                "db_id": p["db_id"],
                "gold_sql": sample["sql"],
                "pred_sql": p["pred_validated"],
                "em_match": em_match,
                "exec_match": exec_match,
                "pred_failed": r.get("pred_failed", False),
                "gold_failed": r.get("gold_failed", False),
            })
            if not r.get("exec_attempted", False):
                continue
            exec_attempted += 1
            if r.get("exec_match"):
                exec_correct += 1
            if r.get("pred_failed"):
                exec_pred_fail += 1
            if r.get("gold_failed"):
                exec_gold_fail += 1

        with open("eval_predictions.json", "w", encoding="utf-8") as f:
            json.dump(detailed_results, f, indent=2)
        phase2_time = time.time() - phase2_start
        print(f"\nPhase 2 done in {phase2_time:.1f}s\n")

    _pool.close_all()
    eval_time = time.time() - phase1_start
    em_accuracy = em_correct / total * 100 if total else 0
    ex_accuracy = exec_correct / exec_attempted * 100 if exec_attempted > 0 else 0
    avg_latency = sum(latencies) / len(latencies) if latencies else 0
    sorted_lat = sorted(latencies)
    p50 = sorted_lat[len(sorted_lat) // 2] if sorted_lat else 0
    p95 = sorted_lat[int(len(sorted_lat) * 0.95)] if sorted_lat else 0

    print("\n" + "=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)
    print(f"Model:              {MODEL_ID}")
    print(f"Adapter:            {ADAPTER_PATH}")
    print(f"Validation Samples: {total}")
    print()
    print(f"Exact-Match Correct: {em_correct}")
    print(f"Exact-Match Acc:     {em_accuracy:.2f}%")
    print()
    if db_available:
        print(f"Exec Attempted:      {exec_attempted}")
        print(f"Exec Correct:        {exec_correct}")
        print(f"Exec Pred Failures:  {exec_pred_fail}")
        print(f"Exec Gold Failures:  {exec_gold_fail}")
        print(f"Execution Acc:       {ex_accuracy:.2f}%")
    else:
        print("Execution Acc:       N/A (databases not found)")
    print()
    print(f"Total Eval Time:     {eval_time:.1f}s")
    print(f"Avg Latency:         {avg_latency:.3f}s / sample")
    print(f"Median Latency:      {p50:.3f}s")
    print(f"P95 Latency:         {p95:.3f}s")
    print("=" * 60)

if __name__ == "__main__":
    main()