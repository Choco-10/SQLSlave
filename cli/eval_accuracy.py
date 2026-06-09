"""Execution-accuracy evaluation on the Spider validation set.

Runs the fine-tuned model on all 1,034 validation samples, executes both
the predicted and gold SQL against the actual Spider SQLite databases,
and compares the result sets.

Reports:
  - Execution accuracy (result-set match, order-independent per Spider protocol)
  - Exact-match accuracy (normalised string comparison)
  - Average inference latency per sample
  - Total evaluation time

Requires the Spider SQLite databases under ``spider/database/``.
Run ``python -m cli.download_spider`` first (updated version extracts DBs).

Usage:
    python -m cli.eval_accuracy
"""
from __future__ import annotations

import json
import re
import sqlite3
import time
from pathlib import Path

from core.model import load_sql_generator


CONVERTED_VAL = Path("converted_spider/validation.jsonl")
SPIDER_DEV = Path("spider/dev.json")
SPIDER_ROOT = Path("spider")
DB_DIR = SPIDER_ROOT / "database"
MODEL_ID = "codellama/CodeLlama-7b-Instruct-hf"
ADAPTER_PATH = "artifacts/qlora_adapter/checkpoint-1750"

# Safety: max rows to fetch to prevent Cartesian-product OOM
MAX_ROWS = 5000


# ── SQL normalisation for exact-match (structural only) ──────────
# Normalises whitespace, parentheses, operators — but does NOT
# lowercase string literals inside quotes.
_CODE_BLOCK_RE = re.compile(r"```.*?```", re.S)
_SQL_START_RE = re.compile(r"(?is)\b(select|with)\b")


def _normalize_structure(sql: str) -> str:
    """Normalize SQL for exact-match: whitespace, parens, operators.

    Preserves the case of string literals inside single/double quotes
    so that ``WHERE name = 'Alice'`` is NOT turned into ``'alice'``.
    """
    sql = sql.strip().rstrip(";")

    # Split on quoted strings so we can normalise the structural parts
    # without touching literals.
    # Pattern: match either a quoted string or a non-quote run.
    parts = re.split(r"('(?:[^'\\]|\\.)*'|\"(?:[^\"\\]|\\.)*\")", sql)

    normalised: list[str] = []
    for part in parts:
        if (part.startswith("'") and part.endswith("'")) or (
            part.startswith('"') and part.endswith('"')
        ):
            # Preserve the literal as-is (but strip surrounding spaces
            # that may have leaked from the split boundary).
            normalised.append(part.strip())
        else:
            # Structural part — lowercase and compress whitespace
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


def extract_sql(text: str) -> str:
    """Extract the first SELECT/WITH statement from raw LLM output."""
    text = _CODE_BLOCK_RE.sub("", text).lstrip()
    m = _SQL_START_RE.search(text)
    if m:
        text = text[m.start():]
    sc = text.find(";")
    if sc != -1:
        text = text[:sc]
    return text.strip()


# ── Load dev.json to get db_id per sample ────────────────────────
def _load_dev_db_ids(dev_path: Path) -> list[str]:
    """Return the db_id for each sample in dev.json, in order."""
    with dev_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    return [ex["db_id"] for ex in data]


def load_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            yield json.loads(line)


# ── Safe SQL execution ───────────────────────────────────────────
def execute_sql(
    db_path: Path, sql: str, timeout: int = 30, sort: bool = True
) -> list[tuple] | None:
    """Execute *sql* against the SQLite database and return rows.

    Parameters
    ----------
    sort : bool
        If True (default, for Spider protocol), sort rows before
        returning so comparison is order-independent.  Set to False
        when you want to honour ORDER BY.
    """
    # Guard: inject LIMIT if the model forgot one, to prevent OOM from
    # accidental Cartesian products.
    sql_upper = sql.upper().rstrip().rstrip(";")
    if not re.search(r"\bLIMIT\b", sql_upper):
        sql = sql.rstrip().rstrip(";") + f" LIMIT {MAX_ROWS}"

    try:
        conn = sqlite3.connect(str(db_path), timeout=timeout)
        conn.execute("PRAGMA journal_mode=WAL")
        cursor = conn.execute(sql)
        rows = cursor.fetchall()
        conn.close()
        if sort:
            # Spider protocol: sort both result sets for comparison
            return sorted(rows, key=lambda r: tuple(
                ("" if v is None else str(v)) for v in r
            ))
        return rows
    except Exception:
        return None


def result_sets_match(
    pred_rows: list[tuple], gold_rows: list[tuple], ordered: bool = False
) -> bool:
    """Compare two result sets.

    When *ordered* is True, rows must match positionally.
    When False (Spider default), row order is ignored.
    """
    if len(pred_rows) != len(gold_rows):
        return False

    if ordered:
        rows_iter = zip(pred_rows, gold_rows)
    else:
        # Spider protocol: sort both sets so comparison is order-independent
        def sort_key(row):
            return tuple("" if v is None else str(v) for v in row)
        rows_iter = zip(sorted(pred_rows, key=sort_key), sorted(gold_rows, key=sort_key))

    for p_row, g_row in rows_iter:
        if len(p_row) != len(g_row):
            return False
        for p_val, g_val in zip(p_row, g_row):
            if str(p_val) != str(g_val):
                return False
    return True


# ── Main ─────────────────────────────────────────────────────────
def main():
    if not CONVERTED_VAL.exists():
        print(f"ERROR: Validation file not found at {CONVERTED_VAL}")
        print("Run 'python -m cli.download_spider' first.")
        return

    if not SPIDER_DEV.exists():
        print(f"ERROR: Spider dev.json not found at {SPIDER_DEV}")
        print("Run 'python -m cli.download_spider' first.")
        return

    db_available = DB_DIR.exists() and any(DB_DIR.iterdir())
    if not db_available:
        print(f"WARNING: Spider databases not found at {DB_DIR}")
        print("Execution accuracy will be skipped; only exact-match reported.\n")

    # Load db_ids from the original dev.json (converted JSONL lacks db_id)
    db_ids = _load_dev_db_ids(SPIDER_DEV)

    print("Loading model …\n")
    generator = load_sql_generator(
        base_model_id=MODEL_ID,
        adapter_path=ADAPTER_PATH,
    )

    samples = list(load_jsonl(CONVERTED_VAL))
    if len(samples) != len(db_ids):
        print(
            f"WARNING: sample count mismatch "
            f"(converted={len(samples)}, dev.json={len(db_ids)}). "
            f"Using min."
        )
    total = min(len(samples), len(db_ids))

    em_correct = 0
    exec_correct = 0
    exec_attempted = 0
    exec_pred_fail = 0
    exec_gold_fail = 0
    latencies: list[float] = []

    print(f"Evaluating on {total} samples …\n")
    eval_start = time.time()

    for i in range(total):
        sample = samples[i]
        db_id = db_ids[i]
        question = sample["question"]
        gold_sql = sample["sql"]
        schema = sample.get("schema", [])

        t0 = time.time()
        pred_sql_raw = generator.generate(question, schema)
        latencies.append(time.time() - t0)

        # ── Clean the raw LLM output ──
        pred_extracted = extract_sql(pred_sql_raw)
        pred_clean = _normalize_structure(pred_extracted)
        gold_clean = _normalize_structure(gold_sql)

        # ── Exact-match ──
        em_match = pred_clean == gold_clean
        if em_match:
            em_correct += 1

        # ── Execution accuracy ──
        exec_match = False
        if db_available and db_id:
            db_path = DB_DIR / db_id / f"{db_id}.sqlite"
            if not db_path.exists():
                db_files = (
                    list((DB_DIR / db_id).glob("*.sqlite"))
                    if (DB_DIR / db_id).exists()
                    else []
                )
                db_path = db_files[0] if db_files else None

            if db_path and db_path.exists():
                exec_attempted += 1

                # BUG 1 FIX: pass cleaned SQL, not raw LLM output
                pred_rows = execute_sql(db_path, pred_extracted, sort=True)
                gold_rows = execute_sql(db_path, gold_sql, sort=True)

                if pred_rows is None:
                    exec_pred_fail += 1
                if gold_rows is None:
                    exec_gold_fail += 1

                if pred_rows is not None and gold_rows is not None:
                    exec_match = result_sets_match(pred_rows, gold_rows)
                elif pred_rows is None and gold_rows is None:
                    # Both failed — count as mismatch, not match
                    exec_match = False

                if exec_match:
                    exec_correct += 1

        # ── Live output ──
        em_icon = "✅" if em_match else "❌"
        ex_icon = "✅" if exec_match else ("⏭️" if not db_available else "❌")
        em_pct = em_correct / (i + 1) * 100
        ex_pct = exec_correct / exec_attempted * 100 if exec_attempted > 0 else 0
        avg_lat = sum(latencies) / len(latencies)
        remaining = avg_lat * (total - i - 1)
        eta_m, eta_s = divmod(int(remaining), 60)
        print(
            f"[{i+1:4d}/{total}] EM:{em_icon} EX:{ex_icon}  "
            f"EM:{em_pct:5.1f}%  EX:{ex_pct:5.1f}%  |  "
            f"Avg:{avg_lat:.2f}s  |  "
            f"ETA:{eta_m}m{eta_s:02d}s"
        )

    eval_time = time.time() - eval_start
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
        print(f"Execution Acc:       N/A (databases not found)")
    print()
    print(f"Total Eval Time:     {eval_time:.1f}s")
    print(f"Avg Latency:         {avg_latency:.3f}s / sample")
    print(f"Median Latency:      {p50:.3f}s")
    print(f"P95 Latency:         {p95:.3f}s")
    print("=" * 60)


if __name__ == "__main__":
    main()