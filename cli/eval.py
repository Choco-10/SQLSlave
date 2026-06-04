import time
import json
import re

from core.model import load_sql_generator

CODE_BLOCK_RE = re.compile(r"```.*?```", re.S)
SQL_START_RE = re.compile(r"(?is)\b(select|with)\b")
SPACE_RE = re.compile(r"\s+")
COMMA_RE = re.compile(r"\s*,\s*")
LPAREN_RE = re.compile(r"\s*\(\s*")
RPAREN_RE = re.compile(r"\s*\)\s*")
EQ_RE = re.compile(r"\s*=\s*")
GT_RE = re.compile(r"\s*>\s*")
LT_RE = re.compile(r"\s*<\s*")


def normalize_sql(sql: str) -> str:
    sql = sql.lower().strip().rstrip(";")
    sql = SPACE_RE.sub(" ", sql)
    sql = COMMA_RE.sub(",", sql)
    sql = LPAREN_RE.sub("(", sql)
    sql = RPAREN_RE.sub(")", sql)
    sql = EQ_RE.sub("=", sql)
    sql = GT_RE.sub(">", sql)
    sql = LT_RE.sub("<", sql)
    return sql.strip()


def extract_sql_match(pred: str) -> str:
    pred = CODE_BLOCK_RE.sub("", pred).lstrip()

    m = SQL_START_RE.search(pred)
    if m:
        pred = pred[m.start():]

    semicolon = pred.find(";")
    if semicolon != -1:
        pred = pred[:semicolon]

    return pred.strip()


def load_jsonl(path: str):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            yield json.loads(line)


def format_eta(seconds: float) -> str:
    if seconds < 0:
        return "-"
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def evaluate(model_path: str, data_path: str):
    print("\nLoading model...\n")

    generator = load_sql_generator(
        base_model_id=model_path,
        adapter_path="artifacts/qlora_adapter"
    )

    total = 0
    correct = 0

    start_time = time.time()

    print("Running evaluation...\n")

    for sample in load_jsonl(data_path):
        question = sample["question"]
        gold_sql = sample["sql"]
        schema = sample.get("schema", [])

        pred_sql = generator.generate(question, schema)

        pred_clean = normalize_sql(extract_sql_match(pred_sql))
        gold_clean = normalize_sql(gold_sql)

        total += 1
        if pred_clean == gold_clean:
            correct += 1

        now = time.time()
        elapsed = now - start_time
        avg_time = elapsed / total
        remaining = avg_time * (1034 - total)  # assumes ~1034 max

        acc = (correct / total) * 100

        # print every 10 samples OR first few OR last phase
        if total % 1 == 0 or total < 10:
            print(
                f"[{total}] "
                f"Correct: {correct} | "
                f"Wrong: {total - correct} | "
                f"Acc: {acc:.2f}% | "
                f"Speed: {avg_time:.2f}s/sample | "
                f"ETA: {format_eta(remaining)}"
            )

    accuracy = (correct / total) * 100 if total else 0

    print("\n==============================")
    print(f"Total samples : {total}")
    print(f"Correct       : {correct}")
    print(f"Accuracy      : {accuracy:.2f}%")
    print("==============================\n")


if __name__ == "__main__":
    evaluate(
        model_path="codellama/CodeLlama-7b-Instruct-hf",
        data_path="converted_spider/validation.jsonl"
    )