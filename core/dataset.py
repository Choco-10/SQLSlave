from __future__ import annotations

import argparse
import json
import importlib
import logging
import re
from pathlib import Path
from typing import Any

try:
    from .prompt import build_training_text, build_prompt
except ImportError:  # pragma: no cover - allows direct script execution
    from core.prompt import build_training_text, build_prompt


logger = logging.getLogger(__name__)

try:
    sqlparse = importlib.import_module("sqlparse")
    sqlparse_sql = importlib.import_module("sqlparse.sql")
    sqlparse_tokens = importlib.import_module("sqlparse.tokens")
    Identifier = getattr(sqlparse_sql, "Identifier")
    IdentifierList = getattr(sqlparse_sql, "IdentifierList")
    Name = getattr(sqlparse_tokens, "Name")
    _HAS_SQLPARSE = True
except Exception:
    sqlparse = None  # type: ignore
    Identifier = None  # type: ignore
    IdentifierList = None  # type: ignore
    Name = None  # type: ignore
    _HAS_SQLPARSE = False


def _normalize(text: str) -> str:
    """Normalize a name for token-level comparison.

    Replaces non-word characters with a single underscore and lowercases.
    This preserves separators so matching is done on tokens, not by
    collapsing everything together.
    """
    text = text or ""
    return re.sub(r"[^0-9a-zA-Z_]+", "_", text).strip("_").lower()


def _load_json(path: Path):
    """Load JSON file with basic error context for easier debugging."""
    try:
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle)
    except FileNotFoundError:
        logger.error("JSON file not found: %s", path)
        raise
    except json.JSONDecodeError as exc:
        logger.error("Invalid JSON in %s: %s", path, exc)
        raise


def _load_schema_map(tables_path: Path) -> dict[str, list[dict[str, object]]]:
    tables = _load_json(tables_path)
    schema_map: dict[str, list[dict[str, object]]] = {}

    for db in tables:
        db_id = db.get("db_id")
        if db_id is None:
            continue

        table_names = list(db.get("table_names_original", []))
        column_names = list(db.get("column_names_original", []))
        column_types = list(db.get("column_types", []))
        primary_keys = set(db.get("primary_keys", []))
        foreign_keys = list(db.get("foreign_keys", []))

        column_lookup: dict[int, tuple[int, str]] = {}

        table_columns: dict[int, dict[str, dict]] = {
            i: {} for i in range(len(table_names))
        }

        for col_idx, (table_idx, col_name) in enumerate(column_names):
            if table_idx < 0 or col_name == "*":
                continue

            column_lookup[col_idx] = (table_idx, col_name)

            col_type = column_types[col_idx] if col_idx < len(column_types) else None

            if 0 <= table_idx < len(table_names):
                col_meta = {}
                if col_idx in primary_keys:
                    col_meta["pk"] = True
                if col_type:
                    col_meta["type"] = col_type
                table_columns[table_idx][col_name] = col_meta

        for left_idx, right_idx in foreign_keys:
            left = column_lookup.get(left_idx)
            right = column_lookup.get(right_idx)

            if not left or not right:
                continue

            l_table, l_col = left
            r_table, r_col = right

            if l_table in table_columns:
                if l_col in table_columns[l_table]:
                    table_columns[l_table][l_col]["fk"] = [
                        table_names[r_table],
                        r_col
                    ]

            if r_table in table_columns:
                if r_col in table_columns[r_table]:
                    table_columns[r_table][r_col]["fk"] = [
                        table_names[l_table],
                        l_col
                    ]

        schema: list[dict[str, object]] = []

        for i, table_name in enumerate(table_names):
            schema.append({
                "name": table_name,
                "columns": table_columns[i]
            })

        schema_map[db_id] = schema

    return schema_map


def _tokenize_sql(sql: str) -> list[str]:
    """Return candidate identifier tokens from SQL.

    This is a lightweight tokenizer that extracts words and dotted names
    (e.g. schema.table). It does not attempt full SQL parsing but avoids
    matching inside other words.
    """
    if not sql:
        return []

    if _HAS_SQLPARSE:
        try:
            parsed = sqlparse.parse(sql)
        except Exception:
            parsed = []

        tokens: list[str] = []

        def _extract_identifiers(tok_list):
            for tok in tok_list:
                if IdentifierList and isinstance(tok, IdentifierList):
                    _extract_identifiers(tok.get_identifiers())
                elif Identifier and isinstance(tok, Identifier):
                    name = tok.get_real_name() or tok.get_name()
                    if name:
                        tokens.append(name.lower())
                else:
                    try:
                        if tok.ttype is Name:
                            tokens.append(str(tok).lower())
                    except Exception:
                        pass

        for statement in parsed:
            _extract_identifiers(statement.tokens)

        if not tokens:
            sql_lc = sql.lower()
            sql_lc = re.sub(r"'([^']*)'", " ", sql_lc)
            sql_lc = re.sub(r'"([^"]*)"', " ", sql_lc)
            for m in re.finditer(r"[a-z_][a-z0-9_]*(?:\.[a-z_][a-z0-9_]*)?", sql_lc):
                tokens.append(m.group(0))

        return tokens

    sql_lc = sql.lower()
    sql_lc = re.sub(r"'([^']*)'", " ", sql_lc)
    sql_lc = re.sub(r'"([^"]*)"', " ", sql_lc)
    tokens = []
    for m in re.finditer(r"[a-z_][a-z0-9_]*(?:\.[a-z_][a-z0-9_]*)?", sql_lc):
        tokens.append(m.group(0))
    return tokens


def get_used_tables(schema: list[dict[str, Any]], sql: str) -> list[str]:
    """Identify which tables from the schema are used in the SQL query."""
    used = []
    sql_lower = sql.lower()
    for table in schema:
        name = table.get("name", "")
        if not name:
            continue
        pattern = r"\b" + re.escape(name.lower()) + r"\b"
        if re.search(pattern, sql_lower):
            used.append(name)
    return used


def prune_schema(schema: list[dict[str, Any]], sql: str, max_tables: int = 8) -> list[dict[str, Any]]:
    """Prune schema to keep all used tables + random decoy tables up to max_tables."""
    used_names = get_used_tables(schema, sql)
    if not used_names:
        return schema[:max_tables]

    used_tables = [t for t in schema if t.get("name") in used_names]
    other_tables = [t for t in schema if t.get("name") not in used_names]

    used_cols = sum(len(t.get("columns", {})) for t in used_tables)
    
    if used_cols > 35:
        max_tables = max(len(used_tables) + 1, 4)
    if used_cols > 50:
        max_tables = len(used_tables)

    if len(schema) <= max_tables:
        return schema

    num_decoys = max(0, max_tables - len(used_tables))

    import random
    rng = random.Random(hash(sql))
    decoys = rng.sample(other_tables, min(num_decoys, len(other_tables)))

    pruned = used_tables + decoys
    name_to_idx = {t.get("name", ""): idx for idx, t in enumerate(schema)}
    pruned.sort(key=lambda t: name_to_idx.get(t.get("name", ""), 999))

    return pruned


def convert_spider_split(
    spider_root: str | Path,
    split_file: str,
    tables_file: str = "tables.json",
    is_training: bool = False,
    max_schema_tables: int = 8,
    tokenizer: Any = None,
) -> list[dict[str, Any]]:

    spider_root = Path(spider_root)
    schema_map = _load_schema_map(spider_root / tables_file)
    examples = _load_json(spider_root / split_file)

    records = []

    for ex in examples:
        question = ex["question"].strip()
        sql = ex["query"].strip()
        db_id = ex["db_id"]

        schema = schema_map.get(db_id, [])
        if is_training:
            schema = prune_schema(schema, sql, max_tables=max_schema_tables)

        prompt_text = build_prompt(question, schema)
        completion_text = sql.strip() + "<|EOT|>"

        if is_training and tokenizer is not None:
            token_count = len(tokenizer.encode(prompt_text + completion_text))
            if token_count > 900:
                schema = prune_schema(schema, sql, max_tables=0)
                prompt_text = build_prompt(question, schema)

        records.append({
            "question": question,
            "sql": sql,
            "schema": schema,
            "text": prompt_text + completion_text,
            "prompt": prompt_text,
            "completion": completion_text,
        })

    return records

def convert_spider_dataset(
    spider_root: str | Path,
    output_dir: str | Path,
    train_split: str = "train_spider.json",
    validation_split: str = "dev.json",
    max_schema_tables: int = 8,
) -> tuple[Path, Path]:

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    from core.model import load_corrected_tokenizer
    try:
        tokenizer = load_corrected_tokenizer("deepseek-ai/deepseek-coder-6.7b-instruct")
    except Exception:
        tokenizer = None

    train_records = convert_spider_split(
        spider_root,
        train_split,
        is_training=True,
        max_schema_tables=max_schema_tables,
        tokenizer=tokenizer,
    )
    val_records = convert_spider_split(spider_root, validation_split, is_training=False)

    train_path = output_dir / "train.jsonl"
    val_path = output_dir / "validation.jsonl"

    def write(path, data):
        with path.open("w", encoding="utf-8") as f:
            for row in data:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

    write(train_path, train_records)
    write(val_path, val_records)

    return train_path, val_path

def main() -> None:
    parser = argparse.ArgumentParser(description="Convert Spider into NL-question to SQL pairs.")
    parser.add_argument("--spider-root", required=True, help="Path to the Spider dataset root.")
    parser.add_argument("--output-dir", required=True, help="Directory to write converted JSONL files.")
    parser.add_argument("--train-split", default="train_spider.json")
    parser.add_argument("--validation-split", default="dev.json")
    args = parser.parse_args()

    train_path, validation_path = convert_spider_dataset(
        spider_root=args.spider_root,
        output_dir=args.output_dir,
        train_split=args.train_split,
        validation_split=args.validation_split,
    )
    print(f"Wrote {train_path}")
    print(f"Wrote {validation_path}")


if __name__ == "__main__":
    main()