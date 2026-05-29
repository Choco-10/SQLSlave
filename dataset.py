from __future__ import annotations

import argparse
import json
import importlib
import logging
import re
from pathlib import Path
from typing import Iterable

try:
    from .prompt import build_training_text
except ImportError:  # pragma: no cover - allows direct script execution
    from prompt import build_training_text


logger = logging.getLogger(__name__)

# Prefer sqlparse for robust identifier extraction when available.
# Import it dynamically so the module stays usable even when the package
# is not installed in the current analysis environment.
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
    # Replace any sequence of non-alphanumeric/underscore with a single underscore
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


def _load_table_map(tables_path: Path) -> dict[str, list[str]]:
    """Load Spider `tables.json` and return mapping db_id -> list of table names."""
    tables = _load_json(tables_path)
    table_map: dict[str, list[str]] = {}
    for db in tables:
        db_id = db.get("db_id")
        if db_id is None:
            logger.warning("Skipping table entry without db_id: %s", db)
            continue
        table_map[db_id] = list(db.get("table_names_original", []))
    return table_map


def _tokenize_sql(sql: str) -> list[str]:
    """Return candidate identifier tokens from SQL.

    This is a lightweight tokenizer that extracts words and dotted names
    (e.g. schema.table). It does not attempt full SQL parsing but avoids
    matching inside other words.
    """
    if not sql:
        return []

    # If sqlparse is available, use it to extract identifiers.
    if _HAS_SQLPARSE:
        try:
            parsed = sqlparse.parse(sql)
        except Exception:
            parsed = []

        tokens: list[str] = []

        def _extract_identifiers(tok_list):
            for tok in tok_list:
                # IdentifierList may contain multiple identifiers
                if IdentifierList and isinstance(tok, IdentifierList):
                    _extract_identifiers(tok.get_identifiers())
                elif Identifier and isinstance(tok, Identifier):
                    # Identifier.get_name() returns the last part (after dot)
                    name = tok.get_real_name() or tok.get_name()
                    if name:
                        tokens.append(name.lower())
                else:
                    # Some tokens may be Name tokens
                    try:
                        if tok.ttype is Name:
                            tokens.append(str(tok).lower())
                    except Exception:
                        # fallback; ignore
                        pass

        for statement in parsed:
            _extract_identifiers(statement.tokens)

        # as a fallback to sqlparse extraction, also capture dotted/name patterns
        if not tokens:
            sql_lc = sql.lower()
            sql_lc = re.sub(r"'([^']*)'", " ", sql_lc)
            sql_lc = re.sub(r'"([^"]*)"', " ", sql_lc)
            for m in re.finditer(r"[a-z_][a-z0-9_]*(?:\.[a-z_][a-z0-9_]*)?", sql_lc):
                tokens.append(m.group(0))

        return tokens

    # No sqlparse: lightweight regex-based token extraction
    sql_lc = sql.lower()
    # strip string literals to avoid matching table-like words inside strings
    sql_lc = re.sub(r"'([^']*)'", " ", sql_lc)
    sql_lc = re.sub(r'"([^"]*)"', " ", sql_lc)
    tokens = []
    for m in re.finditer(r"[a-z_][a-z0-9_]*(?:\.[a-z_][a-z0-9_]*)?", sql_lc):
        tokens.append(m.group(0))
    return tokens


def _extract_table_names(query: str, available_tables: Iterable[str]) -> list[str]:
    """Extract matching table names from SQL by token matching.

    Returns a list of matched original table names. If none are matched,
    returns an empty list (caller may choose a fallback).
    """
    available = [t for t in available_tables if t]
    if not available:
        return []

    normalized_map: dict[str, list[str]] = {}
    for t in available:
        norm = _normalize(t)
        if not norm:
            continue
        normalized_map.setdefault(norm, []).append(t)

    tokens = _tokenize_sql(query)
    matches: list[str] = []

    for tok in tokens:
        # consider dotted forms: schema.table -> check last part
        if "." in tok:
            tok = tok.split(".")[-1]
        tok_norm = _normalize(tok)
        if not tok_norm:
            continue
        if tok_norm in normalized_map:
            for orig in normalized_map[tok_norm]:
                if orig not in matches:
                    matches.append(orig)

    return matches


def add_table_hint(question: str, tables: Iterable[str]) -> str:
    """Append a human-readable table hint to the question.

    Examples:
    - single table: "from users table"
    - two tables: "from users and orders tables"
    - many tables: "from a, b, and c tables"
    If `tables` is empty or falsy, the original question is returned.
    """
    tables = [table for table in tables if table]
    question = question.strip()
    if not tables:
        return question

    if len(tables) == 1:
        hint = f"from {tables[0]} table"
    elif len(tables) == 2:
        hint = f"from {tables[0]} and {tables[1]} tables"
    else:
        # Oxford comma for clarity
        hint = "from " + ", ".join(tables[:-1]) + f", and {tables[-1]} tables"

    if question.lower().endswith(hint.lower()):
        return question
    return f"{question} {hint}".strip()


def convert_spider_split(
    spider_root: str | Path,
    split_file: str,
    tables_file: str = "tables.json",
) -> list[dict[str, str]]:
    spider_root = Path(spider_root)
    table_map = _load_table_map(spider_root / tables_file)
    examples = _load_json(spider_root / split_file)
    records: list[dict[str, str]] = []

    for example in examples:
        question = example["question"].strip()
        sql = example["query"].strip()
        db_id = example["db_id"]
        available_tables = table_map.get(db_id, [])
        table_names = _extract_table_names(sql, available_tables)
        question_with_hint = add_table_hint(question, table_names)
        records.append({"text": build_training_text(question_with_hint, sql)})

    return records


def convert_spider_dataset(
    spider_root: str | Path,
    output_dir: str | Path,
    train_split: str = "train_spider.json",
    validation_split: str = "dev.json",
) -> tuple[Path, Path]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_records = convert_spider_split(spider_root, train_split)
    validation_records = convert_spider_split(spider_root, validation_split)

    train_path = output_dir / "train.jsonl"
    validation_path = output_dir / "validation.jsonl"

    with train_path.open("w", encoding="utf-8") as handle:
        for record in train_records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    with validation_path.open("w", encoding="utf-8") as handle:
        for record in validation_records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    return train_path, validation_path


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
