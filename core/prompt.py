from __future__ import annotations

from collections.abc import Mapping, Sequence


SYSTEM_PROMPT = (
    "You are a STRICT Text-to-SQL engine.\n"
    "RULES:\n"
    "1. Use ONLY tables and columns in the schema.\n"
    "2. NEVER invent tables or columns.\n"
    "3. NEVER rename schema elements.\n"
    "4. If missing info → output: SELECT 1;\n"
    "5. Output ONLY SQL."
)


def format_schema_text(schema: Sequence[Mapping[str, object]] | None) -> str:
    if not schema:
        return ""

    lines = ["Schema (STRICT - DO NOT INVENT ANYTHING):"]

    for table in schema:
        table_name = table.get("name", "").strip()
        if not table_name:
            continue

        lines.append(f"\nTABLE {table_name}")

        for col, meta in table.get("columns", {}).items():
            col = str(col).strip()
            if not col:
                continue

            line = f"- {col}"

            if meta.get("pk"):
                line += " [PRIMARY KEY]"

            fk = meta.get("fk")
            if fk and isinstance(fk, (list, tuple)) and len(fk) == 2:
                line += f" [FK -> {fk[0]}.{fk[1]}]"

            lines.append(line)

    return "\n".join(lines)


def build_prompt(question: str, schema: Sequence[Mapping[str, object]] | None = None) -> str:
    question = question.strip()
    schema_text = format_schema_text(schema)

    parts = [
        "<s>[INST] <<SYS>>",
        SYSTEM_PROMPT,
        "<</SYS>>",
        ""
    ]

    if schema_text:
        parts.extend([
            "YOU MUST ONLY USE THIS SCHEMA:",
            schema_text,
            ""
        ])

    parts.extend([
        "Return SQL query using ONLY the schema above.",
        f"Question: {question}",
        "",
        "SQL:[/INST]"
    ])

    return "\n".join(parts)


def build_training_text(question: str, sql: str, schema=None) -> str:
    return f"{build_prompt(question, schema)} {sql.strip()}"