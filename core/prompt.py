
from __future__ import annotations

from collections.abc import Mapping, Sequence


SYSTEM_PROMPT = (
    "You are a STRICT Text-to-SQL engine.\n"
    "RULES:\n"
    "1. Use ONLY the exact table and column names defined in the schema. Do NOT invent, rename, or alias columns/tables to non-schema elements.\n"
    "2. If the schema has insufficient information to answer the question, output ONLY: SELECT 1;\n"
    "3. Output ONLY the raw SQL query. Do not include markdown formatting or explanations.\n"
    "4. When grouping by an entity, select its descriptive attribute in the SELECT clause but group by its primary key (ID) column to handle duplicate names correctly.\n"
    "5. For set differences (e.g., 'but do not have') or relationships, NEVER use EXCEPT/INTERSECT on non-primary key columns. Instead, use subqueries with IN, NOT IN, or EXISTS on the entity's primary key (ID) column.\n"
    "6. WHERE clause string values for countries, continents, types, categories, and genders MUST be lowercase. Only proper names/titles should match the question casing.\n"
    "7. The sequence of columns in the SELECT clause MUST match the exact order they are asked in the question. NEVER prepend the GROUP BY key to the SELECT clause unless it is explicitly requested first.\n"
    "8. Follow foreign key join paths strictly. NEVER join tables on columns without an explicit foreign key definition. You MUST trace joins step-by-step through intermediate tables if no direct relationship exists.\n"
    "9. Do not filter a foreign key column directly with a text string. Join the referenced table and filter on the corresponding descriptive column."
)


def format_schema_text(schema: Sequence[Mapping[str, object]] | None) -> str:
    if not schema:
        return ""

    lines = ["Schema:"]

    for table in schema:
        table_name = table.get("name", "").strip()
        if not table_name:
            continue

        col_parts = []
        for col, meta in table.get("columns", {}).items():
            col = str(col).strip()
            if not col:
                continue

            part = col
            col_type = meta.get("type")
            if col_type:
                t_lower = str(col_type).lower().strip()
                if any(x in t_lower for x in ["int", "double", "float", "real", "numeric", "number", "decimal"]):
                    part += " num"
                elif any(x in t_lower for x in ["char", "text", "str", "clob"]):
                    part += " str"
                elif any(x in t_lower for x in ["date", "time", "year"]):
                    part += " date"
                else:
                    part += f" {t_lower}"

            if meta.get("pk"):
                part += " [PK]"

            fk = meta.get("fk")
            if fk and isinstance(fk, (list, tuple)) and len(fk) == 2:
                part += f" [FK -> {fk[0]}.{fk[1]}]"

            col_parts.append(part)

        lines.append(f"{table_name} ({', '.join(col_parts)})")

    return "\n".join(lines)


def build_prompt(question: str, schema: Sequence[Mapping[str, object]] | None = None) -> str:
    question = question.strip()
    schema_text = format_schema_text(schema)

    user_parts = []
    if schema_text:
        user_parts.extend([
            "YOU MUST ONLY USE THIS SCHEMA:",
            schema_text,
            ""
        ])

    user_parts.extend([
        SYSTEM_PROMPT,
        "",
        "Return SQL query using ONLY the schema and the rules above.",
        f"Question: {question}"
    ])

    user_message = "\n".join(user_parts)
    return f"### Instruction:\n{user_message}\n### Response:\n"


def build_training_text(question: str, sql: str, schema=None) -> str:
    return f"{build_prompt(question, schema)}{sql.strip()}<|EOT|>"