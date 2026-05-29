from __future__ import annotations


SYSTEM_PROMPT = (
    "You are a text-to-SQL assistant. Convert the user's natural language question into a valid SQL query. "
    "The question may already include table hints such as 'from orders table'. Use those hints only. "
    "Return only SQL and nothing else."
)


def build_prompt(question: str) -> str:
    question = question.strip()
    return (
        f"<s>[INST] <<SYS>>\n{SYSTEM_PROMPT}\n<</SYS>>\n\n"
        f"Question: {question}\n\nSQL:[/INST]"
    )


def build_training_text(question: str, sql: str) -> str:
    return f"{build_prompt(question)} {sql.strip()}"
