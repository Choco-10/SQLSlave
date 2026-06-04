from __future__ import annotations

import requests
import streamlit as st


st.set_page_config(
    page_title="Text-to-SQL Generator",
    page_icon="SQL",
    layout="centered",
    initial_sidebar_state="collapsed",
)


st.title("Text-to-SQL Generator")
st.write("Define schema with PK/FK constraints and generate SQL from natural language.")

BACKEND_URL = "http://127.0.0.1:8000/generate"


# -----------------------------
# STATE INIT
# -----------------------------
def _init_schema_state() -> None:
    if "schema_table_ids" not in st.session_state:
        st.session_state.schema_table_ids = []

    if "schema_next_id" not in st.session_state:
        st.session_state.schema_next_id = 1

    if "schema_table_column_ids" not in st.session_state:
        st.session_state.schema_table_column_ids = {}

    if "schema_table_next_column_id" not in st.session_state:
        st.session_state.schema_table_next_column_id = {}


def _ensure_table_column_state(table_id: int) -> None:
    if table_id not in st.session_state.schema_table_column_ids:
        st.session_state.schema_table_column_ids[table_id] = []

    if table_id not in st.session_state.schema_table_next_column_id:
        st.session_state.schema_table_next_column_id[table_id] = 1


# -----------------------------
# TABLE / COLUMN MANAGEMENT
# -----------------------------
def _add_schema_table() -> None:
    table_id = st.session_state.schema_next_id
    st.session_state.schema_next_id += 1

    st.session_state.schema_table_ids.append(table_id)
    st.session_state.schema_table_column_ids[table_id] = []
    st.session_state.schema_table_next_column_id[table_id] = 1


def _add_schema_column(table_id: int) -> None:
    _ensure_table_column_state(table_id)

    cid = st.session_state.schema_table_next_column_id[table_id]
    st.session_state.schema_table_next_column_id[table_id] += 1

    st.session_state.schema_table_column_ids[table_id].append(cid)


def _remove_schema_column(table_id: int, column_id: int) -> None:
    cols = [
        c for c in st.session_state.schema_table_column_ids[table_id]
        if c != column_id
    ]
    st.session_state.schema_table_column_ids[table_id] = cols

    for key in [
        f"column_name_{table_id}_{column_id}",
        f"pk_{table_id}_{column_id}",
        f"fk_enabled_{table_id}_{column_id}",
        f"fk_table_{table_id}_{column_id}",
        f"fk_column_{table_id}_{column_id}",
    ]:
        st.session_state.pop(key, None)


def _remove_schema_table(table_id: int) -> None:
    st.session_state.schema_table_ids = [
        t for t in st.session_state.schema_table_ids if t != table_id
    ]

    for cid in st.session_state.schema_table_column_ids.get(table_id, []):
        _remove_schema_column(table_id, cid)

    st.session_state.pop(f"table_name_{table_id}", None)
    st.session_state.schema_table_column_ids.pop(table_id, None)
    st.session_state.schema_table_next_column_id.pop(table_id, None)


# -----------------------------
# BUILD SCHEMA
# -----------------------------
def _build_schema_payload():
    schema = []

    for table_id in st.session_state.schema_table_ids:
        _ensure_table_column_state(table_id)

        table_name = st.session_state.get(f"table_name_{table_id}", "").strip()

        columns = {}

        for cid in st.session_state.schema_table_column_ids[table_id]:
            col = st.session_state.get(f"column_name_{table_id}_{cid}", "").strip()
            if not col:
                continue

            pk = st.session_state.get(f"pk_{table_id}_{cid}", False)

            fk_enabled = st.session_state.get(f"fk_enabled_{table_id}_{cid}", False)
            fk_table = st.session_state.get(f"fk_table_{table_id}_{cid}")
            fk_column = st.session_state.get(f"fk_column_{table_id}_{cid}")

            fk = None
            if fk_enabled and fk_table and fk_column:
                fk = (fk_table, fk_column)

            columns[col] = {"pk": pk, "fk": fk}

        if table_name or columns:
            schema.append({"name": table_name, "columns": columns})

    return schema


# -----------------------------
# INIT
# -----------------------------
_init_schema_state()

question = st.text_area(
    "Question",
    height=120,
    placeholder="Ask your SQL question here..."
)


# -----------------------------
# UI
# -----------------------------
st.subheader("Schema Builder")

for table_id in list(st.session_state.schema_table_ids):
    _ensure_table_column_state(table_id)

    with st.container(border=True):
        top1, top2 = st.columns([5, 1])

        top1.markdown(f"**Table {table_id}**")

        if top2.button("❌", key=f"rm_table_{table_id}"):
            _remove_schema_table(table_id)
            st.rerun()

        st.text_input(
            "Table name",
            key=f"table_name_{table_id}",
            placeholder="table_name"
        )

        st.markdown("Columns")

        for cid in list(st.session_state.schema_table_column_ids[table_id]):
            c1, c2, c3, c4, c5 = st.columns([4, 1, 1, 3, 1])

            c1.text_input(
                "Column",
                key=f"column_name_{table_id}_{cid}",
                placeholder="",
                label_visibility="collapsed"
            )

            c2.checkbox("PK", key=f"pk_{table_id}_{cid}")

            fk_enabled = c3.checkbox("FK", key=f"fk_enabled_{table_id}_{cid}")

            if fk_enabled:
                tables = [
                    st.session_state.get(f"table_name_{t}", "").strip()
                    for t in st.session_state.schema_table_ids
                ]
                tables = [t for t in tables if t]

                fk_table = c4.selectbox(
                    "Table",
                    tables if tables else ["(none)"],
                    key=f"fk_table_{table_id}_{cid}"
                )

                pk_cols = []
                for t in st.session_state.schema_table_ids:
                    if st.session_state.get(f"table_name_{t}", "").strip() != fk_table:
                        continue

                    for ocid in st.session_state.schema_table_column_ids[t]:
                        cname = st.session_state.get(f"column_name_{t}_{ocid}", "").strip()
                        if st.session_state.get(f"pk_{t}_{ocid}", False) and cname:
                            pk_cols.append(cname)

                if pk_cols:
                    c4.selectbox(
                        "Ref",
                        pk_cols,
                        key=f"fk_column_{table_id}_{cid}"
                    )
                else:
                    c4.caption("No PK found")

            # aligned delete button (right-most narrow column)
            if c5.button("❌", key=f"rm_col_{table_id}_{cid}"):
                _remove_schema_column(table_id, cid)
                st.rerun()

        if st.button("Add column", key=f"add_col_{table_id}"):
            _add_schema_column(table_id)
            st.rerun()


if st.button("Add table"):
    _add_schema_table()
    st.rerun()


# -----------------------------
# PREVIEW
# -----------------------------
st.divider()
st.subheader("Schema Preview")

schema = _build_schema_payload()

for t in schema:
    st.markdown(f"### {t['name']}")

    rows = [
        {"column": c, "pk": v["pk"], "fk": v["fk"]}
        for c, v in t["columns"].items()
    ]

    st.dataframe(rows, width="stretch")


# -----------------------------
# GENERATE
# -----------------------------
if st.button("Generate SQL", type="primary"):
    if not question.strip():
        st.error("Enter a question.")
    else:
        with st.spinner("Generating SQL..."):
            res = requests.post(
                BACKEND_URL,
                json={"question": question, "schema": _build_schema_payload()},
                timeout=120
            )

        if res.ok:
            st.subheader("SQL")
            st.code(res.json().get("sql", ""), language="sql")
        else:
            st.error(res.text)