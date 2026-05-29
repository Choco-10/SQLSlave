from __future__ import annotations

import requests
import streamlit as st


st.set_page_config(page_title="Text-to-SQL Generator", page_icon="SQL", layout="centered")

st.title("Text-to-SQL Generator")
st.write("Enter a natural language question that includes the table hint in the text.")

backend_url = st.text_input("Backend URL", value="http://127.0.0.1:8000/generate")
question = st.text_area(
    "Question",
    value="Show total sales by region from orders table",
    height=120,
)


if st.button("Generate SQL", type="primary"):
    if not question.strip():
        st.error("Please enter a question.")
    else:
        with st.spinner("Generating SQL..."):
            response = requests.post(backend_url, json={"question": question}, timeout=120)
        if response.ok:
            sql = response.json().get("sql", "")
            st.subheader("Generated SQL")
            st.code(sql, language="sql")
        else:
            st.error(f"Request failed: {response.status_code}")
            st.text(response.text)
