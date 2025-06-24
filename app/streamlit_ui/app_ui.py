import streamlit as st
import requests
import uuid

# ---------------- UI CONFIG ---------------- #
st.set_page_config(page_title="LLM Assistant", layout="centered")
st.title("LLM Assistant")
st.markdown("Ask a question and get answers powered by GPT, RAG, or intent classification.")

# ---------------- Sidebar Mode ---------------- #
mode = st.sidebar.selectbox("Mode", ["LLM Chat", "Intent Debug", "Raw RAG", "Raw GPT"])

# ---------------- INPUT ---------------- #
with st.form("question_form"):
    user_input = st.text_input("Your Question", placeholder="E.g., What is the return policy for perishables?")
    submit_button = st.form_submit_button("Submit")

# ---------------- API CALL ---------------- #
if submit_button and user_input:
    with st.spinner("Thinking..."):
        try:
            trace_id = str(uuid.uuid4())
            response = requests.post(
                url="http://127.0.0.1:8000/",  
                json={"question": user_input},
                headers={"X-Trace-ID": trace_id},
                timeout=10
            )

            if response.status_code == 200:
                result = response.json()

                # Display
                st.success("Response received")
                st.markdown(f"**Intent**: `{result.get('intent')}`")
                st.markdown(f"**Answer**: {result.get('response') or 'No answer generated.'}")

                if result.get("source_docs"):
                    st.markdown("**Source Documents:**")
                    for i, doc in enumerate(result["source_docs"], start=1):
                        st.code(doc, language="markdown")

                st.caption(f"Trace ID: `{result.get('trace_id')}`")

            else:
                st.error(f"Request failed with status code {response.status_code}")
                st.text(response.text)

        except requests.exceptions.RequestException as e:
            st.error(f"Request failed: {str(e)}")
