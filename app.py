import streamlit as st
from src.vectorstore import build_vectorstore
from src.graph import build_graph
import os

st.set_page_config(page_title="LangGraph Chatbot", page_icon="🕸️")
st.title("🕸️ LangGraph Chatbot")

# ── Sidebar ────────────────────────────────────
with st.sidebar:
    st.header("📂 Documents")
    uploaded_files = st.file_uploader("Dépose tes PDFs ici", type="pdf", accept_multiple_files=True)


    if uploaded_files:
        for uploaded_file in uploaded_files:
            save_path = os.path.join("data/docs", uploaded_file.name)
            with open(save_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            st.success(f"✅ {uploaded_file.name} uploadé !")

            with st.spinner("Indexation automatique..."):
                build_vectorstore()
            
            st.success("✅ Tous les documents sont prêts !")
    
        if "graph" in st.session_state:
            del st.session_state["graph"]

# ── Init graphe ────────────────────────────────
if "graph" not in st.session_state:
    try:
        st.session_state.graph = build_graph()
    except FileNotFoundError:
        st.info("👈 Dépose un PDF dans la sidebar pour commencer.")
        st.stop()

# ── Historique messages ────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

# ── Input ──────────────────────────────────────
question = st.chat_input("Pose ta question...")

if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.write(question)

    with st.spinner("Recherche en cours..."):
        result = st.session_state.graph.invoke({
            "question": question,
            "documents": [],
            "generation": "",
            "relevance_check": None,
            "retry_count": 0,
            "messages": st.session_state.get("chat_messages", [])
        })
    st.session_state.chat_messages = result.get("messages", [])

    answer = result["generation"]
    st.session_state.messages.append({"role": "assistant", "content": answer})
    with st.chat_message("assistant"):
        st.write(answer)

    # Debug
    st.write("relevance_check:", result["relevance_check"])
    st.write("retry_count:", result["retry_count"])
    st.write("nb docs trouvés:", len(result["documents"]))

    with st.expander("📄 Source"):
        for doc in result.get("documents", []):
            st.markdown(f"- `{doc.metadata.get('source', '?')}` — page {doc.metadata.get('page', '?')}")