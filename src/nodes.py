from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from src.config import GROQ_API_KEY, LLM_MODEL
from src.vectorstore import load_vectorstore
from src.state import GraphState

llm = ChatGroq(api_key=GROQ_API_KEY, model_name=LLM_MODEL, temperature=0)

# ── Nœud 1 : Retrieve ──────────────────────────
def retrieve(state: GraphState) -> GraphState:
    """Cherche les chunks pertinents dans FAISS"""
    print("🔍 [retrieve]")
    retriever = load_vectorstore().as_retriever(search_kwargs={"k": 4})
    docs = retriever.invoke(state["question"])
    return {**state, "documents": docs}

# ── Nœud 2 : Grade Documents ───────────────────
def grade_documents(state: GraphState) -> GraphState:
    """Filtre les chunks non pertinents"""
    print("📊 [grade_documents]")
    prompt = PromptTemplate.from_template("""
Tu es un évaluateur strict.
Ce document est-il pertinent pour répondre à la question ?
Réponds uniquement par "yes" ou "no", rien d'autre.

Document : {document}
Question : {question}
Réponse :""")
    chain = prompt | llm | StrOutputParser()

    relevant_docs = []
    for doc in state["documents"]:
        score = chain.invoke({
            "document": doc.page_content,
            "question": state["question"]
        })
        if score.strip().lower() == "yes":
            relevant_docs.append(doc)

    relevance = "yes" if relevant_docs else "no"
    print(f"   → {len(relevant_docs)}/{len(state['documents'])} docs pertinents")
    return {**state, "documents": relevant_docs, "relevance_check": relevance}

# ── Nœud 3 : Generate ──────────────────────────
def generate(state: GraphState) -> GraphState:
    """Génère la réponse finale"""
    print("💬 [generate]")
    context = "\n\n".join(doc.page_content for doc in state["documents"])
    prompt = PromptTemplate.from_template("""
Tu es un assistant expert. Réponds uniquement à partir du contexte.
Si la réponse n'est pas dans le contexte, dis-le clairement.

Contexte : {context}
Question : {question}
Réponse :""")
    chain = prompt | llm | StrOutputParser()
    generation = chain.invoke({
        "context": context,
        "question": state["question"]
    })
    return {**state, "generation": generation}

# ── Nœud 4 : Rewrite Query ─────────────────────
def rewrite_query(state: GraphState) -> GraphState:
    """Reformule la question si aucun doc pertinent"""
    print("✏️ [rewrite_query]")
    prompt = PromptTemplate.from_template("""
La question suivante n'a trouvé aucun document pertinent.
Reformule-la pour améliorer la recherche. Sois concis.

Question originale : {question}
Nouvelle question :""")
    chain = prompt | llm | StrOutputParser()
    new_question = chain.invoke({"question": state["question"]})
    print(f"   → {new_question}")
    return {
        **state,
        "question": new_question,
        "retry_count": state.get("retry_count", 0) + 1
    }

# ── Nœud 5 : Fallback ──────────────────────────
def fallback(state: GraphState) -> GraphState:
    """Réponse par défaut si trop de retries"""
    print("🆘 [fallback]")
    return {
        **state,
        "generation": "Je n'ai pas trouvé de réponse pertinente dans les documents."
    }