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
    print("📊 [grade_documents]")
    prompt = PromptTemplate.from_template("""
Tu es un évaluateur. Ce document contient-il des informations 
utiles pour répondre à la question, même partiellement ?
Réponds uniquement par "yes" ou "no".

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

    # Si aucun doc pertinent, on garde tous les docs quand même
    if not relevant_docs:
        relevant_docs = state["documents"]
        relevance = "yes"
    else:
        relevance = "yes"

    print(f"   → {len(relevant_docs)} docs pertinents")
    return {**state, "documents": relevant_docs, "relevance_check": relevance}


# ── Nœud 3 : Generate ──────────────────────────
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

def generate(state: GraphState) -> GraphState:
    print("💬 [generate]")
    
    if not state["documents"]:
        return {**state, "generation": "Je n'ai pas trouvé de documents pertinents."}
    
    context = "\n\n".join(doc.page_content for doc in state["documents"])
    messages = state.get("messages", [])
    
    system = SystemMessage(content=f"""
Tu es un assistant expert. Réponds en français uniquement à partir du contexte.
Si la réponse n'est pas dans le contexte, dis-le clairement.

Contexte : {context}
""")
    
    human = HumanMessage(content=state["question"])
    all_messages = [system] + messages + [human]
    
    response = llm.invoke(all_messages)
    generation = response.content
    
    updated_messages = messages + [human, AIMessage(content=generation)]
    
    return {**state, "generation": generation, "messages": updated_messages}


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