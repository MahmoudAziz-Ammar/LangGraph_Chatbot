from langgraph.graph import StateGraph, END
from src.state import GraphState
from src.nodes import retrieve, generate, grade_documents, rewrite_query, fallback
from src.edges import route_after_grading

def build_graph():
    g = StateGraph(GraphState)

    # ── Ajout des nœuds ────────────────────────
    g.add_node("retrieve",        retrieve)
    g.add_node("grade_documents", grade_documents)
    g.add_node("generate",        generate)
    g.add_node("rewrite_query",   rewrite_query)
    g.add_node("fallback",        fallback)

    # ── Edges fixes ────────────────────────────
    g.set_entry_point("retrieve")
    g.add_edge("retrieve",       "grade_documents")
    g.add_edge("rewrite_query",  "retrieve")        # boucle retry
    g.add_edge("generate",       END)
    g.add_edge("fallback",       END)

    # ── Edge conditionnel ──────────────────────
    g.add_conditional_edges(
        "grade_documents",
        route_after_grading,
        {
            "generate":      "generate",
            "rewrite_query": "rewrite_query",
            "fallback":      "fallback",
        }
    )

    return g.compile()