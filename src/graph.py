from langgraph.graph import StateGraph, END
from src.state import GraphState
from src.nodes import retrieve, generate, grade_documents, rewrite_query, fallback
from src.edges import route_after_grading

def build_graph():
    g = StateGraph(GraphState)

    g.add_node("retrieve", retrieve)
    g.add_node("generate", generate)

    g.set_entry_point("retrieve")
    g.add_edge("retrieve", "generate")
    g.add_edge("generate", END)

    return g.compile()