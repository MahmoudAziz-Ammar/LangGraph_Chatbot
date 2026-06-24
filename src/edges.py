from src.state import GraphState

def route_after_grading(state: GraphState) -> str:
    """
    Décide le prochain nœud après grade_documents :
    - docs pertinents        → generate
    - pas pertinents + retry < 2 → rewrite_query
    - pas pertinents + retry >= 2 → fallback
    """
    if state["relevance_check"] == "yes":
        return "generate"
    elif state.get("retry_count", 0) >= 2:
        return "fallback"
    else:
        return "rewrite_query"