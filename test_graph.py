from src.vectorstore import build_vectorstore
from src.graph import build_graph

# 1. Indexe un PDF d'abord
build_vectorstore()

# 2. Lance le graphe
graph = build_graph()
result = graph.invoke({
    "question": "est ce que c un bon profil de data scientist ?",
    "documents": [],
    "generation": "",
    "relevance_check": None,
    "retry_count": 0
})

print("\n✅ Réponse :", result["generation"])