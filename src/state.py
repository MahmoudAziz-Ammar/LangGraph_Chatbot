from typing import TypedDict, List, Optional
from langchain_core.documents import Document

class GraphState(TypedDict):
    question: str                   # question de l'utilisateur
    documents: List[Document]       # chunks récupérés depuis FAISS
    generation: str                 # réponse générée par le LLM
    relevance_check: Optional[str]  # "yes" ou "no" après grading
    retry_count: int                # nombre de tentatives de rewrite