

from typing import TypedDict, List, Optional
from langchain_core.documents import Document
from langchain_core.messages import BaseMessage

class GraphState(TypedDict):
    question: str
    documents: List[Document]
    generation: str
    relevance_check: Optional[str]
    retry_count: int
    messages: List[BaseMessage]      # ← historique de la conversation