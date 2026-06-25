from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from src.config import DATA_DIR, VECTORSTORE_PATH, CHUNK_SIZE, CHUNK_OVERLAP, EMBEDDING_MODEL
import os

def get_embeddings():
    """Modèle d'embeddings local — 0€"""
    return HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

def build_vectorstore():
    """Charge les PDFs, découpe en chunks, crée l'index FAISS"""
    loader = DirectoryLoader(DATA_DIR, glob="**/*.pdf", loader_cls=PyPDFLoader)
    docs   = loader.load()
    print(f"✅ {len(docs)} pages chargées")

    chunks = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP
    ).split_documents(docs)
    print(f"✅ {len(chunks)} chunks créés")

    vs = FAISS.from_documents(chunks, get_embeddings())
    vs.save_local(VECTORSTORE_PATH)
    print(f"✅ Index FAISS sauvegardé")
    return vs

_vectorstore_cache = None

def load_vectorstore():
    global _vectorstore_cache
    if _vectorstore_cache is not None:
        return _vectorstore_cache
    if not os.path.exists(VECTORSTORE_PATH):
        raise FileNotFoundError("Lance d'abord build_vectorstore()")
    _vectorstore_cache = FAISS.load_local(
        VECTORSTORE_PATH,
        get_embeddings(),
        allow_dangerous_deserialization=True
    )
    return _vectorstore_cache