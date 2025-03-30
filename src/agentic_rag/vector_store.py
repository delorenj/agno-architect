from langchain_community.vectorstores import Chroma
from langchain_core.vectorstores import VectorStore
from .interfaces import VectorStoreProvider
from .config import VectorStoreConfig

class ChromaVectorStore(VectorStoreProvider):
    """Concrete implementation of VectorStoreProvider for ChromaDB."""
    
    def __init__(self, embedding_model):
        self.embedding_model = embedding_model

    def create_store(self, documents: list, collection_name: str) -> VectorStore:
        """Create a Chroma vector store from documents."""
        return Chroma.from_documents(
            documents=documents,
            collection_name=collection_name,
            embedding=self.embedding_model.get_embeddings()
        )

def create_vector_store(embedding_model) -> VectorStoreProvider:
    """Factory function for creating vector stores."""
    return ChromaVectorStore(embedding_model)