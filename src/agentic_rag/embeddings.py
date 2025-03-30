from langchain_openai import OpenAIEmbeddings
from .interfaces import EmbeddingModel
from .config import EmbeddingConfig

class OpenAIEmbeddingModel(EmbeddingModel):
    """Concrete implementation of EmbeddingModel using OpenAI embeddings."""
    
    def __init__(self, config: EmbeddingConfig):
        self.config = config
        self._embeddings = OpenAIEmbeddings(model=config.model_name)

    def get_embeddings(self) -> OpenAIEmbeddings:
        """Get the OpenAI embeddings instance."""
        return self._embeddings

def create_embedding_model(config: EmbeddingConfig) -> EmbeddingModel:
    """Factory function for creating embedding models."""
    return OpenAIEmbeddingModel(config)