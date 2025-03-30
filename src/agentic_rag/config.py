from dataclasses import dataclass
from typing import List, Optional

@dataclass
class DocumentLoaderConfig:
    """Configuration for document loaders."""
    urls: List[str]
    # Add other loader-specific configs here

@dataclass
class TextSplitterConfig:
    """Configuration for text splitters."""
    chunk_size: int = 100
    chunk_overlap: int = 50
    # Add other splitter-specific configs here

@dataclass
class EmbeddingConfig:
    """Configuration for embedding models."""
    model_name: str = "text-embedding-ada-002"
    # Add other embedding-specific configs here

@dataclass
class VectorStoreConfig:
    """Configuration for vector stores."""
    collection_name: str = "rag-chroma"
    # Add other vector store-specific configs here

@dataclass
class LanguageModelConfig:
    """Configuration for language models."""
    agent_model: str = "gpt-4-turbo"
    grader_model: str = "gpt-4o"
    generator_model: str = "gpt-4o-mini"
    temperature: float = 0.0
    # Add other model-specific configs here

@dataclass
class RAGConfig:
    """Top-level configuration for the RAG system."""
    document_loader: DocumentLoaderConfig
    text_splitter: TextSplitterConfig
    embedding: EmbeddingConfig
    vector_store: VectorStoreConfig
    language_model: LanguageModelConfig

    @classmethod
    def default(cls) -> "RAGConfig":
        """Create a default configuration."""
        return cls(
            document_loader=DocumentLoaderConfig(
                urls=[
                    "https://lilianweng.github.io/posts/2023-06-23-agent/",
                    "https://lilianweng.github.io/posts/2023-03-15-prompt-engineering/",
                    "https://lilianweng.github.io/posts/2023-10-25-adv-attack-llm/",
                ]
            ),
            text_splitter=TextSplitterConfig(),
            embedding=EmbeddingConfig(),
            vector_store=VectorStoreConfig(),
            language_model=LanguageModelConfig(),
        )