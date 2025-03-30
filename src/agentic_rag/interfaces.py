from abc import ABC, abstractmethod
from typing import List, Optional, Protocol, runtime_checkable
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseLanguageModel
from langchain_core.vectorstores import VectorStore

@runtime_checkable
class DocumentLoader(Protocol):
    """Protocol for document loaders that can load from various sources."""
    @abstractmethod
    def load(self) -> List[Document]:
        """Load documents from the source."""
        ...

@runtime_checkable
class TextSplitter(Protocol):
    """Protocol for text splitters that can split documents into chunks."""
    @abstractmethod
    def split_documents(self, documents: List[Document]) -> List[Document]:
        """Split documents into chunks."""
        ...

@runtime_checkable
class EmbeddingModel(Protocol):
    """Protocol for embedding models that can embed text."""
    @abstractmethod
    def get_embeddings(self) -> Embeddings:
        """Get the embedding model instance."""
        ...

@runtime_checkable
class VectorStoreProvider(Protocol):
    """Protocol for vector store providers that can create and manage vector stores."""
    @abstractmethod
    def create_store(self, documents: List[Document], collection_name: str) -> VectorStore:
        """Create a vector store from documents."""
        ...

@runtime_checkable
class LanguageModelProvider(Protocol):
    """Protocol for language model providers that can create different types of models."""
    @abstractmethod
    def get_agent_model(self) -> BaseLanguageModel:
        """Get a language model configured for agent tasks."""
        ...

    @abstractmethod
    def get_grader_model(self) -> BaseLanguageModel:
        """Get a language model configured for grading tasks."""
        ...

    @abstractmethod
    def get_generator_model(self) -> BaseLanguageModel:
        """Get a language model configured for generation tasks."""
        ...