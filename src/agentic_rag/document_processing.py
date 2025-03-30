from typing import List
from langchain_core.documents import Document
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from .interfaces import DocumentLoader, TextSplitter
from .config import DocumentLoaderConfig, TextSplitterConfig

class WebDocumentLoader(DocumentLoader):
    """Concrete implementation of DocumentLoader for web URLs."""
    
    def __init__(self, config: DocumentLoaderConfig):
        self.config = config
        self.loaders = [WebBaseLoader(url) for url in config.urls]

    def load(self) -> List[Document]:
        """Load documents from web URLs."""
        docs = []
        for loader in self.loaders:
            try:
                loaded = loader.load()
                docs.extend(loaded)
            except Exception as e:
                print(f"Error loading documents from {loader.web_path}: {e}")
        return docs

class RecursiveTiktokenSplitter(TextSplitter):
    """Concrete implementation of TextSplitter using tiktoken tokenizer."""
    
    def __init__(self, config: TextSplitterConfig):
        self.config = config
        self.splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
            chunk_size=config.chunk_size,
            chunk_overlap=config.chunk_overlap
        )

    def split_documents(self, documents: List[Document]) -> List[Document]:
        """Split documents using tiktoken tokenizer."""
        return self.splitter.split_documents(documents)

def create_document_loader(config: DocumentLoaderConfig) -> DocumentLoader:
    """Factory function for creating document loaders."""
    return WebDocumentLoader(config)

def create_text_splitter(config: TextSplitterConfig) -> TextSplitter:
    """Factory function for creating text splitters."""
    return RecursiveTiktokenSplitter(config)