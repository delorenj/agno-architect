import os
from langchain_core.messages import HumanMessage
from .agentic_rag.config import RAGConfig
from .agentic_rag.document_processing import create_document_loader, create_text_splitter
from .agentic_rag.embeddings import create_embedding_model
from .agentic_rag.vector_store import create_vector_store
from .agentic_rag.llm import create_language_model_provider
from .agentic_rag.graph_builder import create_graph_builder

def run_rag_workflow(config: RAGConfig = None):
    """Run the RAG workflow with optional configuration."""
    # Use default config if none provided
    config = config or RAGConfig.default()
    
    # Initialize components
    document_loader = create_document_loader(config.document_loader)
    text_splitter = create_text_splitter(config.text_splitter)
    embedding_model = create_embedding_model(config.embedding)
    vector_store_provider = create_vector_store(embedding_model)
    model_provider = create_language_model_provider(config.language_model)

    # Process documents
    print("Loading documents...")
    docs = document_loader.load()
    print(f"Loaded {len(docs)} documents")
    
    print("Splitting documents...")
    splits = text_splitter.split_documents(docs)
    print(f"Created {len(splits)} splits")

    # Create vector store
    print("Creating vector store...")
    vectorstore = vector_store_provider.create_store(
        splits,
        config.vector_store.collection_name
    )
    retriever = vectorstore.as_retriever()

    # Build and run graph
    print("Building workflow graph...")
    graph_builder = create_graph_builder(model_provider, retriever)
    graph = graph_builder.build()

    # Example query
    inputs = {
        "messages": [
            HumanMessage(content="What does Lilian Weng say about the types of agent memory?"),
        ]
    }

    print("\nRunning workflow...")
    for output in graph.stream(inputs, {"recursion_limit": 5}):
        for key, value in output.items():
            print(f"\nOutput from node '{key}':")
            print("---")
            if isinstance(value, dict) and 'messages' in value:
                for msg in value['messages']:
                    print(f"{type(msg).__name__}: {msg.content}")
            else:
                print(value)
        print("\n" + "-"*20)

if __name__ == "__main__":
    # Check for API key
    if not os.getenv("OPENAI_API_KEY"):
        print("Error: OPENAI_API_KEY environment variable not set")
    else:
        run_rag_workflow()