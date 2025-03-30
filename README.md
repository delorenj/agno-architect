# AGNO Architect

An agentic Retrieval Augmented Generation (RAG) system built with LangChain and LangGraph.

## Project Overview

AGNO Architect is a flexible framework for building agentic RAG workflows. The system implements an advanced retrieval and generation workflow with self-reflexive capabilities such as query rewriting and document relevance grading.

### Key Features

- **Modular Architecture**: Clean interfaces for each component, making it easy to swap implementations
- **Agentic Workflow**: Uses LangGraph to orchestrate complex decision flows
- **Self-improvement Loops**: Query rewriting and document relevance assessment
- **Configurable Components**: Flexible configuration system for all components

## Architecture

The system is built around several key components:

1. **Document Processing**: Loads documents from the web or other sources and splits them into appropriate chunks
2. **Embedding**: Creates vector embeddings for document chunks
3. **Vector Store**: Stores and retrieves document chunks based on semantic similarity
4. **Language Models**: Provides various LLM capabilities for different parts of the workflow
5. **Graph Workflow**: Orchestrates the flow of information through the system

### Workflow Graph

The RAG workflow is implemented as a directed graph with the following nodes:

- **Agent**: Decides whether to retrieve information or answer directly
- **Retrieve**: Executes retrieval tools to find relevant information
- **Grade Documents**: Evaluates whether retrieved documents are relevant to the query
- **Rewrite**: Reformulates queries that didn't yield relevant results
- **Generate**: Creates the final response using retrieved information

## Getting Started

### Prerequisites

- Python 3.12+
- OpenAI API key

### Installation

1. Clone the repository:

   ```bash
   git clone [repository-url]
   cd agno-architect
   ```

2. Set up a virtual environment:

   ```bash
   python -m venv .venv
   source .venv/bin/activate
   ```

3. Install dependencies:

   ```bash
   pip install -e .
   ```

4. Set up your OpenAI API key:

   ```bash
   export OPENAI_API_KEY="your-api-key"
   ```

### Running the System

Execute the main module:

```bash
python -m src.main
```

By default, this will use the example configuration that retrieves information from Lilian Weng's blog posts.

## Development Guide

### Component Architecture

The system follows a clean architecture pattern with clear interfaces for each component:

- `DocumentLoader`: Loads documents from various sources
- `TextSplitter`: Breaks documents into manageable chunks
- `EmbeddingModel`: Creates vector representations of text
- `VectorStoreProvider`: Manages vector databases for semantic retrieval
- `LanguageModelProvider`: Provides different LLM interfaces for various tasks

### Configuration System

The configuration is hierarchical, with each component having its own configuration class:

- `RAGConfig`: Top-level configuration
  - `DocumentLoaderConfig`: URL sources, etc.
  - `TextSplitterConfig`: Chunking parameters
  - `EmbeddingConfig`: Embedding model selection
  - `VectorStoreConfig`: Vector DB settings
  - `LanguageModelConfig`: LLM selection and parameters

### Adding a New Component

To implement a new component:

1. Define its interface in `interfaces.py` if needed
2. Create a new implementation file
3. Add factory functions to instantiate your component
4. Update the relevant configuration classes

## Future Development

Potential next steps for the project:

1. **Additional Document Sources**: Support for more document types (PDFs, databases, APIs)
2. **Improved Evaluation**: More sophisticated document relevance assessment
3. **Memory System**: Persistent memory across queries for better context
4. **Web Interface**: Simple UI for interacting with the system
5. **Tool Integration**: Add more tools beyond retrieval (calculation, web search, etc.)
6. **Custom Embeddings**: Support for custom embedding models
7. **Parallelized Processing**: Concurrent document processing and embedding
8. **Caching Layer**: Improve performance with smart caching strategies

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

[Specify the license here]

