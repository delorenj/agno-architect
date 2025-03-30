from langgraph.graph import END, StateGraph, START
from langgraph.prebuilt import ToolNode, tools_condition
from typing import Annotated, Sequence, TypedDict
from langgraph.graph.message import add_messages
from .graph_components import (
    AgentState,
    create_retriever_tool_wrapper,
    agent_node,
    grade_documents_node,
    rewrite_node,
    generate_node
)

class RAGGraphBuilder:
    """Builds and compiles the RAG workflow graph."""
    
    def __init__(self, model_provider, retriever):
        self.model_provider = model_provider
        self.retriever = retriever
        self.tools = [create_retriever_tool_wrapper(retriever)]

    def build(self):
        """Build and compile the RAG workflow graph."""
        workflow = StateGraph(AgentState)

        # Define nodes with partial application of dependencies
        workflow.add_node(
            "agent", 
            lambda state: agent_node(state, self.model_provider, self.tools)
        )
        workflow.add_node("retrieve", ToolNode(self.tools))
        workflow.add_node(
            "rewrite", 
            lambda state: rewrite_node(state, self.model_provider)
        )
        workflow.add_node(
            "generate", 
            lambda state: generate_node(state, self.model_provider)
        )

        # Define edges
        workflow.add_edge(START, "agent")
        workflow.add_conditional_edges(
            "agent",
            tools_condition,
            {"tools": "retrieve", END: END}
        )
        workflow.add_conditional_edges(
            "retrieve",
            lambda state: grade_documents_node(state, self.model_provider),
            {"generate": "generate", "rewrite": "rewrite"}
        )
        workflow.add_edge("generate", END)
        workflow.add_edge("rewrite", "agent")

        return workflow.compile()

def create_graph_builder(model_provider, retriever):
    """Factory function for creating graph builders."""
    return RAGGraphBuilder(model_provider, retriever)