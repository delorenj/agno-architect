from typing import Literal, Sequence
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, ToolMessage
from langchain.tools.retriever import create_retriever_tool
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain import hub
from pydantic import BaseModel, Field
from .interfaces import LanguageModelProvider

class AgentState(TypedDict):
    """State definition for the agent graph."""
    messages: Annotated[Sequence[BaseMessage], add_messages]

class Grade(BaseModel):
    """Binary score for relevance check."""
    binary_score: str = Field(description="Relevance score 'yes' or 'no'")

def create_retriever_tool_wrapper(retriever):
    """Create a retriever tool for the agent."""
    return create_retriever_tool(
        retriever,
        "retrieve_blog_posts",
        "Search and return information about blog posts.",
    )

def agent_node(state: AgentState, model_provider: LanguageModelProvider, tools: list):
    """Node that invokes the agent model."""
    model = model_provider.get_agent_model().bind_tools(tools)
    response = model.invoke(state["messages"])
    return {"messages": [response]}

def grade_documents_node(state: AgentState, model_provider: LanguageModelProvider):
    """Node that grades document relevance."""
    model = model_provider.get_grader_model().with_structured_output(Grade)
    prompt = PromptTemplate(
        template="""You are a grader assessing relevance of a retrieved document to a user question. 
        Here is the retrieved document: \n\n {context} \n\n
        Here is the user question: {question} \n
        If the document contains keyword(s) or semantic meaning related to the user question, grade it as relevant. 
        Give a binary score 'yes' or 'no'.""",
        input_variables=["context", "question"],
    )
    chain = prompt | model

    messages = state["messages"]
    question = messages[0].content if messages else ""
    docs = next(
        (msg.content for msg in reversed(messages) 
        if isinstance(msg, ToolMessage) and msg.content
    ), "")

    result = chain.invoke({"question": question, "context": docs})
    return "generate" if result.binary_score.lower() == "yes" else "rewrite"

def rewrite_node(state: AgentState, model_provider: LanguageModelProvider):
    """Node that rewrites the query."""
    model = model_provider.get_grader_model()
    question = state["messages"][0].content if state["messages"] else ""
    
    msg_content = f"""Look at the input and reason about the underlying semantic intent.
    Here is the initial question: \n ------- \n {question} \n ------- \n
    Formulate an improved question based on the original intent:"""
    
    response = model.invoke([HumanMessage(content=msg_content)])
    return {
        "messages": [
            AIMessage(content=f"Query rewritten to: {response.content}"),
            HumanMessage(content=response.content)
        ]
    }

def generate_node(state: AgentState, model_provider: LanguageModelProvider):
    """Node that generates the final answer."""
    model = model_provider.get_generator_model()
    prompt = hub.pull("rlm/rag-prompt")
    chain = prompt | model | StrOutputParser()

    question = state["messages"][0].content if state["messages"] else ""
    docs = next(
        (msg.content for msg in reversed(state["messages"]) 
         if isinstance(msg, ToolMessage) and msg.content),
        ""
    )

    response = chain.invoke({"question": question, "context": docs})
    return {"messages": [AIMessage(content=response)]}