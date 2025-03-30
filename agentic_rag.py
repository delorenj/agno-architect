# Extracted from docs/session/plan.md

# Retriever Section
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.tools.retriever import create_retriever_tool

urls = [
    "https://lilianweng.github.io/posts/2023-06-23-agent/",
    "https://lilianweng.github.io/posts/2023-03-15-prompt-engineering/",
    "https://lilianweng.github.io/posts/2023-10-25-adv-attack-llm/",
]

docs = [WebBaseLoader(url).load() for url in urls]
docs_list = [item for sublist in docs for item in sublist]

text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
    chunk_size=100, chunk_overlap=50
)
doc_splits = text_splitter.split_documents(docs_list)

# Add to vectorDB
# Ensure OPENAI_API_KEY is set in the environment before running
vectorstore = Chroma.from_documents(
    documents=doc_splits,
    collection_name="rag-chroma",
    embedding=OpenAIEmbeddings(),
)
retriever = vectorstore.as_retriever()

retriever_tool = create_retriever_tool(
    retriever,
    "retrieve_blog_posts",
    "Search and return information about Lilian Weng blog posts on LLM agents, prompt engineering, and adversarial attacks on LLMs.",
)

tools = [retriever_tool]

# Agent State Section
from typing import Annotated, Sequence, Literal
from typing_extensions import TypedDict
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, ToolMessage
from langgraph.graph.message import add_messages
from pydantic import BaseModel, Field

class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]

# Nodes and Edges Section
from langchain import hub
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import tools_condition
from langgraph.graph import END, StateGraph, START
from langgraph.prebuilt import ToolNode
import pprint
# from IPython.display import Image, display # Commented out for non-notebook environments
import os

# Ensure OpenAI API key is set (replace with your actual key handling method if needed)
# Example: os.environ["OPENAI_API_KEY"] = "YOUR_API_KEY"
# Make sure to handle API keys securely, e.g., via environment variables or a config file.
if not os.getenv("OPENAI_API_KEY"):
    print("Warning: OPENAI_API_KEY environment variable not set.")
    # You might want to add logic here to load the key from a .env file or prompt the user
    # For now, the script might fail if the key isn't set externally.

### Edges

def grade_documents(state) -> Literal["generate", "rewrite"]:
    """
    Determines whether the retrieved documents are relevant to the question.

    Args:
        state (messages): The current state

    Returns:
        str: A decision for whether the documents are relevant or not
    """

    print("---CHECK RELEVANCE---")

    # Data model
    class grade(BaseModel):
        """Binary score for relevance check."""

        binary_score: str = Field(description="Relevance score 'yes' or 'no'")

    # LLM
    # Make sure OPENAI_API_KEY is available in the environment
    try:
        model = ChatOpenAI(temperature=0, model="gpt-4o", streaming=True)
    except Exception as e:
        print(f"Error initializing ChatOpenAI for grading: {e}")
        print("Ensure OPENAI_API_KEY is set correctly.")
        return "rewrite" # Fallback

    # LLM with tool and validation
    llm_with_tool = model.with_structured_output(grade)

    # Prompt
    prompt = PromptTemplate(
        template="""You are a grader assessing relevance of a retrieved document to a user question. \n
        Here is the retrieved document: \n\n {context} \n\n
        Here is the user question: {question} \n
        If the document contains keyword(s) or semantic meaning related to the user question, grade it as relevant. \n
        Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question.""",
        input_variables=["context", "question"],
    )

    # Chain
    chain = prompt | llm_with_tool

    messages = state["messages"]
    # Ensure there are messages and the last one is a ToolMessage
    if not messages:
        print("Error: No messages in state for grading.")
        return "rewrite" # Or handle error appropriately

    # Find the most recent ToolMessage with content
    docs = ""
    question = ""
    if messages:
        question = messages[0].content # Assumes first message is the user question
        for msg in reversed(messages):
            if isinstance(msg, ToolMessage) and msg.content:
                docs = msg.content
                break # Found the latest relevant docs

    if not question:
        print("Error: Could not determine the question from the state.")
        return "rewrite"
    if not docs:
        print("Error: No relevant ToolMessage content found for grading.")
        # This might happen if the agent decided not to retrieve or retrieval failed.
        # Depending on the desired flow, you might want to end or try rewriting.
        return "rewrite" # Decide how to handle this case

    try:
        scored_result = chain.invoke({"question": question, "context": docs})
        score = scored_result.binary_score

        if score and score.lower() == "yes":
            print("---DECISION: DOCS RELEVANT---")
            return "generate"
        else:
            print(f"---DECISION: DOCS NOT RELEVANT (Score: {score})---")
            return "rewrite"
    except Exception as e:
        print(f"Error during relevance grading: {e}")
        return "rewrite" # Fallback decision on error


### Nodes

def agent(state):
    """
    Invokes the agent model to generate a response based on the current state. Given
    the question, it will decide to retrieve using the retriever tool, or simply end.

    Args:
        state (messages): The current state

    Returns:
        dict: The updated state with the agent response appended to messages
    """
    print("---CALL AGENT---")
    messages = state["messages"]
    # Make sure OPENAI_API_KEY is available in the environment
    try:
        model = ChatOpenAI(temperature=0, streaming=True, model="gpt-4-turbo")
        model = model.bind_tools(tools)
        response = model.invoke(messages)
        # We return a list, because this will get added to the existing list
        return {"messages": [response]}
    except Exception as e:
        print(f"Error calling agent model: {e}")
        # Return an error message or handle appropriately
        return {"messages": [AIMessage(content=f"Error in agent node: {e}")]}


def rewrite(state):
    """
    Transform the query to produce a better question.

    Args:
        state (messages): The current state

    Returns:
        dict: The updated state with re-phrased question
    """

    print("---TRANSFORM QUERY---")
    messages = state["messages"]
    if not messages:
        print("Error: No messages in state for rewrite.")
        return {"messages": [HumanMessage(content="Error: Cannot rewrite without an initial question.")]}

    question = messages[0].content # Assumes first message is the user question

    msg_content = f""" \n
    Look at the input and try to reason about the underlying semantic intent / meaning. \n
    Here is the initial question:
    \n ------- \n
    {question}
    \n ------- \n
    Formulate an improved question based on the original intent, suitable for retrieval: """
    msg = [HumanMessage(content=msg_content)]

    # Grader/Rewriter LLM
    # Make sure OPENAI_API_KEY is available in the environment
    try:
        model = ChatOpenAI(temperature=0, model="gpt-4-0125-preview", streaming=True)
        response = model.invoke(msg)
        # Ensure the response is wrapped in a message type LangGraph expects (e.g., HumanMessage or AIMessage)
        # Using HumanMessage here to represent the *rewritten* user query for the next agent step
        rewritten_question_message = HumanMessage(content=response.content)
        print(f"--- Rewritten Question: {response.content} ---")
        # Replace the original question with the rewritten one for the next agent cycle
        # Or append it, depending on how state is managed. Let's append for clarity.
        # return {"messages": [rewritten_question_message]} # This replaces state, which might not be desired
        # Let's return an AIMessage indicating the rewrite happened, and the agent can use the latest HumanMessage
        return {"messages": [AIMessage(content=f"Query rewritten to: {response.content}"), rewritten_question_message]}

    except Exception as e:
        print(f"Error during query rewrite: {e}")
        return {"messages": [AIMessage(content=f"Error rewriting query: {e}")]}


def generate(state):
    """
    Generate answer based on retrieved documents.

    Args:
        state (messages): The current state

    Returns:
         dict: The updated state with generated answer
    """
    print("---GENERATE---")
    messages = state["messages"]
    if not messages:
        print("Error: No messages in state for generation.")
        return {"messages": [AIMessage(content="Error: Cannot generate answer without context.")]}

    question = messages[0].content # Assumes first message is the user question

    # Find the most recent ToolMessage content
    docs_content = ""
    for msg in reversed(messages):
        if isinstance(msg, ToolMessage) and msg.content:
            docs_content = msg.content
            break

    if not docs_content:
        print("Error: No document content found in ToolMessages for generation.")
        # Handle error: maybe return a message indicating failure or fallback
        return {"messages": [AIMessage(content="Could not find relevant documents to generate an answer.")]}


    # Prompt
    try:
        prompt = hub.pull("rlm/rag-prompt")
    except Exception as e:
        print(f"Error pulling RAG prompt: {e}")
        return {"messages": [AIMessage(content=f"Error loading RAG prompt: {e}")]}


    # LLM
    # Make sure OPENAI_API_KEY is available in the environment
    try:
        llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0, streaming=True)
    except Exception as e:
        print(f"Error initializing ChatOpenAI for generation: {e}")
        return {"messages": [AIMessage(content=f"Error setting up generation model: {e}")]}

    # Chain
    rag_chain = prompt | llm | StrOutputParser()

    # Run
    try:
        response = rag_chain.invoke({"context": docs_content, "question": question})
        # Wrap final response in AIMessage
        return {"messages": [AIMessage(content=response)]}
    except Exception as e:
        print(f"Error during generation: {e}")
        return {"messages": [AIMessage(content=f"Error generating answer: {e}")]}


# Print the RAG prompt for reference
print("*" * 20 + " Prompt[rlm/rag-prompt] " + "*" * 20)
try:
    rag_prompt_template = hub.pull("rlm/rag-prompt")
    print(rag_prompt_template.pretty_print())
except Exception as e:
    print(f"Could not pull or print rag-prompt: {e}")
print("*" * 50)


# Graph Definition
workflow = StateGraph(AgentState)

# Define the nodes
workflow.add_node("agent", agent)
retrieve_node = ToolNode(tools) # Use 'tools' which contains the retriever_tool
workflow.add_node("retrieve", retrieve_node)
workflow.add_node("rewrite", rewrite)
workflow.add_node("generate", generate)

# Define the edges
workflow.add_edge(START, "agent")

workflow.add_conditional_edges(
    "agent",
    tools_condition, # Decides if tools should be called
    {
        "tools": "retrieve", # If tools are needed, go to retrieve node
        END: END, # If no tools needed, end
    },
)

workflow.add_conditional_edges(
    "retrieve",
    grade_documents, # Check relevance after retrieval
    {
        "generate": "generate", # If relevant, generate answer
        "rewrite": "rewrite", # If not relevant, rewrite question
    }
)
workflow.add_edge("generate", END) # End after generation
workflow.add_edge("rewrite", "agent") # Go back to agent after rewriting

# Compile the graph
graph = workflow.compile()

# Display graph visualization (optional, requires extra dependencies)
# This part might fail if graphviz is not installed or not in PATH
try:
    # Ensure graphviz is installed (`pip install graphviz pygraphviz` or `conda install python-graphviz pygraphviz`)
    # and the graphviz binaries are in your system PATH
    png_data = graph.get_graph(xray=True).draw_mermaid_png()
    # In a non-notebook environment, save the image to a file:
    with open("agentic_rag_graph.png", "wb") as f:
        f.write(png_data)
    print("Graph visualization saved to agentic_rag_graph.png")
    # If in IPython/Jupyter: display(Image(png_data))
except ImportError:
    print("Could not import graphviz/pygraphviz. Skipping graph visualization.")
    print("Install with: pip install graphviz pygraphviz")
except Exception as e:
    print(f"Could not generate graph visualization: {e}")
    print("Ensure graphviz binaries are installed and in your system's PATH.")
    pass

# Run the graph
if __name__ == "__main__":
    print("\n" + "="*10 + " Running the Agentic RAG Graph " + "="*10)
    # Check for API key one last time before running
    if not os.getenv("OPENAI_API_KEY"):
        print("\nFATAL ERROR: OPENAI_API_KEY is not set. Cannot run the graph.")
        print("Please set the OPENAI_API_KEY environment variable.")
    else:
        inputs = {
            "messages": [
                HumanMessage(content="What does Lilian Weng say about the types of agent memory?"),
            ]
        }
        try:
            # Use stream to see intermediate steps
            for output in graph.stream(inputs, {"recursion_limit": 5}): # Added recursion limit
                for key, value in output.items():
                    print(f"\nOutput from node '{key}':")
                    print("---")
                    # Check if value is a dict and contains 'messages'
                    if isinstance(value, dict) and 'messages' in value:
                         pprint.pprint(value['messages'], indent=2, width=80, depth=None)
                    else:
                         pprint.pprint(value, indent=2, width=80, depth=None) # Print raw value if not standard message format
                print("\n" + "-"*20)
        except Exception as e:
            print(f"\nAn error occurred during graph execution: {e}")

        print("\n" + "="*10 + " Graph execution finished " + "="*10)