import os
from langgraph.graph import START, StateGraph, MessagesState
from langgraph.prebuilt import tools_condition
from langgraph.prebuilt import ToolNode
from langchain_community.document_loaders import WikipediaLoader
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint, HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.tools import tool
from dotenv import load_dotenv

load_dotenv() 
@tool
def multiply(a: int, b: int) -> int:
    """
    Multiplies two integers.
    Args:
        a (int): first integer
        b (int): second integer

    Returns:
        int: product of a and b
    """
    return a * b

@tool
def add(a: int, b: int) -> int:
    """
    Adds two integers.
    Args:
        a (int): first integer
        b (int): second integer

    Returns:
        int: sum of a and b
    """
    return a + b

@tool
def subtract(a: int, b: int) -> int:
    """
    Subtracts two integers.
    Args:
        a (int): first integer
        b (int): second integer

    Returns:
        int: difference of a and b
    """
    return a - b

@tool
def divide(a: int, b: int) -> int:
    """
    Divides two integers.
    Args:
        a (int): first integer
        b (int): second integer

    Returns:
        int: quotient of a and b
    """
    if b == 0:
        raise ZeroDivisionError("Cannot divide by zero")
    return a / b

@tool
def modulus(a: int, b: int) -> int:
    """
    Computes the modulus of two integers.
    Args:
        a (int): first integer
        b (int): second integer

    Returns:
        int: remainder of a divided by b
    """
    return a % b

@tool
def power(a: int, b: int) -> int:
    """
    Raises a to the power of b.
    Args:
        a (int): base
        b (int): exponent

    Returns:
        int: result of a raised to the power of b
    """
    return a ** b

@tool
def wikipedia_search(query: str) -> str:
    """
    Searches Wikipedia for the given query and returns the first results.
    Args:
        query (str): search query

    Returns:
        str: first results from Wikipedia search
    """
    search_docs = WikipediaLoader(query=query, load_max_docs=2).load()
    formatted_search_docs = "\n\n---\n\n".join(
        [
            f'<Document source="{doc.metadata["source"]}" page="{doc.metadata.get("page", "")}"/>\n{doc.page_content}\n</Document>'
            for doc in search_docs
        ])
    return {"wiki_results": formatted_search_docs}

# load the system prompt from the file
with open("system_prompt.txt", "r", encoding="utf-8") as f:
    system_prompt = f.read()

sys_msg = SystemMessage(content=system_prompt)

tools = [
    multiply,
    add,
    subtract,
    divide,
    modulus,
    power,
    wikipedia_search,
]

# Build graph function
def build_graph(provider: str="google"):
    """Build the graph"""
    # Load environment variables from .env file
    if provider == "google":
        # Google Gemini
        llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0)
    elif provider == "huggingface":

        llm = ChatHuggingFace(
            llm=HuggingFaceEndpoint(
                repo_id="meta-llama/Llama-3.1-8B-Instruct",
                provider="sambanova",
                huggingfacehub_api_token=os.getenv("HUGGINGFACEHUB_API_TOKEN")
            ),
        )

    llm_with_tools = llm.bind_tools(tools)
    # Node
    def assistant(state: MessagesState):
        """Assistant node"""
        return {"messages": [llm_with_tools.invoke(state["messages"])]}

    builder = StateGraph(MessagesState)
    builder.add_node("assistant", assistant)
    builder.add_node("tools", ToolNode(tools))
    builder.add_edge(START, "assistant")
    builder.add_conditional_edges(
        "assistant",
        tools_condition,
    )
    builder.add_edge("tools", "assistant")

    # Compile graph
    return builder.compile()

if __name__ == "__main__":
    question = "When was a picture of St. Thomas Aquinas first added to the Wikipedia page on the Principle of double effect?"
    # Build the graph
    graph = build_graph(provider="huggingface")
    # Run the graph
    messages = [HumanMessage(content=question)]
    messages = graph.invoke({"messages": messages})
    for m in messages["messages"]:
        m.pretty_print()