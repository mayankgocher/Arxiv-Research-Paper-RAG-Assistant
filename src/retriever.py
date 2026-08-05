from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from src.config import Config
import logging

# Configure logging for the multi-query retriever so we can see generated queries
logging.basicConfig(level=logging.INFO)
logging.getLogger("langchain.retrievers.multi_query").setLevel(logging.INFO)

def get_vectorstore():
    """Initializes and returns the Chroma vector store connection."""
    embeddings = OpenAIEmbeddings()
    vectorstore = Chroma(
        persist_directory=Config.CHROMA_PERSIST_DIRECTORY,
        embedding_function=embeddings
    )
    return vectorstore

def get_retriever():
    """
    Returns an advanced MultiQueryRetriever.
    This uses an LLM to generate multiple perspective queries from the user's input,
    retrieves documents for each query, and returns the unique union of all retrieved documents.
    """
    vectorstore = get_vectorstore()
    
    # The base retriever
    base_retriever = vectorstore.as_retriever(search_kwargs={"k": 4})
    
    # LLM used to generate alternative queries
    llm = ChatOpenAI(
        model_name=Config.MODEL_NAME, 
        temperature=Config.TEMPERATURE
    )
    
    # Initialize MultiQueryRetriever
    retriever = MultiQueryRetriever.from_llm(
        retriever=base_retriever, 
        llm=llm
    )
    
    return retriever
