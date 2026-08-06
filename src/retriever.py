from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.retrievers.document_compressors import CrossEncoderReranker
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from langchain.retrievers import ContextualCompressionRetriever
from src.config import Config
import logging
import os

# Configure logging for the multi-query retriever so we can see generated queries
logging.basicConfig(level=logging.INFO)
logging.getLogger("langchain.retrievers.multi_query").setLevel(logging.INFO)

def get_vectorstore():
    """Initializes and returns the Chroma vector store connection."""
    if not os.path.exists(Config.CHROMA_PERSIST_DIRECTORY) or not os.listdir(Config.CHROMA_PERSIST_DIRECTORY):
        raise FileNotFoundError(
            f"Vector store not found or empty at {Config.CHROMA_PERSIST_DIRECTORY}. "
            "Please run 'python src/ingest.py' to ingest documents first."
        )
        
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
    
    # The base retriever fetches a larger pool of candidates
    base_retriever = vectorstore.as_retriever(search_kwargs={"k": 10})
    
    # LLM used to generate alternative queries
    llm = ChatOpenAI(
        model_name=Config.MODEL_NAME, 
        temperature=Config.TEMPERATURE
    )
    
    # Initialize MultiQueryRetriever
    multi_query_retriever = MultiQueryRetriever.from_llm(
        retriever=base_retriever, 
        llm=llm
    )
    
    # Initialize the Cross-Encoder Reranker
    model = HuggingFaceCrossEncoder(model_name=Config.RERANKER_MODEL_NAME)
    compressor = CrossEncoderReranker(model=model, top_n=3)
    
    # Wrap the MultiQueryRetriever with ContextualCompressionRetriever
    compression_retriever = ContextualCompressionRetriever(
        base_compressor=compressor, 
        base_retriever=multi_query_retriever
    )
    
    return compression_retriever
