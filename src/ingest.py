import os
import glob
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from src.config import Config

def load_documents(data_dir: str = "data"):
    """Loads all PDF and TXT files from the specified directory."""
    documents = []
    
    if not os.path.exists(data_dir):
        print(f"Data directory '{data_dir}' not found. Creating it...")
        os.makedirs(data_dir)
        return documents

    # Load PDFs
    for file in glob.glob(os.path.join(data_dir, "*.pdf")):
        print(f"Loading {file}...")
        try:
            loader = PyPDFLoader(file)
            documents.extend(loader.load())
        except Exception as e:
            print(f"Warning: Failed to load {file}. Error: {e}")
        
    # Load TXTs
    for file in glob.glob(os.path.join(data_dir, "*.txt")):
        print(f"Loading {file}...")
        try:
            loader = TextLoader(file, encoding="utf-8")
            documents.extend(loader.load())
        except Exception as e:
            print(f"Warning: Failed to load {file}. Error: {e}")
        
    return documents

def chunk_documents(documents):
    """Splits documents into smaller chunks."""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=Config.CHUNK_SIZE,
        chunk_overlap=Config.CHUNK_OVERLAP,
    )
    chunks = text_splitter.split_documents(documents)
    return chunks

def ingest_data():
    """Main function to load, split, and ingest documents into ChromaDB."""
    print("Starting data ingestion process...")
    
    if not Config.OPENAI_API_KEY:
        raise ValueError("OPENAI_API_KEY is not set. Cannot create embeddings.")
        
    documents = load_documents()
    if not documents:
        print("No documents found in the data directory. Exiting ingestion.")
        return

    print(f"Loaded {len(documents)} documents. Splitting...")
    chunks = chunk_documents(documents)
    print(f"Created {len(chunks)} chunks.")
    
    print("Initializing embeddings and Vector Store...")
    embeddings = OpenAIEmbeddings()
    
    # Create or update vector store
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=Config.CHROMA_PERSIST_DIRECTORY
    )
    vectorstore.persist()
    print(f"Successfully ingested data into ChromaDB at {Config.CHROMA_PERSIST_DIRECTORY}")

if __name__ == "__main__":
    ingest_data()
