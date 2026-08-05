from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from src.config import Config
from src.retriever import get_retriever

def get_agent_chain():
    """
    Constructs the core conversational RAG chain.
    It combines the advanced retriever with an LLM for answering questions
    based solely on the provided context.
    """
    llm = ChatOpenAI(
        model_name=Config.MODEL_NAME, 
        temperature=Config.TEMPERATURE
    )
    
    # Advanced Prompt Template for Research Paper Assistance
    system_prompt = (
        "You are an advanced Research Paper Assistant. You are given a user question and "
        "some extracted context from research papers. Use the provided context to answer the question. "
        "If you don't know the answer or the context doesn't contain the information, "
        "just say that you don't know, don't try to make up an answer. "
        "Keep your answer concise, academic, and well-structured. "
        "Always cite the source documents if possible.\n\n"
        "Context:\n{context}"
    )
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input}"),
    ])
    
    # Document chain handles the passing of documents into the prompt
    question_answer_chain = create_stuff_documents_chain(llm, prompt)
    
    # Retrieval chain ties the retriever to the document chain
    retriever = get_retriever()
    rag_chain = create_retrieval_chain(retriever, question_answer_chain)
    
    return rag_chain

def run_query(query: str):
    """Executes a single query against the RAG agent."""
    rag_chain = get_agent_chain()
    print(f"Executing query: '{query}'")
    response = rag_chain.invoke({"input": query})
    return response['answer'], response['context']
