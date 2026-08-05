import streamlit as st
import os
from src.agent import run_query
from src.config import Config

st.set_page_config(page_title="Arxiv Assistant", page_icon="📚", layout="centered")

st.title("📚 Arxiv Research Paper Assistant")
st.markdown("Ask questions about the ingested research papers, and I will use an advanced Multi-Query Retrieval strategy to find the best answers from the documents.")

if not Config.OPENAI_API_KEY:
    st.error("⚠️ OPENAI_API_KEY is not set. Please add it to your .env file or environment variables.")
    st.stop()

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# React to user input
if prompt := st.chat_input("What is the Transformer architecture?"):
    # Display user message in chat message container
    st.chat_message("user").markdown(prompt)
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.spinner("Searching and generating answer..."):
        try:
            answer, context = run_query(prompt)
            
            # Format sources
            sources = list(set([doc.metadata.get("source", "Unknown") for doc in context]))
            source_text = "\n\n**Sources:**\n"
            for i, source in enumerate(sources):
                source_text += f"- [{i+1}] {os.path.basename(source)}\n"
            
            full_response = answer + source_text
            
            # Display assistant response in chat message container
            with st.chat_message("assistant"):
                st.markdown(full_response)
                
            # Add assistant response to chat history
            st.session_state.messages.append({"role": "assistant", "content": full_response})
            
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
