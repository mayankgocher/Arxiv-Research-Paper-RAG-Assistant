import pytest
from unittest.mock import patch, MagicMock
from src.agent import get_agent_chain, run_query
from src.config import Config

def test_config_defaults():
    assert Config.CHUNK_SIZE > 0
    assert Config.MODEL_NAME is not None

@patch('src.agent.ChatOpenAI')
@patch('src.agent.get_retriever')
def test_get_agent_chain(mock_get_retriever, mock_chat_openai):
    # Mock the LLM and Retriever
    mock_llm_instance = MagicMock()
    mock_chat_openai.return_value = mock_llm_instance
    
    mock_retriever_instance = MagicMock()
    mock_get_retriever.return_value = mock_retriever_instance
    
    # Execute the function
    chain = get_agent_chain()
    
    # Assertions
    assert chain is not None
    mock_chat_openai.assert_called_once()
    mock_get_retriever.assert_called_once()

@patch('src.agent.get_agent_chain')
def test_run_query_error_handling(mock_get_agent_chain):
    # Mock the chain to raise an exception
    mock_chain = MagicMock()
    mock_chain.invoke.side_effect = Exception("API Limit Reached")
    mock_get_agent_chain.return_value = mock_chain
    
    answer, context = run_query("test query")
    
    assert "An error occurred while generating the answer: API Limit Reached" in answer
    assert context == []
