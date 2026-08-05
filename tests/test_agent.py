import pytest
from unittest.mock import patch, MagicMock
from src.agent import get_agent_chain
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
