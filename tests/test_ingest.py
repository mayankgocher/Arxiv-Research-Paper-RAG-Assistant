import pytest
from unittest.mock import patch, MagicMock
from langchain_core.documents import Document
from src.ingest import chunk_documents, load_documents
from src.config import Config

def test_chunk_documents():
    # Create a dummy document larger than typical chunk size
    large_text = "A" * 2500
    doc = Document(page_content=large_text, metadata={"source": "test.txt"})
    
    chunks = chunk_documents([doc])
    
    # Assertions
    assert len(chunks) > 1, "Document should be split into multiple chunks"
    
    # Each chunk should not exceed the chunk size significantly (accounting for chunk logic)
    for chunk in chunks:
        assert len(chunk.page_content) <= Config.CHUNK_SIZE + 50

@patch('src.ingest.glob.glob')
@patch('src.ingest.PyPDFLoader')
@patch('src.ingest.os.path.exists')
def test_load_documents_error_handling(mock_exists, mock_pdf_loader, mock_glob):
    mock_exists.return_value = True
    # Return one fake PDF and no TXTs
    mock_glob.side_effect = [["data/fake.pdf"], []]
    
    # Make loader raise an exception to simulate a corrupt file
    mock_loader_instance = MagicMock()
    mock_loader_instance.load.side_effect = Exception("Corrupt PDF File")
    mock_pdf_loader.return_value = mock_loader_instance
    
    documents = load_documents("data")
    
    # It should catch the exception, not crash, and return 0 documents
    assert len(documents) == 0
    mock_pdf_loader.assert_called_once_with("data/fake.pdf")
