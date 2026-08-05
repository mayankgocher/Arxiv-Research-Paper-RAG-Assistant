import pytest
from langchain_core.documents import Document
from src.ingest import chunk_documents
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
