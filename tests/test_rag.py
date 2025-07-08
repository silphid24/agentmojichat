"""RAG system tests"""

import pytest
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

from app.rag.document_processor import DocumentProcessor, ProcessedDocument
from app.rag.embeddings import LocalEmbeddings
from app.rag.retriever import VectorRetriever
from app.rag.pipeline import RAGPipeline, RAGQuery, RAGResponse


@pytest.fixture
def sample_text():
    """Sample text for testing"""
    return """
    MOJI is an intelligent AI assistant designed to help users with various tasks.
    It uses advanced natural language processing to understand queries.
    The system can retrieve information from a knowledge base.
    """


@pytest.fixture
def temp_file(sample_text):
    """Create a temporary text file"""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
        f.write(sample_text)
        temp_path = f.name

    yield temp_path

    # Cleanup
    Path(temp_path).unlink()


@pytest.mark.asyncio
async def test_document_processor(temp_file):
    """Test document processing"""
    processor = DocumentProcessor(chunk_size=100, chunk_overlap=20)

    # Process file
    doc = await processor.process_file(temp_file)

    assert isinstance(doc, ProcessedDocument)
    assert doc.filename.endswith(".txt")
    assert len(doc.content) > 0
    assert len(doc.chunks) > 0
    assert doc.metadata["file_type"] == ".txt"


@pytest.mark.asyncio
async def test_document_processor_text(sample_text):
    """Test processing raw text"""
    processor = DocumentProcessor()

    # Process text
    doc = await processor.process_text(sample_text, {"source": "test"})

    assert isinstance(doc, ProcessedDocument)
    assert doc.content == sample_text
    assert doc.metadata["source"] == "test"
    assert len(doc.chunks) > 0


@pytest.mark.asyncio
async def test_embeddings():
    """Test embeddings generation"""
    # Test local embeddings (mock)
    embeddings = LocalEmbeddings()

    # Single embedding
    embedding = await embeddings.embed_query("test query")
    assert isinstance(embedding, list)
    assert len(embedding) == 384  # Default dimension

    # Multiple embeddings
    texts = ["text1", "text2", "text3"]
    embeddings_list = await embeddings.embed_documents(texts)
    assert len(embeddings_list) == 3
    assert all(len(e) == 384 for e in embeddings_list)


@pytest.mark.asyncio
async def test_vector_retriever():
    """Test vector retriever"""
    from app.rag.embeddings import get_embeddings

    embeddings = get_embeddings(use_local=True)
    retriever = VectorRetriever(embeddings=embeddings)

    # Create test documents
    from langchain.schema import Document

    docs = [
        Document(page_content="MOJI is an AI assistant", metadata={"source": "doc1"}),
        Document(
            page_content="The system uses machine learning", metadata={"source": "doc2"}
        ),
    ]

    # Create index
    await retriever.create_index(docs)

    # Search
    results = await retriever.search("AI assistant", k=1)

    assert len(results) > 0
    assert isinstance(results[0][0], Document)
    assert isinstance(results[0][1], float)  # Score


@pytest.mark.asyncio
async def test_rag_pipeline(sample_text):
    """Test complete RAG pipeline"""
    pipeline = RAGPipeline(use_local_embeddings=True)

    # Add text
    result = await pipeline.add_text(sample_text)
    assert result["success"] is True
    assert result["chunks"] > 0

    # Mock LLM response
    with patch("app.llm.router.llm_router.generate") as mock_generate:
        mock_response = Mock()
        mock_response.content = "Based on the context, MOJI is an AI assistant."
        mock_generate.return_value = mock_response

        # Query
        rag_query = RAGQuery(query="What is MOJI?", k=2)
        response = await pipeline.query(rag_query)

        assert isinstance(response, RAGResponse)
        assert len(response.answer) > 0
        assert response.query == "What is MOJI?"


@pytest.mark.asyncio
async def test_rag_query_no_results():
    """Test RAG query with no results"""
    pipeline = RAGPipeline(use_local_embeddings=True)

    # Query empty index
    rag_query = RAGQuery(query="Random unrelated query")
    response = await pipeline.query(rag_query)

    assert isinstance(response, RAGResponse)
    assert "couldn't find" in response.answer.lower()
    assert response.total_sources == 0


def test_rag_stats():
    """Test RAG statistics"""
    pipeline = RAGPipeline(use_local_embeddings=True)

    stats = pipeline.get_index_stats()

    assert isinstance(stats, dict)
    assert "status" in stats
