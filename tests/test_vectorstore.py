"""Tests for vector store implementations"""

import pytest
from unittest.mock import Mock, AsyncMock, patch
from langchain.schema import Document

from app.vectorstore.base import VectorStoreConfig, SearchResult
from app.vectorstore.chroma_store import ChromaVectorStore
from app.vectorstore.manager import VectorStoreManager, VectorStoreType


@pytest.fixture
def vector_config():
    """Test vector store configuration"""
    return VectorStoreConfig(
        collection_name="test_collection",
        persist_directory="test_data/chroma",
        distance_metric="cosine",
    )


@pytest.fixture
def sample_documents():
    """Sample documents for testing"""
    return [
        Document(
            page_content="This is a test document about AI",
            metadata={"source": "test1.txt", "page": 1},
        ),
        Document(
            page_content="Machine learning is a subset of AI",
            metadata={"source": "test2.txt", "page": 1},
        ),
        Document(
            page_content="Deep learning uses neural networks",
            metadata={"source": "test3.txt", "page": 2},
        ),
    ]


@pytest.fixture
def mock_embeddings():
    """Mock embeddings function"""
    mock = Mock()
    mock.embed_documents = Mock(return_value=[[0.1, 0.2, 0.3]] * 3)
    mock.embed_query = Mock(return_value=[0.1, 0.2, 0.3])
    return mock


class TestChromaVectorStore:
    """Test Chroma vector store implementation"""

    @pytest.mark.asyncio
    async def test_initialize(self, vector_config, mock_embeddings):
        """Test store initialization"""
        with patch("chromadb.Client") as mock_client:
            mock_collection = Mock()
            mock_client.return_value.get_or_create_collection.return_value = (
                mock_collection
            )

            store = ChromaVectorStore(vector_config, embeddings=mock_embeddings)
            await store.initialize()

            assert store.is_initialized
            assert store.client is not None
            assert store.collection is not None

            mock_client.return_value.get_or_create_collection.assert_called_once_with(
                name="test_collection",
                metadata={"distance_metric": "cosine", "index_params": {}},
            )

    @pytest.mark.asyncio
    async def test_add_documents(
        self, vector_config, sample_documents, mock_embeddings
    ):
        """Test adding documents"""
        with patch("chromadb.Client") as mock_client:
            mock_collection = Mock()
            mock_client.return_value.get_or_create_collection.return_value = (
                mock_collection
            )

            with patch("langchain.vectorstores.Chroma") as mock_langchain_chroma:
                mock_lc_instance = Mock()
                mock_lc_instance.add_documents.return_value = ["id1", "id2", "id3"]
                mock_langchain_chroma.return_value = mock_lc_instance

                store = ChromaVectorStore(vector_config, embeddings=mock_embeddings)
                await store.initialize()

                ids = await store.add_documents(sample_documents)

                assert ids == ["id1", "id2", "id3"]
                mock_lc_instance.add_documents.assert_called_once()

    @pytest.mark.asyncio
    async def test_search(self, vector_config, mock_embeddings):
        """Test document search"""
        with patch("chromadb.Client") as mock_client:
            mock_collection = Mock()
            mock_client.return_value.get_or_create_collection.return_value = (
                mock_collection
            )

            with patch("langchain.vectorstores.Chroma") as mock_langchain_chroma:
                mock_lc_instance = Mock()
                mock_lc_instance.similarity_search_with_score.return_value = [
                    (Document(page_content="Result 1", metadata={"score": 0.9}), 0.1),
                    (Document(page_content="Result 2", metadata={"score": 0.8}), 0.2),
                ]
                mock_langchain_chroma.return_value = mock_lc_instance

                store = ChromaVectorStore(vector_config, embeddings=mock_embeddings)
                await store.initialize()

                results = await store.search("test query", k=2)

                assert len(results) == 2
                assert isinstance(results[0], SearchResult)
                assert results[0].document.page_content == "Result 1"
                assert results[0].score == 0.9  # 1 - 0.1

    @pytest.mark.asyncio
    async def test_delete_documents(self, vector_config, mock_embeddings):
        """Test document deletion"""
        with patch("chromadb.Client") as mock_client:
            mock_collection = Mock()
            mock_client.return_value.get_or_create_collection.return_value = (
                mock_collection
            )

            with patch("langchain.vectorstores.Chroma") as mock_langchain_chroma:
                mock_lc_instance = Mock()
                mock_langchain_chroma.return_value = mock_lc_instance

                store = ChromaVectorStore(vector_config, embeddings=mock_embeddings)
                await store.initialize()

                # Test delete by IDs
                success = await store.delete(ids=["id1", "id2"])
                assert success
                mock_collection.delete.assert_called_once_with(ids=["id1", "id2"])

                # Test delete by filter
                mock_collection.reset_mock()
                success = await store.delete(filter={"source": "test.txt"})
                assert success
                mock_collection.delete.assert_called_once_with(
                    where={"source": "test.txt"}
                )

    @pytest.mark.asyncio
    async def test_hybrid_search(self, vector_config, mock_embeddings):
        """Test hybrid search functionality"""
        with patch("chromadb.Client") as mock_client:
            mock_collection = Mock()
            mock_collection.get.return_value = {
                "documents": ["Document with keyword match", "Another document"],
                "metadatas": [{"source": "test1.txt"}, {"source": "test2.txt"}],
            }
            mock_client.return_value.get_or_create_collection.return_value = (
                mock_collection
            )

            with patch("langchain.vectorstores.Chroma") as mock_langchain_chroma:
                mock_lc_instance = Mock()
                mock_lc_instance.similarity_search_with_score.return_value = [
                    (
                        Document(
                            page_content="Vector result", metadata={"source": "vec.txt"}
                        ),
                        0.1,
                    )
                ]
                mock_langchain_chroma.return_value = mock_lc_instance

                store = ChromaVectorStore(vector_config, embeddings=mock_embeddings)
                await store.initialize()

                results = await store.hybrid_search(
                    "test query", k=3, alpha=0.7, filter={"text_match": "keyword"}
                )

                assert len(results) > 0
                # Verify both vector and keyword results are included


class TestVectorStoreManager:
    """Test vector store manager"""

    @pytest.mark.asyncio
    async def test_create_store(self, vector_config):
        """Test creating a new store"""
        manager = VectorStoreManager()

        with patch.object(ChromaVectorStore, "initialize", new_callable=AsyncMock):
            store = await manager.create_store(
                store_id="test_store",
                store_type=VectorStoreType.CHROMA,
                config=vector_config,
                set_as_default=True,
            )

            assert isinstance(store, ChromaVectorStore)
            assert "test_store" in manager.stores
            assert manager.default_store == "test_store"

    def test_get_store(self, vector_config):
        """Test getting a store"""
        manager = VectorStoreManager()

        # Create mock store
        mock_store = Mock(spec=ChromaVectorStore)
        manager.stores["test_store"] = mock_store
        manager.default_store = "test_store"

        # Test get by ID
        store = manager.get_store("test_store")
        assert store == mock_store

        # Test get default
        store = manager.get_store()
        assert store == mock_store

        # Test not found
        with pytest.raises(ValueError, match="Vector store not found"):
            manager.get_store("nonexistent")

    @pytest.mark.asyncio
    async def test_search_all_stores(self):
        """Test searching across all stores"""
        manager = VectorStoreManager()

        # Create mock stores
        mock_store1 = Mock(spec=ChromaVectorStore)
        mock_store1.search = AsyncMock(
            return_value=[
                SearchResult(
                    document=Document(page_content="Result 1"), score=0.9, metadata={}
                )
            ]
        )

        mock_store2 = Mock(spec=ChromaVectorStore)
        mock_store2.search = AsyncMock(
            return_value=[
                SearchResult(
                    document=Document(page_content="Result 2"), score=0.8, metadata={}
                )
            ]
        )

        manager.stores = {"store1": mock_store1, "store2": mock_store2}

        results = await manager.search_all_stores("test query", k=2)

        assert "store1" in results
        assert "store2" in results
        assert len(results["store1"]) == 1
        assert len(results["store2"]) == 1

    @pytest.mark.asyncio
    async def test_optimize_stores(self):
        """Test store optimization"""
        manager = VectorStoreManager()

        # Create mock store with optimize method
        mock_store1 = Mock(spec=ChromaVectorStore)
        mock_store1.optimize = AsyncMock()

        # Create mock store without optimize method
        mock_store2 = Mock(spec=ChromaVectorStore)
        mock_store2.persist = AsyncMock()

        manager.stores = {"store1": mock_store1, "store2": mock_store2}

        results = await manager.optimize_stores()

        assert results["store1"] is True
        assert results["store2"] is True
        mock_store1.optimize.assert_called_once()
        mock_store2.persist.assert_called_once()


@pytest.mark.asyncio
async def test_vector_store_integration():
    """Test full vector store integration"""
    with patch("chromadb.Client") as mock_client:
        mock_collection = Mock()
        mock_collection.count.return_value = 0
        mock_client.return_value.get_or_create_collection.return_value = mock_collection

        with patch("langchain.vectorstores.Chroma") as mock_langchain_chroma:
            mock_lc_instance = Mock()
            mock_lc_instance.add_documents.return_value = ["id1", "id2"]
            mock_lc_instance.similarity_search_with_score.return_value = [
                (
                    Document(
                        page_content="Test result", metadata={"source": "test.txt"}
                    ),
                    0.1,
                )
            ]
            mock_langchain_chroma.return_value = mock_lc_instance

            # Create manager and store
            manager = VectorStoreManager()
            config = VectorStoreConfig(collection_name="test")

            store = await manager.create_store(
                store_id="test", store_type=VectorStoreType.CHROMA, config=config
            )

            # Add documents
            docs = [
                Document(page_content="Test document 1"),
                Document(page_content="Test document 2"),
            ]
            ids = await store.add_documents(docs)
            assert len(ids) == 2

            # Search
            results = await store.search("test query", k=1)
            assert len(results) == 1
            assert results[0].document.page_content == "Test result"
