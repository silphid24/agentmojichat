"""Tests for vector store API endpoints"""

import pytest
from fastapi.testclient import TestClient
from unittest.mock import Mock, AsyncMock, patch
from langchain.schema import Document

from app.main import app
from app.vectorstore.base import SearchResult
from app.vectorstore.manager import VectorStoreType


@pytest.fixture
def client():
    """Test client"""
    return TestClient(app)


@pytest.fixture
def auth_headers():
    """Authentication headers for testing"""
    return {"Authorization": "Bearer test_token"}


@pytest.fixture
def mock_auth(monkeypatch):
    """Mock authentication"""
    mock_user = "test_user"
    monkeypatch.setattr(
        "app.api.v1.endpoints.vectorstore.get_current_user",
        lambda: mock_user
    )
    return mock_user


class TestVectorStoreAPI:
    """Test vector store API endpoints"""
    
    def test_create_store(self, client, auth_headers, mock_auth):
        """Test creating a vector store"""
        with patch('app.api.v1.endpoints.vectorstore.vector_store_manager') as mock_manager:
            mock_store = Mock()
            mock_store.get_collection_stats = AsyncMock(return_value={
                "store_type": "chroma",
                "document_count": 0
            })
            mock_manager.create_store = AsyncMock(return_value=mock_store)
            
            response = client.post(
                "/api/v1/vectorstore/stores",
                json={
                    "store_id": "test_store",
                    "store_type": "chroma",
                    "collection_name": "test_collection",
                    "set_as_default": True
                },
                headers=auth_headers
            )
            
            assert response.status_code == 200
            data = response.json()
            assert data["store_id"] == "test_store"
            assert data["store_type"] == "chroma"
            assert data["is_default"] == True
    
    def test_list_stores(self, client, auth_headers, mock_auth):
        """Test listing vector stores"""
        with patch('app.api.v1.endpoints.vectorstore.vector_store_manager') as mock_manager:
            mock_manager.get_stats = AsyncMock(return_value={
                "store1": {"document_count": 100},
                "store2": {"document_count": 200},
                "default_store": "store1",
                "total_stores": 2
            })
            
            response = client.get(
                "/api/v1/vectorstore/stores",
                headers=auth_headers
            )
            
            assert response.status_code == 200
            data = response.json()
            assert data["total_stores"] == 2
            assert data["default_store"] == "store1"
    
    def test_search_vectors(self, client, auth_headers, mock_auth):
        """Test vector search"""
        with patch('app.api.v1.endpoints.vectorstore.vector_store_manager') as mock_manager:
            mock_store = Mock()
            mock_store.search = AsyncMock(return_value=[
                SearchResult(
                    document=Document(
                        page_content="Search result 1",
                        metadata={"source": "doc1.txt"}
                    ),
                    score=0.95,
                    metadata={"source": "doc1.txt"}
                ),
                SearchResult(
                    document=Document(
                        page_content="Search result 2",
                        metadata={"source": "doc2.txt"}
                    ),
                    score=0.85,
                    metadata={"source": "doc2.txt"}
                )
            ])
            mock_manager.get_store.return_value = mock_store
            mock_manager.default_store = "default"
            
            response = client.post(
                "/api/v1/vectorstore/search",
                json={
                    "query": "test search query",
                    "k": 2,
                    "filter": {"source": "doc1.txt"}
                },
                headers=auth_headers
            )
            
            assert response.status_code == 200
            data = response.json()
            assert data["query"] == "test search query"
            assert len(data["results"]) == 2
            assert data["results"][0]["score"] == 0.95
            assert data["results"][0]["content"] == "Search result 1"
    
    def test_hybrid_search(self, client, auth_headers, mock_auth):
        """Test hybrid search"""
        with patch('app.api.v1.endpoints.vectorstore.vector_store_manager') as mock_manager:
            mock_store = Mock()
            mock_store.hybrid_search = AsyncMock(return_value=[
                SearchResult(
                    document=Document(page_content="Hybrid result"),
                    score=0.9,
                    metadata={}
                )
            ])
            mock_manager.get_store.return_value = mock_store
            
            response = client.post(
                "/api/v1/vectorstore/search",
                json={
                    "query": "test query",
                    "k": 5,
                    "use_hybrid": True,
                    "hybrid_alpha": 0.7
                },
                headers=auth_headers
            )
            
            assert response.status_code == 200
            data = response.json()
            assert len(data["results"]) == 1
            mock_store.hybrid_search.assert_called_once_with(
                query="test query",
                k=5,
                alpha=0.7,
                filter=None
            )
    
    def test_add_documents(self, client, auth_headers, mock_auth):
        """Test adding documents"""
        with patch('app.api.v1.endpoints.vectorstore.vector_store_manager') as mock_manager:
            mock_store = Mock()
            mock_store.add_documents = AsyncMock(return_value=["id1", "id2", "id3"])
            mock_manager.get_store.return_value = mock_store
            mock_manager.default_store = "default"
            
            response = client.post(
                "/api/v1/vectorstore/documents",
                json={
                    "texts": [
                        "Document 1 content",
                        "Document 2 content",
                        "Document 3 content"
                    ],
                    "metadatas": [
                        {"source": "doc1"},
                        {"source": "doc2"},
                        {"source": "doc3"}
                    ]
                },
                headers=auth_headers
            )
            
            assert response.status_code == 200
            data = response.json()
            assert data["message"] == "Added 3 documents"
            assert len(data["document_ids"]) == 3
    
    def test_upload_and_index_file(self, client, auth_headers, mock_auth):
        """Test file upload and indexing"""
        with patch('app.api.v1.endpoints.vectorstore.vector_store_manager') as mock_manager:
            mock_store = Mock()
            mock_store.add_documents = AsyncMock(return_value=["id1", "id2"])
            mock_manager.get_store.return_value = mock_store
            mock_manager.default_store = "default"
            
            with patch('app.api.v1.endpoints.vectorstore.DocumentProcessor') as mock_processor:
                mock_processor_instance = Mock()
                mock_processor_instance.process_file = AsyncMock(return_value=[
                    Document(page_content="Chunk 1"),
                    Document(page_content="Chunk 2")
                ])
                mock_processor.return_value = mock_processor_instance
                
                # Create test file
                test_content = b"Test file content for vector store"
                
                response = client.post(
                    "/api/v1/vectorstore/documents/upload",
                    files={"file": ("test.txt", test_content, "text/plain")},
                    data={
                        "chunk_size": 500,
                        "chunk_overlap": 50
                    },
                    headers=auth_headers
                )
                
                assert response.status_code == 200
                data = response.json()
                assert data["chunks_created"] == 2
                assert len(data["document_ids"]) == 2
                assert data["message"] == "Processed and indexed test.txt"
    
    def test_delete_documents(self, client, auth_headers, mock_auth):
        """Test document deletion"""
        with patch('app.api.v1.endpoints.vectorstore.vector_store_manager') as mock_manager:
            mock_store = Mock()
            mock_store.delete = AsyncMock(return_value=True)
            mock_manager.get_store.return_value = mock_store
            mock_manager.default_store = "default"
            
            response = client.delete(
                "/api/v1/vectorstore/documents",
                json={
                    "ids": ["id1", "id2", "id3"]
                },
                headers=auth_headers
            )
            
            assert response.status_code == 200
            data = response.json()
            assert data["message"] == "Documents deleted successfully"
            
            mock_store.delete.assert_called_once_with(
                ids=["id1", "id2", "id3"],
                filter=None
            )
    
    def test_clear_store(self, client, auth_headers, mock_auth):
        """Test clearing a store"""
        with patch('app.api.v1.endpoints.vectorstore.vector_store_manager') as mock_manager:
            mock_store = Mock()
            mock_store.clear = AsyncMock(return_value=True)
            mock_manager.get_store.return_value = mock_store
            
            response = client.post(
                "/api/v1/vectorstore/stores/test_store/clear",
                headers=auth_headers
            )
            
            assert response.status_code == 200
            data = response.json()
            assert data["message"] == "Store test_store cleared successfully"
    
    def test_optimize_stores(self, client, auth_headers, mock_auth):
        """Test store optimization"""
        with patch('app.api.v1.endpoints.vectorstore.vector_store_manager') as mock_manager:
            mock_manager.optimize_stores = AsyncMock(return_value={
                "store1": True,
                "store2": True,
                "store3": False
            })
            
            response = client.post(
                "/api/v1/vectorstore/stores/optimize",
                headers=auth_headers
            )
            
            assert response.status_code == 200
            data = response.json()
            assert data["message"] == "Optimization completed"
            assert data["results"]["store1"] == True
            assert data["results"]["store3"] == False
    
    def test_migrate_data(self, client, auth_headers, mock_auth):
        """Test data migration between stores"""
        with patch('app.api.v1.endpoints.vectorstore.vector_store_manager') as mock_manager:
            mock_manager.migrate_data = AsyncMock(return_value=True)
            
            response = client.post(
                "/api/v1/vectorstore/stores/migrate",
                params={
                    "from_store_id": "store1",
                    "to_store_id": "store2"
                },
                headers=auth_headers
            )
            
            assert response.status_code == 200
            data = response.json()
            assert data["success"] == True
            assert "migrated from store1 to store2" in data["message"]
            
            mock_manager.migrate_data.assert_called_once_with(
                from_store_id="store1",
                to_store_id="store2"
            )
    
    def test_search_all_stores(self, client, auth_headers, mock_auth):
        """Test searching across all stores"""
        with patch('app.api.v1.endpoints.vectorstore.vector_store_manager') as mock_manager:
            mock_manager.search_all_stores = AsyncMock(return_value={
                "store1": [
                    SearchResult(
                        document=Document(page_content="Result from store1"),
                        score=0.9,
                        metadata={}
                    )
                ],
                "store2": [
                    SearchResult(
                        document=Document(page_content="Result from store2"),
                        score=0.85,
                        metadata={}
                    )
                ]
            })
            
            response = client.post(
                "/api/v1/vectorstore/search/all",
                json={
                    "query": "search all stores",
                    "k": 3
                },
                headers=auth_headers
            )
            
            assert response.status_code == 200
            data = response.json()
            assert data["query"] == "search all stores"
            assert data["total_results"] == 2
            assert "store1" in data["results_by_store"]
            assert "store2" in data["results_by_store"]
    
    def test_delete_store(self, client, auth_headers, mock_auth):
        """Test deleting a store"""
        with patch('app.api.v1.endpoints.vectorstore.vector_store_manager') as mock_manager:
            mock_manager.remove_store.return_value = True
            
            response = client.delete(
                "/api/v1/vectorstore/stores/test_store",
                headers=auth_headers
            )
            
            assert response.status_code == 200
            data = response.json()
            assert data["message"] == "Store test_store deleted successfully"
            
            mock_manager.remove_store.assert_called_once_with("test_store")