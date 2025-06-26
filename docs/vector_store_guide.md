# Vector Store Guide

## Overview

The MOJI AI Agent system includes a sophisticated vector store implementation that supports multiple vector database backends, hybrid search capabilities, and efficient document management. This guide covers the architecture, usage, and best practices for the vector store system.

## Architecture

### Core Components

1. **BaseVectorStore** (`app/vectorstore/base.py`)
   - Abstract base class defining the vector store interface
   - Common operations: add, search, delete, update
   - Support for hybrid search and relevance filtering

2. **ChromaVectorStore** (`app/vectorstore/chroma_store.py`)
   - Chroma DB implementation
   - Persistent storage support
   - Hybrid search combining vector and keyword search
   - Efficient metadata filtering

3. **VectorStoreManager** (`app/vectorstore/manager.py`)
   - Manages multiple vector stores
   - Store routing and selection
   - Cross-store search capabilities
   - Data migration between stores

### Vector Store Types

Currently supported:
- **FAISS**: In-memory vector store (via existing RAG implementation)
- **Chroma**: Persistent vector store with hybrid search

Future support planned:
- **Weaviate**: GraphQL-based vector database
- **Pinecone**: Cloud-native vector database

## API Endpoints

### Store Management

#### Create Store
```http
POST /api/v1/vectorstore/stores
{
  "store_id": "knowledge_base",
  "store_type": "chroma",
  "collection_name": "my_documents",
  "persist_directory": "data/my_vectors",
  "distance_metric": "cosine",
  "set_as_default": true
}
```

#### List Stores
```http
GET /api/v1/vectorstore/stores
```

#### Delete Store
```http
DELETE /api/v1/vectorstore/stores/{store_id}
```

### Document Operations

#### Add Documents
```http
POST /api/v1/vectorstore/documents
{
  "texts": ["Document 1 content", "Document 2 content"],
  "metadatas": [
    {"source": "doc1.pdf", "page": 1},
    {"source": "doc2.pdf", "page": 1}
  ],
  "store_id": "knowledge_base"
}
```

#### Upload and Index File
```http
POST /api/v1/vectorstore/documents/upload
Content-Type: multipart/form-data

file: <file>
store_id: knowledge_base
chunk_size: 1000
chunk_overlap: 200
```

#### Delete Documents
```http
DELETE /api/v1/vectorstore/documents
{
  "ids": ["doc_id_1", "doc_id_2"],
  "store_id": "knowledge_base"
}
```

### Search Operations

#### Basic Search
```http
POST /api/v1/vectorstore/search
{
  "query": "What is machine learning?",
  "k": 5,
  "filter": {"source": "ml_book.pdf"},
  "store_id": "knowledge_base"
}
```

#### Hybrid Search
```http
POST /api/v1/vectorstore/search
{
  "query": "neural networks",
  "k": 5,
  "use_hybrid": true,
  "hybrid_alpha": 0.7,
  "store_id": "knowledge_base"
}
```

#### Search All Stores
```http
POST /api/v1/vectorstore/search/all
{
  "query": "AI applications",
  "k": 3
}
```

### Store Maintenance

#### Clear Store
```http
POST /api/v1/vectorstore/stores/{store_id}/clear
```

#### Optimize Stores
```http
POST /api/v1/vectorstore/stores/optimize
```

#### Migrate Data
```http
POST /api/v1/vectorstore/stores/migrate?from_store_id=store1&to_store_id=store2
```

## Usage Examples

### Python Client Example

```python
import httpx
from typing import List, Dict, Any

class VectorStoreClient:
    def __init__(self, base_url: str, api_key: str):
        self.base_url = base_url
        self.headers = {"Authorization": f"Bearer {api_key}"}
    
    async def create_store(self, store_id: str, store_type: str = "chroma"):
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{self.base_url}/api/v1/vectorstore/stores",
                json={
                    "store_id": store_id,
                    "store_type": store_type,
                    "collection_name": f"{store_id}_collection"
                },
                headers=self.headers
            )
            return response.json()
    
    async def add_documents(self, texts: List[str], store_id: str = None):
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{self.base_url}/api/v1/vectorstore/documents",
                json={
                    "texts": texts,
                    "store_id": store_id
                },
                headers=self.headers
            )
            return response.json()
    
    async def search(self, query: str, k: int = 5, store_id: str = None):
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{self.base_url}/api/v1/vectorstore/search",
                json={
                    "query": query,
                    "k": k,
                    "store_id": store_id
                },
                headers=self.headers
            )
            return response.json()

# Usage
client = VectorStoreClient("http://localhost:8000", "your_api_key")

# Create a specialized store
await client.create_store("technical_docs", "chroma")

# Add documents
await client.add_documents([
    "Machine learning is a subset of AI",
    "Deep learning uses neural networks"
], "technical_docs")

# Search
results = await client.search("What is deep learning?", k=3, store_id="technical_docs")
```

### Enhanced RAG Pipeline Usage

```python
from app.rag.enhanced_pipeline import enhanced_rag_pipeline, EnhancedRAGQuery

# Create specialized stores
await enhanced_rag_pipeline.create_specialized_store(
    store_id="product_knowledge",
    store_type=VectorStoreType.CHROMA,
    collection_name="products",
    description="Product documentation and specifications"
)

await enhanced_rag_pipeline.create_specialized_store(
    store_id="support_tickets",
    store_type=VectorStoreType.CHROMA,
    collection_name="tickets",
    description="Historical support tickets and resolutions"
)

# Add documents to specific store
await enhanced_rag_pipeline.add_documents(
    file_paths=["docs/product_manual.pdf"],
    store_id="product_knowledge"
)

# Query single store
query = EnhancedRAGQuery(
    query="How to configure the API endpoint?",
    k=5,
    store_id="product_knowledge",
    use_hybrid=True
)
response = await enhanced_rag_pipeline.query(query)

# Query all stores
query = EnhancedRAGQuery(
    query="Error message: connection timeout",
    k=3,
    use_all_stores=True
)
response = await enhanced_rag_pipeline.query(query)
```

## Best Practices

### 1. Store Organization

- Create specialized stores for different content types
- Use meaningful store IDs and descriptions
- Separate frequently updated content from static content

### 2. Document Processing

- Choose appropriate chunk sizes based on content type
  - Technical docs: 1000-1500 tokens
  - Conversational data: 500-800 tokens
  - Code: 200-500 tokens
- Use sufficient chunk overlap (10-20% of chunk size)
- Include relevant metadata for filtering

### 3. Search Optimization

- Use hybrid search for better recall
- Adjust alpha parameter based on use case
  - Higher alpha (0.7-0.9): Emphasize semantic similarity
  - Lower alpha (0.3-0.5): Emphasize keyword matching
- Apply score thresholds to filter irrelevant results

### 4. Performance Tuning

- Regularly optimize stores to improve query performance
- Monitor store statistics and growth
- Consider data migration for store consolidation
- Use appropriate distance metrics:
  - Cosine: General text similarity
  - Euclidean: When magnitude matters
  - Dot product: Pre-normalized embeddings

### 5. Multi-Store Strategies

- **Domain Separation**: Create stores for different domains
  ```
  - technical_docs: Engineering documentation
  - business_docs: Business processes and policies
  - customer_data: Customer interactions and feedback
  ```

- **Temporal Separation**: Separate by time periods
  ```
  - current_quarter: Recent data for fast access
  - archive_2023: Historical data for reference
  ```

- **Access Pattern Separation**: Based on query patterns
  ```
  - frequently_accessed: Hot data with optimized indices
  - cold_storage: Rarely accessed archival data
  ```

## Troubleshooting

### Common Issues

1. **Slow Search Performance**
   - Run store optimization
   - Check index parameters
   - Consider reducing result count (k)
   - Monitor embedding generation time

2. **Poor Search Results**
   - Verify chunk size is appropriate
   - Check if documents were properly processed
   - Try hybrid search with different alpha values
   - Review document metadata for filtering

3. **Storage Issues**
   - Check disk space for persistent stores
   - Clean up old indices
   - Use store migration to consolidate data
   - Enable compression if supported

4. **Memory Usage**
   - Monitor vector store memory consumption
   - Use persistent stores instead of in-memory
   - Implement pagination for large result sets
   - Clear unused stores

## Configuration

### Environment Variables

```bash
# Vector Store Configuration
VECTOR_STORE_TYPE=chroma
VECTOR_PERSIST_DIR=data/vectors
VECTOR_COLLECTION_NAME=moji_vectors

# Chroma Specific
CHROMA_SERVER_HOST=localhost
CHROMA_SERVER_PORT=8000
CHROMA_SERVER_SSL=false

# Performance
VECTOR_BATCH_SIZE=100
VECTOR_MAX_WORKERS=4
```

### Advanced Configuration

```python
# Custom Chroma configuration
from chromadb.config import Settings

custom_settings = Settings(
    chroma_db_impl="duckdb+parquet",
    persist_directory="data/custom_chroma",
    anonymized_telemetry=False,
    allow_reset=True
)

# Custom index parameters
index_params = {
    "hnsw_space": "cosine",
    "hnsw_construction_ef": 200,
    "hnsw_search_ef": 100,
    "hnsw_M": 16,
    "hnsw_num_threads": 4
}
```

## Future Enhancements

1. **Additional Vector Stores**
   - Weaviate integration
   - Pinecone support
   - Qdrant implementation

2. **Advanced Features**
   - Real-time index updates
   - Distributed vector search
   - GPU-accelerated operations
   - Custom embedding models per store

3. **Management Tools**
   - Web UI for store management
   - Automated backup and restore
   - Performance monitoring dashboard
   - A/B testing for search configurations