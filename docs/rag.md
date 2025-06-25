# RAG (Retrieval-Augmented Generation) Documentation

## Overview

The RAG system in MOJI enables the AI to access and utilize external knowledge sources to provide more accurate and contextual responses. It combines document retrieval with language generation to answer questions based on specific documents or knowledge bases.

## Architecture

### Components

1. **Document Processor**: Handles document loading and chunking
2. **Embeddings**: Converts text into vector representations
3. **Vector Retriever**: Manages vector storage and similarity search
4. **RAG Pipeline**: Orchestrates the complete RAG workflow

## Features

### Document Processing
- Supports text files (.txt, .md, .markdown)
- Configurable chunk size and overlap
- Automatic metadata extraction
- UTF-8 encoding support

### Embeddings
- Local embeddings (using mock vectors for MVP)
- LLM-based embeddings (configurable provider)
- 384-dimensional vectors by default

### Vector Storage
- FAISS vector database
- Persistent storage support
- Similarity search with scoring
- Metadata filtering

### RAG Pipeline
- Document ingestion
- Query processing
- Context-aware response generation
- Source attribution

## API Endpoints

### Query RAG System
```http
POST /api/v1/rag/query
Authorization: Bearer <token>
Content-Type: application/json

{
  "query": "What is MOJI?",
  "k": 3,
  "score_threshold": 0.5,
  "include_sources": true
}
```

Response:
```json
{
  "answer": "Based on the documents, MOJI is...",
  "sources": [
    {
      "content": "MOJI is an AI assistant...",
      "metadata": {
        "filename": "moji_intro.txt",
        "chunk_id": "doc1_chunk_0"
      },
      "score": 0.85
    }
  ],
  "query": "What is MOJI?",
  "total_sources": 3
}
```

### Add Text
```http
POST /api/v1/rag/add/text
Authorization: Bearer <token>
Content-Type: application/json

{
  "text": "MOJI is an advanced AI assistant...",
  "metadata": {
    "source": "manual_input",
    "category": "introduction"
  }
}
```

### Add File
```http
POST /api/v1/rag/add/file
Authorization: Bearer <token>
Content-Type: multipart/form-data

file: <file content>
```

### Add Multiple Files
```http
POST /api/v1/rag/add/files
Authorization: Bearer <token>
Content-Type: multipart/form-data

files: <multiple files>
```

### Get Statistics
```http
GET /api/v1/rag/stats
Authorization: Bearer <token>
```

Response:
```json
{
  "index_stats": {
    "status": "active",
    "total_vectors": 150,
    "dimension": 384,
    "index_type": "IndexFlatL2"
  },
  "pipeline_status": "active"
}
```

### Clear Index
```http
DELETE /api/v1/rag/index
Authorization: Bearer <token>
```

### Update Chunk Configuration
```http
PUT /api/v1/rag/config/chunks
Authorization: Bearer <token>
Content-Type: application/x-www-form-urlencoded

chunk_size=1000&chunk_overlap=200
```

## Usage Examples

### Python SDK Usage
```python
from app.rag.pipeline import RAGPipeline, RAGQuery

# Initialize pipeline
rag = RAGPipeline()

# Add documents
await rag.add_documents([
    "docs/intro.txt",
    "docs/features.md"
])

# Query
query = RAGQuery(
    query="What are MOJI's main features?",
    k=5,
    include_sources=True
)
response = await rag.query(query)

print(response.answer)
for source in response.sources:
    print(f"- {source['metadata']['filename']}: {source['score']}")
```

### RAG Agent Usage
```python
from app.agents.rag_agent import RAGAgent

# Create RAG-enabled agent
agent = RAGAgent(rag_k=5)
await agent.initialize()

# Add knowledge
await agent.add_knowledge(
    "MOJI supports multiple platforms including Slack and Teams.",
    metadata={"category": "platforms"}
)

# Query with agent
response = await agent.process([
    HumanMessage(content="Which platforms does MOJI support?")
])
```

## Configuration

### Environment Variables
```bash
# Vector index location
VECTOR_INDEX_PATH=data/vector_index

# Embedding configuration
USE_LOCAL_EMBEDDINGS=true
EMBEDDING_MODEL=all-MiniLM-L6-v2
```

### Chunk Parameters
- **chunk_size**: Number of characters per chunk (default: 1000)
- **chunk_overlap**: Overlap between chunks (default: 200)
- **separators**: Text splitting separators (paragraphs, sentences, words)

## Best Practices

1. **Document Preparation**
   - Use clear, well-structured documents
   - Include descriptive headings
   - Avoid excessive formatting

2. **Chunk Size Selection**
   - Smaller chunks (500-1000): Better precision
   - Larger chunks (2000-3000): Better context
   - Adjust based on document type

3. **Query Optimization**
   - Be specific in queries
   - Use relevant keywords
   - Set appropriate k values (3-5 for most cases)

4. **Performance**
   - Index documents in batches
   - Monitor vector index size
   - Clear old documents periodically

## Troubleshooting

### Common Issues

1. **No results found**
   - Check if documents are indexed
   - Verify query relevance
   - Lower score threshold

2. **Poor quality results**
   - Adjust chunk size
   - Improve document quality
   - Increase k value

3. **Slow queries**
   - Reduce index size
   - Optimize chunk parameters
   - Use local embeddings

### Debug Mode
Enable debug logging:
```python
import logging
logging.getLogger("app.rag").setLevel(logging.DEBUG)
```

## Future Enhancements

- PDF and DOCX support
- Advanced chunking strategies
- Hybrid search (keyword + semantic)
- Multi-language support
- Real-time document updates
- Query result re-ranking
- Custom embedding models