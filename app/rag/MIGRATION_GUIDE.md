# RAG Module Migration Guide

## Overview
The RAG module has been consolidated to use `enhanced_rag.py` as the primary implementation. 

## Migration Status

### Active Implementation
- **enhanced_rag.py**: Primary RAG implementation with query rewriting and confidence scoring
- **adapter.py**: Adapter to maintain backwards compatibility with existing API

### Deprecated Files
- **pipeline.py**: Basic RAG implementation (deprecated)
- **enhanced_pipeline.py**: Multi-store implementation (deprecated, features not yet migrated)

## Migration Path

### For API Endpoints
The `/app/api/v1/endpoints/rag.py` now uses `RAGPipelineAdapter` which wraps `EnhancedRAGPipeline`:

```python
# Old
from app.rag.pipeline import RAGPipeline
rag_pipeline = RAGPipeline()

# New
from app.rag.adapter import RAGPipelineAdapter
rag_pipeline = RAGPipelineAdapter()
```

### For Direct Usage
If you're using RAG directly, migrate to `EnhancedRAGPipeline`:

```python
# Old
from app.rag.pipeline import RAGPipeline
pipeline = RAGPipeline()

# New
from app.rag.enhanced_rag import EnhancedRAGPipeline
pipeline = EnhancedRAGPipeline()
```

## Key Differences

### Features Added
1. **Query Rewriting**: Automatically generates alternative queries for better search
2. **Confidence Scoring**: Returns confidence levels (HIGH/MEDIUM/LOW) for answers
3. **Better Korean Support**: Optimized prompts for Korean language
4. **Document Loading**: Built-in support for .txt, .md, .docx files

### API Changes
- Score threshold adjusted for ChromaDB (lower is better, default 1.6)
- Enhanced metadata in responses including confidence and reasoning
- Source citations included in responses

### Configuration Changes
- Uses OpenAI embeddings (text-embedding-3-small) by default
- Requires OPENAI_API_KEY or compatible LLM_API_KEY

## Future Work
- Migrate multi-store functionality from enhanced_pipeline.py
- Add conversation history support from pipeline.py
- Implement hybrid search capabilities