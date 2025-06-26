"""Vector store API endpoints"""

from typing import List, Dict, Any, Optional
from fastapi import APIRouter, HTTPException, Depends, UploadFile, File, Form
from pydantic import BaseModel, Field

from app.vectorstore.manager import vector_store_manager, VectorStoreType
from app.vectorstore.base import VectorStoreConfig
from app.rag.document_processor import DocumentProcessor
from app.api.v1.endpoints.auth import get_current_user
from app.core.logging import logger

router = APIRouter()


class CreateStoreRequest(BaseModel):
    """Request model for creating a vector store"""
    store_id: str
    store_type: VectorStoreType
    collection_name: str = "moji_vectors"
    persist_directory: Optional[str] = None
    distance_metric: str = "cosine"
    set_as_default: bool = False


class SearchRequest(BaseModel):
    """Request model for searching vectors"""
    query: str
    k: int = Field(default=4, ge=1, le=100)
    filter: Optional[Dict[str, Any]] = None
    store_id: Optional[str] = None
    use_hybrid: bool = False
    hybrid_alpha: float = Field(default=0.5, ge=0.0, le=1.0)


class AddDocumentsRequest(BaseModel):
    """Request model for adding documents"""
    texts: List[str]
    metadatas: Optional[List[Dict[str, Any]]] = None
    store_id: Optional[str] = None


class DeleteRequest(BaseModel):
    """Request model for deleting documents"""
    ids: Optional[List[str]] = None
    filter: Optional[Dict[str, Any]] = None
    store_id: Optional[str] = None


@router.post("/stores")
async def create_store(
    request: CreateStoreRequest,
    current_user: str = Depends(get_current_user)
):
    """Create a new vector store"""
    try:
        config = VectorStoreConfig(
            collection_name=request.collection_name,
            persist_directory=request.persist_directory,
            distance_metric=request.distance_metric
        )
        
        store = await vector_store_manager.create_store(
            store_id=request.store_id,
            store_type=request.store_type,
            config=config,
            set_as_default=request.set_as_default
        )
        
        stats = await store.get_collection_stats()
        
        return {
            "store_id": request.store_id,
            "store_type": request.store_type,
            "is_default": request.set_as_default,
            "stats": stats
        }
        
    except Exception as e:
        logger.error(f"Error creating store: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/stores")
async def list_stores(
    current_user: str = Depends(get_current_user)
):
    """List all vector stores with their statistics"""
    try:
        stats = await vector_store_manager.get_stats()
        return stats
        
    except Exception as e:
        logger.error(f"Error listing stores: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/stores/{store_id}")
async def delete_store(
    store_id: str,
    current_user: str = Depends(get_current_user)
):
    """Delete a vector store"""
    try:
        success = vector_store_manager.remove_store(store_id)
        if not success:
            raise HTTPException(status_code=404, detail="Store not found")
            
        return {"message": f"Store {store_id} deleted successfully"}
        
    except Exception as e:
        logger.error(f"Error deleting store: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/search")
async def search_vectors(
    request: SearchRequest,
    current_user: str = Depends(get_current_user)
):
    """Search for similar documents"""
    try:
        store = vector_store_manager.get_store(request.store_id)
        
        if request.use_hybrid and hasattr(store, 'hybrid_search'):
            results = await store.hybrid_search(
                query=request.query,
                k=request.k,
                alpha=request.hybrid_alpha,
                filter=request.filter
            )
        else:
            results = await store.search(
                query=request.query,
                k=request.k,
                filter=request.filter
            )
        
        # Convert results to response format
        response = []
        for result in results:
            response.append({
                "content": result.document.page_content,
                "metadata": result.document.metadata,
                "score": result.score
            })
        
        return {
            "query": request.query,
            "results": response,
            "count": len(response),
            "store_id": request.store_id or vector_store_manager.default_store
        }
        
    except Exception as e:
        logger.error(f"Error searching vectors: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/search/all")
async def search_all_stores(
    request: SearchRequest,
    current_user: str = Depends(get_current_user)
):
    """Search across all vector stores"""
    try:
        results = await vector_store_manager.search_all_stores(
            query=request.query,
            k=request.k,
            filter=request.filter
        )
        
        # Format results by store
        formatted_results = {}
        for store_id, store_results in results.items():
            formatted_results[store_id] = [
                {
                    "content": result.document.page_content,
                    "metadata": result.document.metadata,
                    "score": result.score
                }
                for result in store_results
            ]
        
        return {
            "query": request.query,
            "results_by_store": formatted_results,
            "total_results": sum(len(r) for r in formatted_results.values())
        }
        
    except Exception as e:
        logger.error(f"Error searching all stores: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/documents")
async def add_documents(
    request: AddDocumentsRequest,
    current_user: str = Depends(get_current_user)
):
    """Add documents to a vector store"""
    try:
        store = vector_store_manager.get_store(request.store_id)
        
        # Create Document objects
        from langchain.schema import Document
        documents = []
        for i, text in enumerate(request.texts):
            metadata = request.metadatas[i] if request.metadatas else {}
            documents.append(Document(page_content=text, metadata=metadata))
        
        # Add documents
        ids = await store.add_documents(documents)
        
        return {
            "message": f"Added {len(documents)} documents",
            "document_ids": ids,
            "store_id": request.store_id or vector_store_manager.default_store
        }
        
    except Exception as e:
        logger.error(f"Error adding documents: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/documents/upload")
async def upload_and_index_file(
    file: UploadFile = File(...),
    store_id: Optional[str] = Form(None),
    chunk_size: int = Form(1000),
    chunk_overlap: int = Form(200),
    current_user: str = Depends(get_current_user)
):
    """Upload a file and add it to the vector store"""
    try:
        # Process the uploaded file
        processor = DocumentProcessor(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
        
        # Save uploaded file temporarily
        import tempfile
        import os
        
        with tempfile.NamedTemporaryFile(delete=False, suffix=file.filename) as tmp_file:
            content = await file.read()
            tmp_file.write(content)
            tmp_file_path = tmp_file.name
        
        try:
            # Process document
            documents = await processor.process_file(tmp_file_path)
            
            # Add source metadata
            for doc in documents:
                doc.metadata["source"] = file.filename
                doc.metadata["uploaded_by"] = current_user
            
            # Add to store
            store = vector_store_manager.get_store(store_id)
            ids = await store.add_documents(documents)
            
            return {
                "message": f"Processed and indexed {file.filename}",
                "chunks_created": len(documents),
                "document_ids": ids,
                "store_id": store_id or vector_store_manager.default_store
            }
            
        finally:
            # Clean up temp file
            os.unlink(tmp_file_path)
            
    except Exception as e:
        logger.error(f"Error uploading file: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/documents")
async def delete_documents(
    request: DeleteRequest,
    current_user: str = Depends(get_current_user)
):
    """Delete documents from a vector store"""
    try:
        store = vector_store_manager.get_store(request.store_id)
        
        success = await store.delete(
            ids=request.ids,
            filter=request.filter
        )
        
        if not success:
            raise HTTPException(status_code=500, detail="Failed to delete documents")
        
        return {
            "message": "Documents deleted successfully",
            "store_id": request.store_id or vector_store_manager.default_store
        }
        
    except Exception as e:
        logger.error(f"Error deleting documents: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/stores/{store_id}/clear")
async def clear_store(
    store_id: str,
    current_user: str = Depends(get_current_user)
):
    """Clear all documents from a vector store"""
    try:
        store = vector_store_manager.get_store(store_id)
        success = await store.clear()
        
        if not success:
            raise HTTPException(status_code=500, detail="Failed to clear store")
        
        return {"message": f"Store {store_id} cleared successfully"}
        
    except Exception as e:
        logger.error(f"Error clearing store: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/stores/optimize")
async def optimize_stores(
    current_user: str = Depends(get_current_user)
):
    """Optimize all vector stores"""
    try:
        results = await vector_store_manager.optimize_stores()
        
        return {
            "message": "Optimization completed",
            "results": results
        }
        
    except Exception as e:
        logger.error(f"Error optimizing stores: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/stores/migrate")
async def migrate_store_data(
    from_store_id: str,
    to_store_id: str,
    current_user: str = Depends(get_current_user)
):
    """Migrate data between vector stores"""
    try:
        success = await vector_store_manager.migrate_data(
            from_store_id=from_store_id,
            to_store_id=to_store_id
        )
        
        if not success:
            raise HTTPException(status_code=500, detail="Migration failed")
        
        return {
            "message": f"Data migrated from {from_store_id} to {to_store_id}",
            "success": success
        }
        
    except Exception as e:
        logger.error(f"Error migrating data: {e}")
        raise HTTPException(status_code=500, detail=str(e))