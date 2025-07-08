"""RAG (Retrieval-Augmented Generation) endpoints"""

from fastapi import APIRouter, Depends, HTTPException, status, UploadFile, File, Form
from typing import Annotated, List, Optional, Dict, Any
from pydantic import BaseModel, Field
import os
import tempfile
from pathlib import Path

from app.schemas.auth import UserInDB
from app.api.v1.endpoints.auth import get_current_user
from app.rag.pipeline import RAGQuery, RAGResponse
from app.rag.adapter import RAGPipelineAdapter
from app.core.logging import logger
from app.core.exceptions import RAGError, ValidationError
from app.core.error_handlers import handle_errors

router = APIRouter()

# Global RAG pipeline instance using adapter for enhanced_rag
rag_pipeline = RAGPipelineAdapter()


class AddTextRequest(BaseModel):
    """Request to add text to RAG"""

    text: str = Field(..., min_length=1)
    metadata: Optional[Dict[str, Any]] = None


class RAGStatsResponse(BaseModel):
    """RAG statistics response"""

    index_stats: Dict[str, Any]
    pipeline_status: str = "active"


@router.post("/query", response_model=RAGResponse)
@handle_errors()
async def query_rag(
    query: RAGQuery, current_user: Annotated[UserInDB, Depends(get_current_user)]
):
    """Query the RAG system"""
    logger.info(f"RAG query from user {current_user.username}: {query.query}")

    response = await rag_pipeline.query(query)
    if not response:
        raise RAGError("Failed to generate response", operation="query")

    return response


@router.post("/add/text")
@handle_errors()
async def add_text(
    request: AddTextRequest,
    current_user: Annotated[UserInDB, Depends(get_current_user)],
):
    """Add text to the RAG system"""
    logger.info(f"Adding text to RAG (length: {len(request.text)})")

    result = await rag_pipeline.add_text(text=request.text, metadata=request.metadata)

    if not result["success"]:
        raise RAGError(result.get("error", "Failed to add text"), operation="add_text")

    return result


@router.post("/add/file")
@handle_errors()
async def add_file(
    file: UploadFile = File(...),
    current_user: Annotated[UserInDB, Depends(get_current_user)] = None,
):
    """Add a file to the RAG system"""
    # Check file type
    allowed_extensions = [".txt", ".md", ".markdown"]
    file_ext = Path(file.filename).suffix.lower()

    if file_ext not in allowed_extensions:
        raise ValidationError(
            f"Unsupported file type. Allowed: {allowed_extensions}", field="file"
        )

    # Save uploaded file temporarily
    try:
        with tempfile.NamedTemporaryFile(
            delete=False, suffix=file_ext, mode="wb"
        ) as tmp_file:
            content = await file.read()
            tmp_file.write(content)
            tmp_path = tmp_file.name

        # Process the file
        result = await rag_pipeline.add_documents([tmp_path])

        # Clean up
        os.unlink(tmp_path)

        return {
            "filename": file.filename,
            "processed": result["processed"],
            "chunks": result["total_chunks"],
            "success": result["processed"] > 0,
        }

    except Exception as e:
        logger.error(f"Error processing file upload: {e}")
        if "tmp_path" in locals() and os.path.exists(tmp_path):
            os.unlink(tmp_path)

        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error processing file: {str(e)}",
        )


@router.post("/add/files")
async def add_multiple_files(
    files: List[UploadFile] = File(...),
    current_user: Annotated[UserInDB, Depends(get_current_user)] = None,
):
    """Add multiple files to the RAG system"""
    results = []
    temp_paths = []

    try:
        # Save all files temporarily
        for file in files:
            file_ext = Path(file.filename).suffix.lower()

            if file_ext not in [".txt", ".md", ".markdown"]:
                results.append(
                    {
                        "filename": file.filename,
                        "success": False,
                        "error": "Unsupported file type",
                    }
                )
                continue

            with tempfile.NamedTemporaryFile(
                delete=False, suffix=file_ext, mode="wb"
            ) as tmp_file:
                content = await file.read()
                tmp_file.write(content)
                temp_paths.append(tmp_file.name)

        # Process all valid files
        if temp_paths:
            process_result = await rag_pipeline.add_documents(temp_paths)

            # Map results back to filenames
            for i, file in enumerate(files):
                if i < len(temp_paths):
                    results.append(
                        {
                            "filename": file.filename,
                            "success": True,
                            "chunks": process_result.get("total_chunks", 0)
                            // len(temp_paths),
                        }
                    )

        return {
            "total_files": len(files),
            "processed": len(temp_paths),
            "results": results,
        }

    finally:
        # Clean up all temp files
        for path in temp_paths:
            if os.path.exists(path):
                os.unlink(path)


@router.get("/stats", response_model=RAGStatsResponse)
async def get_rag_stats(current_user: Annotated[UserInDB, Depends(get_current_user)]):
    """Get RAG system statistics"""
    try:
        index_stats = rag_pipeline.get_index_stats()

        return RAGStatsResponse(index_stats=index_stats, pipeline_status="active")
    except Exception as e:
        logger.error(f"Error getting RAG stats: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error getting stats: {str(e)}",
        )


@router.delete("/index")
async def clear_index(current_user: Annotated[UserInDB, Depends(get_current_user)]):
    """Clear the RAG index"""
    try:
        rag_pipeline.clear_index()
        return {"message": "RAG index cleared successfully"}
    except Exception as e:
        logger.error(f"Error clearing index: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error clearing index: {str(e)}",
        )


@router.put("/config/chunks")
async def update_chunk_config(
    chunk_size: int = Form(..., ge=100, le=5000),
    chunk_overlap: int = Form(..., ge=0, le=1000),
    current_user: Annotated[UserInDB, Depends(get_current_user)] = None,
):
    """Update chunk size configuration"""
    try:
        await rag_pipeline.update_chunk_size(chunk_size, chunk_overlap)
        return {
            "message": "Chunk configuration updated",
            "chunk_size": chunk_size,
            "chunk_overlap": chunk_overlap,
        }
    except Exception as e:
        logger.error(f"Error updating chunk config: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error updating configuration: {str(e)}",
        )
