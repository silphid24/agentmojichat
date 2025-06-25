"""Document processing for RAG pipeline"""

import os
from typing import List, Dict, Any, Optional
from datetime import datetime
from pathlib import Path

from langchain.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from pydantic import BaseModel, Field

from app.core.logging import logger


class ProcessedDocument(BaseModel):
    """Processed document model"""
    id: str
    filename: str
    content: str
    metadata: Dict[str, Any] = Field(default_factory=dict)
    chunks: List[Document] = Field(default_factory=list)
    processed_at: datetime = Field(default_factory=datetime.utcnow)


class DocumentProcessor:
    """Handles document loading and processing"""
    
    def __init__(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        separators: Optional[List[str]] = None
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
        # Default separators for text splitting
        self.separators = separators or [
            "\n\n",  # Paragraphs
            "\n",    # Lines
            ". ",    # Sentences
            " ",     # Words
            ""       # Characters
        ]
        
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=self.separators,
            length_function=len
        )
        
        logger.info(
            f"Initialized DocumentProcessor with chunk_size={chunk_size}, "
            f"overlap={chunk_overlap}"
        )
    
    async def process_file(self, file_path: str) -> ProcessedDocument:
        """Process a single file"""
        try:
            path = Path(file_path)
            if not path.exists():
                raise FileNotFoundError(f"File not found: {file_path}")
            
            # Check file type
            if path.suffix.lower() not in ['.txt', '.md', '.markdown']:
                raise ValueError(f"Unsupported file type: {path.suffix}")
            
            # Load document
            loader = TextLoader(file_path, encoding='utf-8')
            documents = loader.load()
            
            # Extract content
            content = "\n".join([doc.page_content for doc in documents])
            
            # Create metadata
            metadata = {
                "filename": path.name,
                "file_path": str(path.absolute()),
                "file_size": path.stat().st_size,
                "created_at": datetime.fromtimestamp(path.stat().st_ctime).isoformat(),
                "modified_at": datetime.fromtimestamp(path.stat().st_mtime).isoformat(),
                "file_type": path.suffix.lower()
            }
            
            # Split into chunks
            chunks = self.text_splitter.create_documents(
                texts=[content],
                metadatas=[metadata]
            )
            
            # Add chunk-specific metadata
            for i, chunk in enumerate(chunks):
                chunk.metadata.update({
                    "chunk_id": f"{path.stem}_chunk_{i}",
                    "chunk_index": i,
                    "total_chunks": len(chunks)
                })
            
            # Create processed document
            doc_id = f"doc_{path.stem}_{datetime.utcnow().timestamp()}"
            processed_doc = ProcessedDocument(
                id=doc_id,
                filename=path.name,
                content=content,
                metadata=metadata,
                chunks=chunks
            )
            
            logger.info(
                f"Processed document: {path.name} "
                f"({len(chunks)} chunks, {len(content)} chars)"
            )
            
            return processed_doc
            
        except Exception as e:
            logger.error(f"Error processing file {file_path}: {e}")
            raise
    
    async def process_directory(
        self,
        directory_path: str,
        pattern: str = "*.txt"
    ) -> List[ProcessedDocument]:
        """Process all matching files in a directory"""
        processed_docs = []
        path = Path(directory_path)
        
        if not path.exists() or not path.is_dir():
            raise ValueError(f"Invalid directory: {directory_path}")
        
        # Find matching files
        files = list(path.glob(pattern))
        logger.info(f"Found {len(files)} files matching pattern: {pattern}")
        
        # Process each file
        for file_path in files:
            try:
                doc = await self.process_file(str(file_path))
                processed_docs.append(doc)
            except Exception as e:
                logger.error(f"Failed to process {file_path}: {e}")
                continue
        
        return processed_docs
    
    async def process_text(
        self,
        text: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> ProcessedDocument:
        """Process raw text"""
        try:
            # Default metadata
            base_metadata = {
                "source": "raw_text",
                "created_at": datetime.utcnow().isoformat()
            }
            if metadata:
                base_metadata.update(metadata)
            
            # Split into chunks
            chunks = self.text_splitter.create_documents(
                texts=[text],
                metadatas=[base_metadata]
            )
            
            # Add chunk-specific metadata
            for i, chunk in enumerate(chunks):
                chunk.metadata.update({
                    "chunk_id": f"text_chunk_{i}",
                    "chunk_index": i,
                    "total_chunks": len(chunks)
                })
            
            # Create processed document
            doc_id = f"doc_text_{datetime.utcnow().timestamp()}"
            processed_doc = ProcessedDocument(
                id=doc_id,
                filename="raw_text",
                content=text,
                metadata=base_metadata,
                chunks=chunks
            )
            
            logger.info(f"Processed text: {len(chunks)} chunks, {len(text)} chars")
            
            return processed_doc
            
        except Exception as e:
            logger.error(f"Error processing text: {e}")
            raise
    
    def update_chunk_size(self, chunk_size: int, chunk_overlap: int) -> None:
        """Update text splitter parameters"""
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=self.separators,
            length_function=len
        )
        
        logger.info(f"Updated chunk parameters: size={chunk_size}, overlap={chunk_overlap}")