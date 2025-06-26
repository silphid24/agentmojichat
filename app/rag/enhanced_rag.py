"""Enhanced RAG Pipeline with Query Rewriting and Confidence Scoring"""

import os
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
import hashlib
from pathlib import Path

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.messages import HumanMessage, SystemMessage
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate

from app.llm.router import llm_router
from app.core.logging import logger


class EnhancedRAGPipeline:
    """Enhanced RAG pipeline with query rewriting and confidence scoring"""
    
    def __init__(
        self,
        documents_dir: str = "data/documents",
        vectordb_dir: str = "data/vectordb",
        collection_name: str = "moji_documents",
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
    ):
        self.documents_dir = Path(documents_dir)
        self.vectordb_dir = Path(vectordb_dir)
        self.collection_name = collection_name
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
        # Initialize components
        from app.core.config import settings
        self.embeddings = OpenAIEmbeddings(
            openai_api_key=settings.llm_api_key,
            model="text-embedding-3-small"  # OpenAI's small embedding model
        )
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )
        
        # Initialize or load vector store
        self.vectorstore = Chroma(
            collection_name=collection_name,
            embedding_function=self.embeddings,
            persist_directory=str(self.vectordb_dir),
        )
        
        # Query rewriting prompt
        self.query_rewrite_prompt = PromptTemplate(
            template="""당신은 한국어 검색 쿼리를 개선하는 전문가입니다.
주어진 질문을 바탕으로 관련 정보를 더 잘 찾을 수 있는 3가지 대안 검색어를 생성하세요.
동의어, 관련 개념, 다른 표현 방식을 고려하세요.

원본 질문: {query}

회사 정보, 조직, 제도, 복지 등과 관련된 키워드를 포함하여 3개의 대안 검색어를 생성하세요 (한 줄에 하나씩):""",
            input_variables=["query"]
        )
        
        # Answer generation prompt with confidence scoring
        self.answer_prompt = PromptTemplate(
            template="""You are a helpful assistant answering questions based on provided context.
Analyze the context carefully and provide an accurate answer with confidence assessment.

Context:
{context}

Question: {question}

Provide your answer in the following format:
ANSWER: [Your detailed answer based on the context]
CONFIDENCE: [HIGH/MEDIUM/LOW]
REASONING: [Brief explanation of your confidence level]
SOURCES: [List the document sources used]

If the context doesn't contain relevant information, say so clearly.""",
            input_variables=["context", "question"]
        )
    
    async def load_documents(self, file_paths: Optional[List[str]] = None) -> Dict[str, Any]:
        """Load documents from files or directory"""
        try:
            documents = []
            processed_files = []
            
            if file_paths:
                # Load specific files
                for file_path in file_paths:
                    if os.path.exists(file_path):
                        docs = await self._load_single_document(file_path)
                        documents.extend(docs)
                        processed_files.append(file_path)
            else:
                # Load all documents from directory
                self.documents_dir.mkdir(parents=True, exist_ok=True)
                for file_path in self.documents_dir.glob("**/*"):
                    if file_path.is_file() and file_path.suffix in ['.txt', '.md', '.docx']:
                        docs = await self._load_single_document(str(file_path))
                        documents.extend(docs)
                        processed_files.append(str(file_path))
            
            # Add documents to vector store
            if documents:
                self.vectorstore.add_documents(documents)
                self.vectorstore.persist()
            
            return {
                "success": True,
                "processed_files": processed_files,
                "total_chunks": len(documents),
                "message": f"Successfully processed {len(processed_files)} files into {len(documents)} chunks"
            }
            
        except Exception as e:
            logger.error(f"Error loading documents: {e}")
            return {
                "success": False,
                "error": str(e),
                "message": "Failed to load documents"
            }
    
    async def _load_single_document(self, file_path: str) -> List[Document]:
        """Load and process a single document"""
        try:
            # Read file content
            if file_path.endswith('.docx'):
                # Use python-docx for Word documents
                try:
                    import docx
                    doc = docx.Document(file_path)
                    content = "\n".join([paragraph.text for paragraph in doc.paragraphs])
                except ImportError:
                    logger.warning("python-docx not installed, skipping .docx file")
                    return []
            else:
                # Text and markdown files
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
            
            # Create document ID
            doc_id = hashlib.md5(file_path.encode()).hexdigest()[:8]
            
            # Split into chunks
            chunks = self.text_splitter.split_text(content)
            
            # Create Document objects with metadata
            documents = []
            for i, chunk in enumerate(chunks):
                doc = Document(
                    page_content=chunk,
                    metadata={
                        "source": file_path,
                        "doc_id": doc_id,
                        "chunk_id": f"{doc_id}_{i}",
                        "chunk_index": i,
                        "total_chunks": len(chunks),
                        "file_name": os.path.basename(file_path),
                        "created_at": datetime.utcnow().isoformat(),
                    }
                )
                documents.append(doc)
            
            logger.info(f"Loaded {len(documents)} chunks from {file_path}")
            return documents
            
        except Exception as e:
            logger.error(f"Error loading document {file_path}: {e}")
            return []
    
    async def rewrite_query(self, query: str) -> List[str]:
        """Rewrite query to improve search results"""
        try:
            # Get LLM
            llm = await llm_router.get_langchain_model()
            
            # Create chain
            chain = LLMChain(llm=llm, prompt=self.query_rewrite_prompt)
            
            # Generate alternative queries
            result = await chain.ainvoke({"query": query})
            
            # Parse results
            alternative_queries = [q.strip() for q in result["text"].strip().split('\n') if q.strip()]
            
            # Add original query
            all_queries = [query] + alternative_queries
            
            logger.info(f"Generated {len(all_queries)} query variations")
            return all_queries
            
        except Exception as e:
            logger.error(f"Error rewriting query: {e}")
            return [query]  # Return original query on error
    
    async def search_with_rewriting(
        self, 
        query: str, 
        k: int = 5,
        score_threshold: float = 1.6
    ) -> Tuple[List[Document], Dict[str, Any]]:
        """Search with query rewriting and relevance scoring"""
        try:
            # Rewrite query
            queries = await self.rewrite_query(query)
            
            # Search with all query variations
            all_results = []
            seen_chunks = set()
            
            for q in queries:
                results = self.vectorstore.similarity_search_with_score(q, k=k*2)  # Get more results initially
                
                for doc, score in results:
                    chunk_id = doc.metadata.get('chunk_id')
                    # ChromaDB uses distance scores - lower is better
                    if chunk_id not in seen_chunks and score <= score_threshold:
                        seen_chunks.add(chunk_id)
                        all_results.append((doc, score, q))
            
            # Sort by score (ascending - lower distance is better)
            all_results.sort(key=lambda x: x[1])
            
            # Take top k results
            top_results = all_results[:k]
            
            # Extract documents and metadata
            documents = [doc for doc, _, _ in top_results]
            search_metadata = {
                "original_query": query,
                "rewritten_queries": queries,
                "total_results": len(all_results),
                "returned_results": len(documents),
                "score_threshold": score_threshold,
                "result_details": [
                    {
                        "chunk_id": doc.metadata.get('chunk_id'),
                        "source": doc.metadata.get('source'),
                        "score": score,
                        "matched_query": matched_query
                    }
                    for doc, score, matched_query in top_results
                ]
            }
            
            return documents, search_metadata
            
        except Exception as e:
            logger.error(f"Error in search with rewriting: {e}")
            return [], {"error": str(e)}
    
    async def answer_with_confidence(
        self, 
        query: str,
        k: int = 5,
        score_threshold: float = 0.7
    ) -> Dict[str, Any]:
        """Generate answer with confidence scoring and source citations"""
        try:
            # Search for relevant documents
            documents, search_metadata = await self.search_with_rewriting(
                query, k=k, score_threshold=score_threshold
            )
            
            if not documents:
                return {
                    "answer": "죄송합니다. 관련된 정보를 찾을 수 없습니다.",
                    "confidence": "LOW",
                    "reasoning": "No relevant documents found in the knowledge base",
                    "sources": [],
                    "search_metadata": search_metadata
                }
            
            # Prepare context
            context_parts = []
            for i, doc in enumerate(documents):
                source = os.path.basename(doc.metadata.get('source', 'Unknown'))
                context_parts.append(f"[Source {i+1}: {source}]\n{doc.page_content}")
            
            context = "\n\n".join(context_parts)
            
            # Generate answer with simpler prompt
            llm = await llm_router.get_langchain_model()
            
            # Use a simpler prompt for better compatibility
            simple_prompt = PromptTemplate(
                template="""다음 문맥을 바탕으로 질문에 정확하고 자세하게 답변해주세요.

문맥:
{context}

질문: {question}

답변:""",
                input_variables=["context", "question"]
            )
            
            chain = LLMChain(llm=llm, prompt=simple_prompt)
            result = await chain.ainvoke({"context": context, "question": query})
            
            # Get the answer text
            answer_text = result["text"].strip()
            
            # Simple confidence scoring based on answer length and content
            confidence = "HIGH" if len(answer_text) > 100 else "MEDIUM" if len(answer_text) > 50 else "LOW"
            
            # Map sources to actual file paths
            source_files = list(set([
                doc.metadata.get('source', 'Unknown') 
                for doc in documents
            ]))
            
            return {
                "answer": answer_text,
                "confidence": confidence,
                "reasoning": f"답변이 {len(documents)}개의 관련 문서를 바탕으로 생성되었습니다.",
                "sources": source_files,
                "search_metadata": search_metadata,
                "context_used": len(documents),
                "timestamp": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error generating answer: {e}")
            return {
                "answer": f"오류가 발생했습니다: {str(e)}",
                "confidence": "LOW",
                "reasoning": "Error occurred during processing",
                "sources": [],
                "error": str(e)
            }
    
    def get_collection_stats(self) -> Dict[str, Any]:
        """Get statistics about the vector store collection"""
        try:
            # Get collection
            collection = self.vectorstore._collection
            
            # Get stats
            count = collection.count()
            
            return {
                "collection_name": self.collection_name,
                "total_documents": count,
                "vectordb_path": str(self.vectordb_dir),
                "embedding_model": "text-embedding-3-small",
                "chunk_size": self.chunk_size,
                "chunk_overlap": self.chunk_overlap
            }
            
        except Exception as e:
            logger.error(f"Error getting collection stats: {e}")
            return {"error": str(e)}


# Global instance
rag_pipeline = EnhancedRAGPipeline()