"""Enhanced RAG Pipeline with Query Rewriting and Confidence Scoring"""

import os
import time
import asyncio
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
from app.core.cache import (
    get_cached_query_result, cache_query_result,
    get_cached_embedding, cache_embedding,
    get_cached_llm_response, cache_llm_response
)
from app.core.async_utils import (
    ParallelSearchManager, AsyncBatchProcessor,
    async_retry, async_timeout
)
from app.core.adaptive_features import (
    analyze_query_complexity, get_optimal_search_config,
    response_time_tracker, adaptive_feature_manager
)
from app.core.model_optimization import (
    get_optimized_embedding_model, model_manager,
    adaptive_selector, warm_up_all_models
)
from app.core.monitoring import performance_monitor, timer


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
        
        # Use optimized embedding model (싱글톤 패턴으로 공유)
        try:
            self.embeddings = get_optimized_embedding_model()
            logger.info("Using optimized embedding model")
        except Exception as e:
            logger.warning(f"Failed to get optimized embedding model, using fallback: {e}")
            from app.core.cached_embeddings import CachedOpenAIEmbeddings
            api_key = settings.openai_api_key or settings.llm_api_key
            self.embeddings = CachedOpenAIEmbeddings(
                openai_api_key=api_key,
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
        
        # 병렬 처리 관리자들
        self.parallel_search_manager = ParallelSearchManager(max_concurrent=4)
        self.batch_processor = AsyncBatchProcessor(batch_size=10, max_concurrent=3)
        
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
    
    async def rewrite_query(self, query: str, force_rewrite: bool = False) -> List[str]:
        """Rewrite query to improve search results"""
        try:
            # 적응형 결정: 쿼리 재작성 필요 여부 확인
            if not force_rewrite:
                optimal_config = get_optimal_search_config(query)
                if not optimal_config["use_query_rewriting"]:
                    logger.info("Query rewriting skipped based on adaptive analysis")
                    return [query]
            
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
        k: int = 2, #문서 참조 걔수 
        score_threshold: float = 1.2,
        use_parallel_search: bool = True
    ) -> Tuple[List[Document], Dict[str, Any]]:
        """Search with query rewriting and relevance scoring"""
        try:
            # Rewrite query
            queries = await self.rewrite_query(query)
            
            # Search with all query variations (병렬 또는 순차)
            if use_parallel_search and len(queries) > 1:
                all_results = await self._parallel_vector_search(queries, k, score_threshold)
            else:
                all_results = await self._sequential_vector_search(queries, k, score_threshold)
            
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
    
    async def _parallel_vector_search(
        self, 
        queries: List[str], 
        k: int, 
        score_threshold: float
    ) -> List[Tuple[Document, float, str]]:
        """병렬 벡터 검색 실행"""
        try:
            # 각 쿼리에 대한 검색 태스크 생성
            search_tasks = []
            for query in queries:
                task = self._single_vector_search(query, k*2, score_threshold)
                search_tasks.append(task)
            
            # 병렬 실행
            start_time = time.time()
            results_per_query = await asyncio.gather(*search_tasks, return_exceptions=True)
            elapsed_time = time.time() - start_time
            
            logger.info(f"Parallel vector search completed in {elapsed_time:.3f}s for {len(queries)} queries")
            
            # 결과 통합 및 중복 제거
            all_results = []
            seen_chunks = set()
            
            for i, results in enumerate(results_per_query):
                if isinstance(results, Exception):
                    logger.error(f"Search failed for query '{queries[i]}': {results}")
                    continue
                
                for doc, score in results:
                    chunk_id = doc.metadata.get('chunk_id')
                    if chunk_id not in seen_chunks:
                        seen_chunks.add(chunk_id)
                        all_results.append((doc, score, queries[i]))
            
            return all_results
            
        except Exception as e:
            logger.error(f"Parallel vector search error: {e}")
            # 폴백: 순차 검색
            return await self._sequential_vector_search(queries, k, score_threshold)
    
    async def _sequential_vector_search(
        self, 
        queries: List[str], 
        k: int, 
        score_threshold: float
    ) -> List[Tuple[Document, float, str]]:
        """순차 벡터 검색 실행"""
        all_results = []
        seen_chunks = set()
        
        for q in queries:
            try:
                results = await self._single_vector_search(q, k*2, score_threshold)
                
                for doc, score in results:
                    chunk_id = doc.metadata.get('chunk_id')
                    if chunk_id not in seen_chunks:
                        seen_chunks.add(chunk_id)
                        all_results.append((doc, score, q))
                        
            except Exception as e:
                logger.error(f"Sequential search failed for query '{q}': {e}")
                continue
        
        return all_results
    
    async def _single_vector_search(
        self, 
        query: str, 
        k: int, 
        score_threshold: float
    ) -> List[Tuple[Document, float]]:
        """단일 쿼리 벡터 검색"""
        try:
            # 비동기 실행을 위해 스레드풀에서 실행
            loop = asyncio.get_event_loop()
            results = await loop.run_in_executor(
                None, 
                lambda: self.vectorstore.similarity_search_with_score(query, k=k)
            )
            
            # 점수 필터링
            filtered_results = [
                (doc, score) for doc, score in results 
                if score <= score_threshold
            ]
            
            return filtered_results
            
        except Exception as e:
            logger.error(f"Single vector search error for query '{query}': {e}")
            return []
    
    async def answer_with_confidence(
        self, 
        query: str,
        k: int = 5,
        score_threshold: float = 0.7,
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Generate answer with confidence scoring and source citations"""
        # 성능 추적 시작
        request_id = f"rag_{int(time.time() * 1000)}"
        response_time_tracker.start_tracking(request_id)
        start_time = time.time()
        
        # 모니터링 시작
        monitoring_request_id = performance_monitor.record_request_start(request_id, "rag_query")
        
        try:
            # 1. 쿼리 분석 및 최적 설정 결정
            optimal_config = get_optimal_search_config(query, context)
            logger.info(f"Adaptive config: {optimal_config}")
            
            # 2. 캐시 조회 먼저 시도
            search_params = {
                "k": k, 
                "score_threshold": score_threshold,
                "config": optimal_config
            }
            cached_result = await get_cached_query_result(query, search_params)
            if cached_result:
                # 캐시 히트 시간 기록
                elapsed_time = response_time_tracker.end_tracking(request_id)
                adaptive_feature_manager.record_performance(elapsed_time)
                logger.info(f"Cache hit for query: {query[:50]}... ({elapsed_time:.3f}s)")
                return cached_result
            
            # 3. 적응형 설정에 따른 문서 검색
            documents, search_metadata = await self.search_with_rewriting(
                query, 
                k=k, 
                score_threshold=score_threshold, 
                use_parallel_search=optimal_config["use_parallel_search"]
            )
            
            # 검색 메타데이터에 적응형 정보 추가
            search_metadata["adaptive_config"] = optimal_config
            
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
            
            # LLM 응답 캐시 조회
            llm_params = {"model": "default", "context_length": len(context)}
            cached_llm_response = await get_cached_llm_response(f"{query}||{context[:500]}", llm_params)
            
            if cached_llm_response:
                answer_text = cached_llm_response
                logger.info(f"LLM cache hit for query: {query[:30]}...")
            else:
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
                
                # LLM 응답 캐시 저장 (1시간 TTL)
                await cache_llm_response(f"{query}||{context[:500]}", answer_text, llm_params, ttl=3600)
            
            # Simple confidence scoring based on answer length and content
            confidence = "HIGH" if len(answer_text) > 100 else "MEDIUM" if len(answer_text) > 50 else "LOW"
            
            # Map sources to actual file paths
            source_files = list(set([
                doc.metadata.get('source', 'Unknown') 
                for doc in documents
            ]))
            
            # 최종 결과 구성
            final_result = {
                "answer": answer_text,
                "confidence": confidence,
                "reasoning": f"답변이 {len(documents)}개의 관련 문서를 바탕으로 생성되었습니다.",
                "sources": source_files,
                "search_metadata": search_metadata,
                "context_used": len(documents),
                "timestamp": datetime.utcnow().isoformat(),
                "processing_time": f"{time.time() - start_time:.3f}s"
            }
            
            # 성능 추적 완료
            elapsed_time = response_time_tracker.end_tracking(request_id)
            adaptive_feature_manager.record_performance(elapsed_time)
            
            # 모니터링 완료 (성공)
            performance_monitor.record_request_end(
                monitoring_request_id, 
                success=True, 
                operation="rag_query",
                additional_metrics={
                    "documents_retrieved": len(documents),
                    "processing_mode": optimal_config["complexity"],
                    "cache_used": "cache_hit" in final_result.get("reasoning", "")
                }
            )
            
            # 최종 결과에 성능 정보 추가
            final_result.update({
                "adaptive_config": optimal_config,
                "actual_processing_time": f"{elapsed_time:.3f}s",
                "estimated_vs_actual": f"estimated: {optimal_config['estimated_time']:.1f}s, actual: {elapsed_time:.1f}s",
                "performance_mode": optimal_config["processing_mode"]
            })
            
            # 결과를 캐시에 저장 (30분 TTL)
            await cache_query_result(query, final_result, search_params, ttl=1800)
            logger.info(f"Adaptive query processed: {query[:50]}... ({elapsed_time:.3f}s, mode: {optimal_config['processing_mode']})")
            
            return final_result
            
        except Exception as e:
            logger.error(f"Error generating answer: {e}")
            
            # 모니터링 완료 (실패)
            performance_monitor.record_request_end(
                monitoring_request_id, 
                success=False, 
                operation="rag_query"
            )
            
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

# Hybrid search integration
_hybrid_pipeline = None

def get_hybrid_pipeline():
    """하이브리드 검색 파이프라인 인스턴스 반환"""
    global _hybrid_pipeline
    if _hybrid_pipeline is None:
        from app.rag.hybrid_search import HybridRAGPipeline
        _hybrid_pipeline = HybridRAGPipeline(rag_pipeline)
    return _hybrid_pipeline