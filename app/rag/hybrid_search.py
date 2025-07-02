"""
하이브리드 검색 시스템
벡터 검색 + 키워드 검색 + BM25 점수를 결합하여 더 정확한 검색 결과 제공
"""

import re
import math
import time
import asyncio
from typing import List, Dict, Any, Optional, Tuple
from collections import Counter, defaultdict
from pathlib import Path
from datetime import datetime

from langchain.schema import Document
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings

from app.core.logging import logger
from app.rag.reranker import get_global_reranker
from app.core.cache import get_cached_query_result, cache_query_result
from app.core.async_utils import AsyncBatchProcessor
from app.core.adaptive_features import get_optimal_search_config
from app.core.model_optimization import (
    get_optimized_reranker_model, adaptive_selector
)


class BM25:
    """BM25 알고리즘 구현"""
    
    def __init__(self, corpus: List[str], k1: float = 1.2, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.corpus = corpus
        self.doc_freqs = []
        self.idf = {}
        self.doc_len = []
        self.avgdl = 0
        
        self._initialize()
    
    def _initialize(self):
        """BM25 초기화"""
        nd = len(self.corpus)
        
        # 각 문서의 단어 빈도와 길이 계산
        for document in self.corpus:
            words = self._tokenize(document)
            self.doc_len.append(len(words))
            
            # 단어 빈도 계산
            freq = Counter(words)
            self.doc_freqs.append(freq)
            
            # IDF 계산을 위한 문서 빈도 수집
            for word in set(words):
                if word not in self.idf:
                    self.idf[word] = 0
                self.idf[word] += 1
        
        # 평균 문서 길이
        self.avgdl = sum(self.doc_len) / len(self.doc_len)
        
        # IDF 계산
        for word, freq in self.idf.items():
            self.idf[word] = math.log((nd - freq + 0.5) / (freq + 0.5))
    
    def _tokenize(self, text: str) -> List[str]:
        """텍스트 토큰화 (한글/영문 지원)"""
        # 한글, 영문, 숫자만 추출
        text = re.sub(r'[^\w\s가-힣]', ' ', text.lower())
        # 공백으로 분리
        tokens = text.split()
        # 길이 1 이하 토큰 제거
        return [token for token in tokens if len(token) > 1]
    
    def get_scores(self, query: str) -> List[float]:
        """쿼리에 대한 모든 문서의 BM25 점수 계산"""
        query_words = self._tokenize(query)
        scores = []
        
        for doc_idx, doc_freqs in enumerate(self.doc_freqs):
            score = 0
            doc_len = self.doc_len[doc_idx]
            
            for word in query_words:
                if word in doc_freqs:
                    # BM25 공식
                    tf = doc_freqs[word]
                    idf = self.idf.get(word, 0)
                    
                    score += idf * (tf * (self.k1 + 1)) / (
                        tf + self.k1 * (1 - self.b + self.b * doc_len / self.avgdl)
                    )
            
            scores.append(score)
        
        return scores


class HybridSearchEngine:
    """하이브리드 검색 엔진 (벡터 + 키워드 + BM25)"""
    
    def __init__(
        self,
        vectorstore: Chroma,
        embeddings: OpenAIEmbeddings,
        vector_weight: float = 0.5,
        keyword_weight: float = 0.3,
        bm25_weight: float = 0.2
    ):
        self.vectorstore = vectorstore
        self.embeddings = embeddings
        self.vector_weight = vector_weight
        self.keyword_weight = keyword_weight
        self.bm25_weight = bm25_weight
        
        # 문서 데이터 준비
        self.documents = []
        self.doc_texts = []
        self.bm25 = None
        
        self._initialize_corpus()
    
    def _initialize_corpus(self):
        """BM25를 위한 코퍼스 초기화"""
        try:
            # ChromaDB에서 모든 문서 가져오기
            collection = self.vectorstore._collection
            results = collection.get()
            
            if results and 'documents' in results:
                self.doc_texts = results['documents']
                
                # Document 객체 생성
                metadatas = results.get('metadatas', [])
                for i, text in enumerate(self.doc_texts):
                    metadata = metadatas[i] if i < len(metadatas) else {}
                    doc = Document(page_content=text, metadata=metadata)
                    self.documents.append(doc)
                
                # BM25 초기화
                self.bm25 = BM25(self.doc_texts)
                logger.info(f"Hybrid search initialized with {len(self.doc_texts)} documents")
            else:
                logger.warning("No documents found in vector store")
                
        except Exception as e:
            logger.error(f"Error initializing hybrid search corpus: {e}")
            self.doc_texts = []
            self.documents = []
    
    def _normalize_scores(self, scores: List[float]) -> List[float]:
        """점수 정규화 (0-1 범위)"""
        if not scores or all(score == 0 for score in scores):
            return [0.0] * len(scores)
        
        min_score = min(scores)
        max_score = max(scores)
        
        if max_score == min_score:
            return [1.0] * len(scores)
        
        return [(score - min_score) / (max_score - min_score) for score in scores]
    
    def _vector_search(self, query: str, k: int = 20) -> List[Tuple[Document, float]]:
        """벡터 검색"""
        try:
            results = self.vectorstore.similarity_search_with_score(query, k=k)
            # ChromaDB는 거리 점수 (낮을수록 좋음)이므로 역변환
            return [(doc, 1.0 / (1.0 + score)) for doc, score in results]
        except Exception as e:
            logger.error(f"Vector search error: {e}")
            return []
    
    def _keyword_search(self, query: str) -> List[float]:
        """키워드 검색 (단순 TF-IDF 기반)"""
        query_words = set(re.sub(r'[^\w\s가-힣]', ' ', query.lower()).split())
        scores = []
        
        for text in self.doc_texts:
            text_words = set(re.sub(r'[^\w\s가-힣]', ' ', text.lower()).split())
            
            # 교집합 비율 계산
            intersection = len(query_words & text_words)
            union = len(query_words | text_words)
            
            jaccard_score = intersection / union if union > 0 else 0
            scores.append(jaccard_score)
        
        return scores
    
    def _bm25_search(self, query: str) -> List[float]:
        """BM25 검색"""
        if self.bm25 is None:
            return [0.0] * len(self.doc_texts)
        
        try:
            return self.bm25.get_scores(query)
        except Exception as e:
            logger.error(f"BM25 search error: {e}")
            return [0.0] * len(self.doc_texts)
    
    def search(
        self,
        query: str,
        k: int = 5,
        score_threshold: float = 0.1
    ) -> List[Tuple[Document, float, Dict[str, float]]]:
        """하이브리드 검색 실행"""
        try:
            # 1. 벡터 검색
            vector_results = self._vector_search(query, k=min(50, len(self.documents)))
            
            # 2. 키워드 검색
            keyword_scores = self._keyword_search(query)
            keyword_scores_norm = self._normalize_scores(keyword_scores)
            
            # 3. BM25 검색
            bm25_scores = self._bm25_search(query)
            bm25_scores_norm = self._normalize_scores(bm25_scores)
            
            # 4. 점수 결합
            combined_results = []
            
            # 벡터 검색 결과를 기준으로 결합
            for doc, vector_score in vector_results:
                # 문서 인덱스 찾기
                doc_idx = None
                for i, corpus_doc in enumerate(self.documents):
                    if (corpus_doc.page_content == doc.page_content and 
                        corpus_doc.metadata.get('chunk_id') == doc.metadata.get('chunk_id')):
                        doc_idx = i
                        break
                
                if doc_idx is not None:
                    # 개별 점수들
                    keyword_score = keyword_scores_norm[doc_idx]
                    bm25_score = bm25_scores_norm[doc_idx]
                    
                    # 가중 평균 계산
                    combined_score = (
                        self.vector_weight * vector_score +
                        self.keyword_weight * keyword_score +
                        self.bm25_weight * bm25_score
                    )
                    
                    # 점수 세부 정보
                    score_details = {
                        'vector_score': vector_score,
                        'keyword_score': keyword_score,
                        'bm25_score': bm25_score,
                        'combined_score': combined_score
                    }
                    
                    if combined_score >= score_threshold:
                        combined_results.append((doc, combined_score, score_details))
            
            # 결합 점수로 정렬
            combined_results.sort(key=lambda x: x[1], reverse=True)
            
            # 상위 k개 반환
            return combined_results[:k]
            
        except Exception as e:
            logger.error(f"Hybrid search error: {e}")
            # 폴백: 벡터 검색만 사용
            vector_results = self._vector_search(query, k)
            return [(doc, score, {'vector_score': score, 'keyword_score': 0, 'bm25_score': 0, 'combined_score': score}) 
                   for doc, score in vector_results]
    
    def update_corpus(self):
        """코퍼스 업데이트 (새 문서 추가 시)"""
        self._initialize_corpus()
        logger.info("Hybrid search corpus updated")


class HybridRAGPipeline:
    """하이브리드 검색 기반 RAG 파이프라인"""
    
    def __init__(self, enhanced_rag_pipeline):
        self.base_pipeline = enhanced_rag_pipeline
        self.hybrid_engine = HybridSearchEngine(
            vectorstore=enhanced_rag_pipeline.vectorstore,
            embeddings=enhanced_rag_pipeline.embeddings,
            vector_weight=0.5,  # 벡터 검색 가중치
            keyword_weight=0.3,  # 키워드 검색 가중치
            bm25_weight=0.2     # BM25 검색 가중치
        )
        self.batch_processor = AsyncBatchProcessor(batch_size=5, max_concurrent=3)
    
    async def search_with_hybrid(
        self,
        query: str,
        k: int = 5,
        score_threshold: float = 0.1,
        use_query_rewriting: bool = False,  # 기본적으로 쿼리 재작성 비활성화
        use_reranking: bool = True  # 리랭킹 사용 여부
    ) -> Tuple[List[Document], Dict[str, Any]]:
        """하이브리드 검색 실행"""
        try:
            # 조건부 쿼리 재작성 (성능 최적화)
            if use_query_rewriting:
                queries = await self.base_pipeline.rewrite_query(query)
            else:
                queries = [query]  # 단일 쿼리만 사용
            
            # 하이브리드 검색 실행 (병렬 처리)
            if len(queries) > 1:
                all_results = await self._parallel_hybrid_search(queries, k, score_threshold)
            else:
                # 단일 쿼리는 직접 처리
                results = self.hybrid_engine.search(queries[0], k=k*2, score_threshold=score_threshold)
                all_results = [(doc, score, score_details, queries[0]) for doc, score, score_details in results]
            
            # 점수로 정렬
            all_results.sort(key=lambda x: x[1], reverse=True)
            
            # 리랭킹 적용 (선택적)
            if use_reranking and all_results:
                logger.info(f"Applying reranking to {len(all_results)} results")
                
                # 리랭킹을 위한 데이터 준비
                docs_for_rerank = [result[0] for result in all_results[:k*2]]  # 더 많은 후보로 리랭킹
                scores_for_rerank = [result[1] for result in all_results[:k*2]]
                
                # 최적화된 리랭킹 실행
                try:
                    reranker = get_optimized_reranker_model()
                    logger.info("Using optimized reranker model")
                except Exception:
                    reranker = get_global_reranker()
                    logger.info("Using fallback reranker model")
                
                # 성능 추적 시작
                rerank_start = time.time()
                rerank_results = reranker.advanced_rerank(
                    query, docs_for_rerank, scores_for_rerank, {"original_results": all_results}
                )
                rerank_time = time.time() - rerank_start
                
                # 성능 기록
                adaptive_selector.record_performance(
                    "reranker", "reranking", rerank_time, 
                    quality_score=len(rerank_results)/max(len(docs_for_rerank), 1)
                )
                
                # 리랭킹 결과를 top_results 형식으로 변환
                top_results = []
                for i, rerank_result in enumerate(rerank_results[:k]):
                    # 원본 쿼리 찾기
                    original_query = query  # 단순화 (실제로는 매칭 로직 필요)
                    top_results.append((
                        rerank_result.document,
                        rerank_result.combined_score,
                        {
                            'vector_score': 0,  # 리랭킹 후에는 개별 점수 정보가 변경됨
                            'keyword_score': 0,
                            'bm25_score': 0,
                            'combined_score': rerank_result.original_score,
                            'rerank_score': rerank_result.rerank_score,
                            'final_score': rerank_result.combined_score,
                            'rank_change': rerank_result.rank_change
                        },
                        original_query
                    ))
                
                search_type = "hybrid_with_reranking"
            else:
                # 리랭킹 없이 상위 k개 선택
                top_results = all_results[:k]
                search_type = "hybrid"
            
            # 메타데이터 생성
            documents = [doc for doc, _, _, _ in top_results]
            search_metadata = {
                "search_type": search_type,
                "original_query": query,
                "rewritten_queries": queries,
                "total_results": len(all_results),
                "returned_results": len(documents),
                "score_threshold": score_threshold,
                "result_details": [
                    {
                        "chunk_id": doc.metadata.get('chunk_id'),
                        "source": doc.metadata.get('source'),
                        "combined_score": score,
                        "score_breakdown": score_details,
                        "matched_query": matched_query
                    }
                    for doc, score, score_details, matched_query in top_results
                ]
            }
            
            logger.info(f"Hybrid search completed: {len(documents)} results")
            return documents, search_metadata
            
        except Exception as e:
            logger.error(f"Hybrid search error: {e}")
            # 폴백: 기존 벡터 검색 사용
            return await self.base_pipeline.search_with_rewriting(query, k, score_threshold)
    
    async def answer_with_hybrid_search(
        self,
        query: str,
        k: int = 5,
        score_threshold: float = 0.1,
        use_query_rewriting: bool = None,  # None이면 적응형 결정
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """하이브리드 검색을 사용한 답변 생성"""
        start_time = time.time()
        
        try:
            # 1. 적응형 설정 결정
            if use_query_rewriting is None:
                optimal_config = get_optimal_search_config(query, context)
                use_query_rewriting = optimal_config["use_query_rewriting"]
                use_reranking = optimal_config["use_reranking"]
                logger.info(f"Adaptive hybrid config: rewrite={use_query_rewriting}, rerank={use_reranking}")
            else:
                optimal_config = {"use_reranking": True}  # 기본값
                use_reranking = True
            
            # 2. 캐시 조회 먼저 시도
            search_params = {
                "k": k, 
                "score_threshold": score_threshold, 
                "use_query_rewriting": use_query_rewriting,
                "use_reranking": use_reranking,
                "search_type": "hybrid"
            }
            cached_result = await get_cached_query_result(query, search_params)
            if cached_result:
                logger.info(f"Hybrid cache hit: {query[:50]}... ({time.time() - start_time:.3f}s)")
                return cached_result
            
            # 3. 하이브리드 검색 (적응형 리랭킹 포함)
            documents, search_metadata = await self.search_with_hybrid(
                query, 
                k=k, 
                score_threshold=score_threshold, 
                use_query_rewriting=use_query_rewriting,
                use_reranking=use_reranking
            )
            
            if not documents:
                return {
                    "answer": "죄송합니다. 관련된 정보를 찾을 수 없습니다.",
                    "confidence": "LOW",
                    "reasoning": "No relevant documents found with hybrid search",
                    "sources": [],
                    "search_metadata": search_metadata
                }
            
            # 컨텍스트 준비
            context_parts = []
            for i, doc in enumerate(documents):
                source = doc.metadata.get('source', 'Unknown')
                context_parts.append(f"[Source {i+1}: {Path(source).name}]\\n{doc.page_content}")
            
            context = "\\n\\n".join(context_parts)
            
            # LLM으로 답변 생성
            from langchain.chains import LLMChain
            from langchain.prompts import PromptTemplate
            
            from app.llm.router import llm_router
            llm = await llm_router.get_langchain_model()
            
            prompt = PromptTemplate(
                template="""다음 문맥을 바탕으로 질문에 정확하고 자세하게 답변해주세요.
여러 출처의 정보를 종합하여 일관성 있는 답변을 제공하세요.

문맥:
{context}

질문: {question}

답변:""",
                input_variables=["context", "question"]
            )
            
            chain = LLMChain(llm=llm, prompt=prompt)
            result = await chain.ainvoke({"context": context, "question": query})
            
            answer_text = result["text"].strip()
            
            # 고도화된 신뢰도 평가
            confidence = self._calculate_confidence(answer_text, documents, search_metadata)
            
            # 출처 정보
            source_files = list(set([
                doc.metadata.get('source', 'Unknown') 
                for doc in documents
            ]))
            
            # 최종 결과 구성
            final_result = {
                "answer": answer_text,
                "confidence": confidence,
                "reasoning": f"하이브리드 검색으로 {len(documents)}개 문서에서 정보를 종합했습니다.",
                "sources": source_files,
                "search_metadata": search_metadata,
                "context_used": len(documents),
                "timestamp": datetime.utcnow().isoformat(),
                "search_type": "hybrid",
                "processing_time": f"{time.time() - start_time:.3f}s"
            }
            
            # 결과를 캐시에 저장 (30분 TTL)
            await cache_query_result(query, final_result, search_params, ttl=1800)
            logger.info(f"Hybrid query processed and cached: {query[:50]}... ({time.time() - start_time:.3f}s)")
            
            return final_result
            
        except Exception as e:
            logger.error(f"Hybrid answer generation error: {e}")
            return {
                "answer": f"오류가 발생했습니다: {str(e)}",
                "confidence": "LOW",
                "reasoning": "Error occurred during hybrid search processing",
                "sources": [],
                "error": str(e)
            }
    
    def _calculate_confidence(
        self, 
        answer: str, 
        documents: List[Document], 
        search_metadata: Dict[str, Any]
    ) -> str:
        """고도화된 신뢰도 계산"""
        try:
            confidence_score = 0.0
            
            # 1. 답변 길이 (최대 30점)
            length_score = min(len(answer) / 200 * 30, 30)
            confidence_score += length_score
            
            # 2. 검색 점수 평균 (최대 25점)
            if search_metadata.get('result_details'):
                avg_score = sum(
                    detail.get('combined_score', 0) 
                    for detail in search_metadata['result_details']
                ) / len(search_metadata['result_details'])
                search_score = min(avg_score * 25, 25)
                confidence_score += search_score
            
            # 3. 문서 수 (최대 20점)
            doc_score = min(len(documents) / 5 * 20, 20)
            confidence_score += doc_score
            
            # 4. 답변 품질 휴리스틱 (최대 25점)
            quality_indicators = [
                "따라서", "결론적으로", "요약하면",  # 결론 표현
                "첫째", "둘째", "셋째",  # 구조화
                "예를 들어", "구체적으로",  # 예시
                "참고", "출처", "문서"  # 참조
            ]
            
            quality_score = 0
            for indicator in quality_indicators:
                if indicator in answer:
                    quality_score += 3
            quality_score = min(quality_score, 25)
            confidence_score += quality_score
            
            # 최종 신뢰도 등급
            if confidence_score >= 75:
                return "HIGH"
            elif confidence_score >= 50:
                return "MEDIUM"
            else:
                return "LOW"
                
        except Exception:
            return "MEDIUM"  # 오류 시 기본값
    
    async def _parallel_hybrid_search(
        self,
        queries: List[str],
        k: int,
        score_threshold: float
    ) -> List[Tuple[Document, float, Dict[str, float], str]]:
        """병렬 하이브리드 검색 실행"""
        try:
            # 각 쿼리에 대한 검색 태스크 생성
            search_tasks = []
            for query in queries:
                task = self._single_hybrid_search(query, k*2, score_threshold)
                search_tasks.append(task)
            
            # 병렬 실행
            start_time = time.time()
            results_per_query = await asyncio.gather(*search_tasks, return_exceptions=True)
            elapsed_time = time.time() - start_time
            
            logger.info(f"Parallel hybrid search completed in {elapsed_time:.3f}s for {len(queries)} queries")
            
            # 결과 통합 및 중복 제거
            all_results = []
            seen_chunks = set()
            
            for i, results in enumerate(results_per_query):
                if isinstance(results, Exception):
                    logger.error(f"Hybrid search failed for query '{queries[i]}': {results}")
                    continue
                
                for doc, score, score_details in results:
                    chunk_id = doc.metadata.get('chunk_id')
                    if chunk_id not in seen_chunks:
                        seen_chunks.add(chunk_id)
                        all_results.append((doc, score, score_details, queries[i]))
            
            return all_results
            
        except Exception as e:
            logger.error(f"Parallel hybrid search error: {e}")
            # 폴백: 순차 검색
            all_results = []
            seen_chunks = set()
            
            for query in queries:
                try:
                    results = self.hybrid_engine.search(query, k=k*2, score_threshold=score_threshold)
                    for doc, score, score_details in results:
                        chunk_id = doc.metadata.get('chunk_id')
                        if chunk_id not in seen_chunks:
                            seen_chunks.add(chunk_id)
                            all_results.append((doc, score, score_details, query))
                except Exception as query_error:
                    logger.error(f"Fallback search failed for query '{query}': {query_error}")
            
            return all_results
    
    async def _single_hybrid_search(
        self,
        query: str,
        k: int,
        score_threshold: float
    ) -> List[Tuple[Document, float, Dict[str, float]]]:
        """단일 쿼리 하이브리드 검색"""
        try:
            # 하이브리드 검색은 CPU 집약적이므로 스레드풀에서 실행
            loop = asyncio.get_event_loop()
            results = await loop.run_in_executor(
                None,
                lambda: self.hybrid_engine.search(query, k=k, score_threshold=score_threshold)
            )
            return results
            
        except Exception as e:
            logger.error(f"Single hybrid search error for query '{query}': {e}")
            return []