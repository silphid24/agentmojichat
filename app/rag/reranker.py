"""
리랭킹 시스템 (Re-ranking System)
교차 인코더를 사용하여 검색 결과의 관련도를 재평가하고 순위를 재정렬
"""

import time
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

from langchain.schema import Document

from app.core.logging import logger


@dataclass
class RerankResult:
    """리랭킹 결과 데이터 클래스"""

    document: Document
    original_score: float
    rerank_score: float
    combined_score: float
    rank_change: int  # 순위 변화 (+는 상승, -는 하락)


class CrossEncoderReranker:
    """교차 인코더 기반 리랭킹 시스템"""

    def __init__(
        self,
        model_name: str = "ms-marco-MiniLM-L-6-v2",
        use_local_model: bool = True,
        rerank_weight: float = 0.7,
        original_weight: float = 0.3,
    ):
        self.model_name = model_name
        self.use_local_model = use_local_model
        self.rerank_weight = rerank_weight
        self.original_weight = original_weight
        self.model = None
        self.tokenizer = None

        self._initialize_model()

    def _initialize_model(self):
        """모델 초기화"""
        try:
            if self.use_local_model:
                # sentence-transformers 사용 (로컬 모델)
                try:
                    from sentence_transformers import CrossEncoder

                    # 한국어와 영어 모두 지원하는 모델 선택
                    model_options = [
                        "ms-marco-MiniLM-L-6-v2",  # 영어 특화
                        "cross-encoder/ms-marco-MiniLM-L-6-v2",  # 공식 모델
                        "sentence-transformers/ms-marco-MiniLM-L-6-v2",  # 대안
                    ]

                    model_loaded = False
                    for model_option in model_options:
                        try:
                            self.model = CrossEncoder(model_option)
                            self.model_name = model_option
                            model_loaded = True
                            logger.info(f"Cross-encoder model loaded: {model_option}")
                            break
                        except Exception as e:
                            logger.warning(f"Failed to load {model_option}: {e}")
                            continue

                    if not model_loaded:
                        logger.warning(
                            "All cross-encoder models failed, using fallback similarity"
                        )
                        self.model = None
                        self.use_local_model = False

                except ImportError:
                    logger.warning(
                        "sentence-transformers not available, using fallback similarity"
                    )
                    self.model = None
                    self.use_local_model = False

            if not self.use_local_model:
                # 폴백: 간단한 텍스트 유사도 계산
                self._initialize_fallback()

        except Exception as e:
            logger.error(f"Error initializing reranker: {e}")
            self._initialize_fallback()

    def _initialize_fallback(self):
        """폴백 시스템 초기화 (단순 텍스트 유사도)"""
        self.model = None
        self.use_local_model = False
        logger.info("Using fallback text similarity for reranking")

    def _calculate_text_similarity(self, query: str, text: str) -> float:
        """폴백용 간단한 텍스트 유사도 계산"""
        try:
            # 단어 기반 jaccard 유사도
            query_words = set(query.lower().split())
            text_words = set(text.lower().split())

            if not query_words or not text_words:
                return 0.0

            intersection = len(query_words & text_words)
            union = len(query_words | text_words)

            jaccard = intersection / union if union > 0 else 0.0

            # 텍스트 길이 고려 (너무 짧거나 긴 텍스트 페널티)
            text_len = len(text.split())
            length_factor = 1.0
            if text_len < 10:
                length_factor = 0.8
            elif text_len > 500:
                length_factor = 0.9

            # 쿼리 단어가 텍스트에 연속으로 나타나는지 확인 (구문 매칭)
            phrase_bonus = 0.0
            if query.lower() in text.lower():
                phrase_bonus = 0.2

            return jaccard * length_factor + phrase_bonus

        except Exception:
            return 0.0

    def rerank(
        self,
        query: str,
        documents: List[Document],
        original_scores: List[float],
        top_k: Optional[int] = None,
    ) -> List[RerankResult]:
        """검색 결과 리랭킹"""
        if not documents:
            return []

        try:
            start_time = time.time()

            # 리랭킹 점수 계산
            rerank_scores = self._compute_rerank_scores(query, documents)

            # 결과 생성
            results = []
            for i, (doc, orig_score, rerank_score) in enumerate(
                zip(documents, original_scores, rerank_scores)
            ):
                # 가중 평균으로 최종 점수 계산
                combined_score = (
                    self.original_weight * orig_score
                    + self.rerank_weight * rerank_score
                )

                result = RerankResult(
                    document=doc,
                    original_score=orig_score,
                    rerank_score=rerank_score,
                    combined_score=combined_score,
                    rank_change=0,  # 나중에 계산
                )
                results.append(result)

            # 원본 순위 저장
            original_ranks = {id(result): i for i, result in enumerate(results)}

            # 최종 점수로 정렬
            results.sort(key=lambda x: x.combined_score, reverse=True)

            # 순위 변화 계산
            for new_rank, result in enumerate(results):
                original_rank = original_ranks[id(result)]
                result.rank_change = (
                    original_rank - new_rank
                )  # 양수면 상승, 음수면 하락

            # 상위 k개만 반환
            if top_k:
                results = results[:top_k]

            elapsed_time = time.time() - start_time
            logger.info(
                f"Reranking completed in {elapsed_time:.3f}s for {len(documents)} documents"
            )

            return results

        except Exception as e:
            logger.error(f"Error during reranking: {e}")
            # 폴백: 원본 순서 유지
            return [
                RerankResult(
                    document=doc,
                    original_score=score,
                    rerank_score=score,
                    combined_score=score,
                    rank_change=0,
                )
                for doc, score in zip(documents, original_scores)
            ]

    def _compute_rerank_scores(
        self, query: str, documents: List[Document]
    ) -> List[float]:
        """리랭킹 점수 계산"""
        if self.model and self.use_local_model:
            return self._compute_cross_encoder_scores(query, documents)
        else:
            return self._compute_fallback_scores(query, documents)

    def _compute_cross_encoder_scores(
        self, query: str, documents: List[Document]
    ) -> List[float]:
        """교차 인코더를 사용한 점수 계산"""
        try:
            # 쿼리-문서 쌍 생성
            pairs = [(query, doc.page_content) for doc in documents]

            # 배치 처리로 점수 계산
            scores = self.model.predict(pairs)

            # 점수 정규화 (0-1 범위)
            if len(scores) > 1:
                min_score = min(scores)
                max_score = max(scores)
                if max_score > min_score:
                    scores = [
                        (score - min_score) / (max_score - min_score)
                        for score in scores
                    ]
                else:
                    scores = [1.0] * len(scores)
            else:
                scores = [1.0] * len(scores)

            return scores.tolist() if hasattr(scores, "tolist") else list(scores)

        except Exception as e:
            logger.error(f"Cross-encoder scoring failed: {e}")
            return self._compute_fallback_scores(query, documents)

    def _compute_fallback_scores(
        self, query: str, documents: List[Document]
    ) -> List[float]:
        """폴백 점수 계산 (텍스트 유사도)"""
        scores = []
        for doc in documents:
            score = self._calculate_text_similarity(query, doc.page_content)
            scores.append(score)

        # 정규화
        if scores and max(scores) > 0:
            max_score = max(scores)
            scores = [score / max_score for score in scores]

        return scores


class AdvancedReranker:
    """고급 리랭킹 시스템 (다중 신호 결합)"""

    def __init__(self):
        self.cross_encoder = CrossEncoderReranker()

    def advanced_rerank(
        self,
        query: str,
        documents: List[Document],
        original_scores: List[float],
        search_metadata: Dict[str, Any] = None,
    ) -> List[RerankResult]:
        """고급 리랭킹 (다중 신호 결합)"""
        try:
            # 1. 교차 인코더 리랭킹
            base_results = self.cross_encoder.rerank(query, documents, original_scores)

            # 2. 추가 신호들로 점수 조정
            enhanced_results = []

            for result in base_results:
                enhanced_score = result.combined_score

                # 문서 길이 신호
                doc_length = len(result.document.page_content)
                if 100 <= doc_length <= 2000:  # 적절한 길이
                    enhanced_score *= 1.1
                elif doc_length < 50:  # 너무 짧음
                    enhanced_score *= 0.8
                elif doc_length > 3000:  # 너무 김
                    enhanced_score *= 0.9

                # 소스 파일 타입 신호
                source = result.document.metadata.get("source", "")
                if source.endswith(".docx"):
                    enhanced_score *= 1.05  # DOCX 문서 약간 선호
                elif source.endswith(".md"):
                    enhanced_score *= 1.02  # 마크다운 문서 약간 선호

                # 메타데이터 품질 신호
                metadata = result.document.metadata
                if metadata.get("chunk_index", 0) == 0:  # 문서의 첫 번째 청크
                    enhanced_score *= 1.03

                # 키워드 밀도 신호
                query_words = set(query.lower().split())
                doc_words = result.document.page_content.lower().split()
                keyword_density = sum(
                    1 for word in doc_words if word in query_words
                ) / max(len(doc_words), 1)

                if keyword_density > 0.05:  # 키워드 밀도가 높음
                    enhanced_score *= 1 + keyword_density * 0.5

                # 새로운 결과 생성
                enhanced_result = RerankResult(
                    document=result.document,
                    original_score=result.original_score,
                    rerank_score=result.rerank_score,
                    combined_score=enhanced_score,
                    rank_change=result.rank_change,
                )
                enhanced_results.append(enhanced_result)

            # 최종 정렬
            enhanced_results.sort(key=lambda x: x.combined_score, reverse=True)

            # 순위 변화 재계산
            original_ranks = {id(doc): i for i, doc in enumerate(documents)}
            for new_rank, result in enumerate(enhanced_results):
                doc_id = id(result.document)
                if doc_id in original_ranks:
                    original_rank = original_ranks[doc_id]
                    result.rank_change = original_rank - new_rank

            logger.info(
                f"Advanced reranking completed for {len(enhanced_results)} documents"
            )
            return enhanced_results

        except Exception as e:
            logger.error(f"Advanced reranking failed: {e}")
            # 폴백: 기본 리랭킹 사용
            return self.cross_encoder.rerank(query, documents, original_scores)


def get_reranker(use_advanced: bool = True) -> AdvancedReranker:
    """리랭커 인스턴스 반환"""
    if use_advanced:
        return AdvancedReranker()
    else:
        return CrossEncoderReranker()


# 전역 리랭커 인스턴스
_reranker_instance = None


def get_global_reranker() -> AdvancedReranker:
    """전역 리랭커 인스턴스 반환 (싱글톤)"""
    global _reranker_instance
    if _reranker_instance is None:
        _reranker_instance = AdvancedReranker()
    return _reranker_instance
