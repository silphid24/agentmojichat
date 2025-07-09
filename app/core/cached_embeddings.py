"""
캐시 지원 임베딩 클래스
OpenAI 임베딩 API 호출을 캐시하여 성능 향상
"""

import asyncio
from typing import List
from langchain_openai import OpenAIEmbeddings
from app.core.cache import get_cached_embedding, cache_embedding
from app.core.logging import logger


class CachedOpenAIEmbeddings(OpenAIEmbeddings):
    """캐시 지원 OpenAI 임베딩 클래스"""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # 캐시 통계를 별도 속성으로 관리 (Pydantic 필드와 충돌 방지)
        self.__dict__["_cache_stats"] = {"hits": 0, "misses": 0, "batch_size": 100}
    
    def _get_cache_key(self, text: str) -> str:
        """캐시 키 생성"""
        import hashlib
        return f"embedding:{hashlib.md5(text.encode()).hexdigest()}"

    async def aembed_documents(self, texts: List[str]) -> List[List[float]]:
        """문서 임베딩 생성 (캐시 지원)"""
        results = []
        uncached_texts = []
        uncached_indices = []

        # 캐시 조회
        for i, text in enumerate(texts):
            cached_embedding = await get_cached_embedding(text)
            if cached_embedding:
                results.append(cached_embedding)
                self._cache_stats["hits"] += 1
            else:
                results.append(None)  # 임시 플레이스홀더
                uncached_texts.append(text)
                uncached_indices.append(i)
                self._cache_stats["misses"] += 1

        # 캐시되지 않은 텍스트들을 배치로 처리
        if uncached_texts:
            logger.info(
                f"Generating embeddings for {len(uncached_texts)} uncached texts"
            )

            # 배치로 나누어 처리
            batch_size = self._cache_stats["batch_size"]
            for batch_start in range(0, len(uncached_texts), batch_size):
                batch_end = min(batch_start + batch_size, len(uncached_texts))
                batch_texts = uncached_texts[batch_start:batch_end]
                batch_indices = uncached_indices[batch_start:batch_end]

                # 실제 임베딩 생성 (재시도 로직 추가)
                max_retries = 3
                for attempt in range(max_retries):
                    try:
                        batch_embeddings = await super().aembed_documents(batch_texts)
                        break
                    except Exception as e:
                        if attempt == max_retries - 1:
                            logger.error(f"Failed to generate embeddings after {max_retries} attempts: {e}")
                            raise
                        else:
                            logger.warning(f"Embedding attempt {attempt + 1} failed: {e}. Retrying...")
                            await asyncio.sleep(2 ** attempt)  # Exponential backoff

                # 결과 저장 및 캐시
                for j, (embedding, original_idx) in enumerate(
                    zip(batch_embeddings, batch_indices)
                ):
                    results[original_idx] = embedding
                    # 캐시에 저장 (24시간 TTL)
                    await cache_embedding(batch_texts[j], embedding, ttl=86400)

        hits = self._cache_stats["hits"]
        misses = self._cache_stats["misses"]
        logger.info(f"Embedding cache stats - Hits: {hits}, Misses: {misses}")
        return results

    async def aembed_query(self, text: str) -> List[float]:
        """쿼리 임베딩 생성 (캐시 지원)"""
        # 캐시 조회
        cached_embedding = await get_cached_embedding(text)
        if cached_embedding:
            self._cache_stats["hits"] += 1
            return cached_embedding

        # 캐시 미스 - 새로 생성
        self._cache_stats["misses"] += 1
        embedding = await super().aembed_query(text)

        # 캐시에 저장
        await cache_embedding(text, embedding, ttl=86400)

        return embedding

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """동기식 문서 임베딩 (기존 메서드 사용)"""
        # 이미 실행 중인 이벤트 루프가 있는 경우 기본 메서드 사용
        try:
            loop = asyncio.get_running_loop()
            # 이벤트 루프가 실행 중이면 기본 동기 메서드 사용
            return super().embed_documents(texts)
        except RuntimeError:
            # 이벤트 루프가 없으면 비동기 메서드 실행
            return asyncio.run(self.aembed_documents(texts))

    def embed_query(self, text: str) -> List[float]:
        """동기식 쿼리 임베딩 (기존 메서드 사용)"""
        # 이미 실행 중인 이벤트 루프가 있는 경우 기본 메서드 사용
        try:
            loop = asyncio.get_running_loop()
            # 이벤트 루프가 실행 중이면 기본 동기 메서드 사용
            return super().embed_query(text)
        except RuntimeError:
            # 이벤트 루프가 없으면 비동기 메서드 실행
            return asyncio.run(self.aembed_query(text))

    def get_cache_stats(self) -> dict:
        """캐시 통계 반환"""
        hits = self._cache_stats["hits"]
        misses = self._cache_stats["misses"]
        total = hits + misses
        hit_rate = hits / total if total > 0 else 0.0

        return {
            "cache_hits": hits,
            "cache_misses": misses,
            "hit_rate": hit_rate,
            "total_requests": total,
        }
