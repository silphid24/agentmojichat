"""
RAG 성능 최적화를 위한 캐싱 시스템
쿼리 결과, 임베딩, LLM 응답을 캐시하여 응답 속도 대폭 향상
"""

import json
import hashlib
import time
from typing import Any, Dict, List, Optional, Union
from datetime import datetime, timedelta
import asyncio
import pickle

from app.core.logging import logger


# 지연 임포트로 순환 참조 방지
def get_performance_monitor():
    try:
        from app.core.monitoring import performance_monitor
        return performance_monitor
    except ImportError:
        return None


class MemoryCache:
    """메모리 기반 캐시 (Redis 없을 때 폴백)"""
    
    def __init__(self, max_size: int = 1000, default_ttl: int = 3600):
        self.cache: Dict[str, Dict[str, Any]] = {}
        self.max_size = max_size
        self.default_ttl = default_ttl
        self._access_times: Dict[str, float] = {}
    
    def _is_expired(self, key: str) -> bool:
        """캐시 항목 만료 확인"""
        if key not in self.cache:
            return True
        
        expires_at = self.cache[key].get('expires_at')
        if expires_at and time.time() > expires_at:
            self._remove(key)
            return True
        return False
    
    def _remove(self, key: str) -> None:
        """캐시 항목 제거"""
        self.cache.pop(key, None)
        self._access_times.pop(key, None)
    
    def _evict_if_needed(self) -> None:
        """LRU 방식으로 캐시 크기 관리"""
        if len(self.cache) >= self.max_size:
            # 가장 오래된 항목 제거
            oldest_key = min(self._access_times.keys(), key=lambda k: self._access_times[k])
            self._remove(oldest_key)
    
    def get(self, key: str) -> Optional[Any]:
        """캐시에서 값 조회"""
        monitor = get_performance_monitor()
        
        if self._is_expired(key):
            if monitor:
                monitor.record_cache_miss("memory")
            return None
        
        self._access_times[key] = time.time()
        if monitor:
            monitor.record_cache_hit("memory")
        return self.cache[key]['value']
    
    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        """캐시에 값 저장"""
        self._evict_if_needed()
        
        ttl = ttl or self.default_ttl
        expires_at = time.time() + ttl if ttl > 0 else None
        
        self.cache[key] = {
            'value': value,
            'expires_at': expires_at,
            'created_at': time.time()
        }
        self._access_times[key] = time.time()
    
    def delete(self, key: str) -> None:
        """캐시에서 항목 삭제"""
        self._remove(key)
    
    def clear(self) -> None:
        """전체 캐시 삭제"""
        self.cache.clear()
        self._access_times.clear()
    
    def stats(self) -> Dict[str, Any]:
        """캐시 통계"""
        # 만료된 항목 정리
        expired_keys = [k for k in self.cache.keys() if self._is_expired(k)]
        for key in expired_keys:
            self._remove(key)
        
        return {
            'type': 'memory',
            'size': len(self.cache),
            'max_size': self.max_size,
            'hit_ratio': 0.0,  # 추후 구현
            'memory_usage': sum(len(str(v)) for v in self.cache.values())
        }


class RedisCache:
    """Redis 기반 캐시 (프로덕션 환경용)"""
    
    def __init__(self, redis_client, default_ttl: int = 3600):
        self.redis = redis_client
        self.default_ttl = default_ttl
        self.hit_count = 0
        self.miss_count = 0
    
    def _serialize(self, value: Any) -> bytes:
        """값 직렬화"""
        return pickle.dumps(value)
    
    def _deserialize(self, data: bytes) -> Any:
        """값 역직렬화"""
        return pickle.loads(data)
    
    async def get(self, key: str) -> Optional[Any]:
        """캐시에서 값 조회"""
        try:
            data = await self.redis.get(key)
            if data:
                self.hit_count += 1
                return self._deserialize(data)
            else:
                self.miss_count += 1
                return None
        except Exception as e:
            logger.error(f"Redis get error: {e}")
            self.miss_count += 1
            return None
    
    async def set(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        """캐시에 값 저장"""
        try:
            ttl = ttl or self.default_ttl
            data = self._serialize(value)
            await self.redis.setex(key, ttl, data)
        except Exception as e:
            logger.error(f"Redis set error: {e}")
    
    async def delete(self, key: str) -> None:
        """캐시에서 항목 삭제"""
        try:
            await self.redis.delete(key)
        except Exception as e:
            logger.error(f"Redis delete error: {e}")
    
    async def clear(self) -> None:
        """전체 캐시 삭제"""
        try:
            await self.redis.flushdb()
        except Exception as e:
            logger.error(f"Redis clear error: {e}")
    
    def stats(self) -> Dict[str, Any]:
        """캐시 통계"""
        total_requests = self.hit_count + self.miss_count
        hit_ratio = self.hit_count / total_requests if total_requests > 0 else 0.0
        
        return {
            'type': 'redis',
            'hit_count': self.hit_count,
            'miss_count': self.miss_count,
            'hit_ratio': hit_ratio,
            'total_requests': total_requests
        }


class CacheManager:
    """통합 캐시 관리자"""
    
    def __init__(self):
        self.cache = None
        self._initialize_cache()
    
    def _initialize_cache(self):
        """캐시 시스템 초기화"""
        try:
            # Redis 연결 시도
            import redis.asyncio as redis
            from app.core.config import settings
            
            redis_client = redis.Redis(
                host=getattr(settings, 'redis_host', 'localhost'),
                port=getattr(settings, 'redis_port', 6379),
                password=getattr(settings, 'redis_password', None),
                decode_responses=False  # bytes 형태로 저장
            )
            
            # 연결 테스트
            asyncio.create_task(self._test_redis_connection(redis_client))
            
        except ImportError:
            logger.warning("Redis not available, using memory cache")
            self.cache = MemoryCache()
        except Exception as e:
            logger.warning(f"Redis connection failed: {e}, using memory cache")
            self.cache = MemoryCache()
    
    async def _test_redis_connection(self, redis_client):
        """Redis 연결 테스트"""
        try:
            await redis_client.ping()
            self.cache = RedisCache(redis_client)
            logger.info("Redis cache initialized successfully")
        except Exception as e:
            logger.warning(f"Redis ping failed: {e}, using memory cache")
            self.cache = MemoryCache()
    
    def _generate_key(self, prefix: str, data: Union[str, Dict, List]) -> str:
        """캐시 키 생성"""
        if isinstance(data, str):
            content = data
        else:
            content = json.dumps(data, sort_keys=True, ensure_ascii=False)
        
        hash_value = hashlib.md5(content.encode('utf-8')).hexdigest()
        return f"{prefix}:{hash_value}"
    
    async def get_query_result(self, query: str, search_params: Dict) -> Optional[Dict]:
        """쿼리 결과 캐시 조회"""
        key = self._generate_key("query", {"query": query, "params": search_params})
        return await self._get(key)
    
    async def set_query_result(self, query: str, search_params: Dict, result: Dict, ttl: int = 1800) -> None:
        """쿼리 결과 캐시 저장 (30분 TTL)"""
        key = self._generate_key("query", {"query": query, "params": search_params})
        await self._set(key, result, ttl)
    
    async def get_embedding(self, text: str) -> Optional[List[float]]:
        """임베딩 캐시 조회"""
        key = self._generate_key("embedding", text)
        return await self._get(key)
    
    async def set_embedding(self, text: str, embedding: List[float], ttl: int = 86400) -> None:
        """임베딩 캐시 저장 (24시간 TTL)"""
        key = self._generate_key("embedding", text)
        await self._set(key, embedding, ttl)
    
    async def get_llm_response(self, prompt: str, model_params: Dict) -> Optional[str]:
        """LLM 응답 캐시 조회"""
        key = self._generate_key("llm", {"prompt": prompt, "params": model_params})
        return await self._get(key)
    
    async def set_llm_response(self, prompt: str, model_params: Dict, response: str, ttl: int = 3600) -> None:
        """LLM 응답 캐시 저장 (1시간 TTL)"""
        key = self._generate_key("llm", {"prompt": prompt, "params": model_params})
        await self._set(key, response, ttl)
    
    async def get_rerank_result(self, query: str, doc_ids: List[str]) -> Optional[List[Dict]]:
        """리랭킹 결과 캐시 조회"""
        key = self._generate_key("rerank", {"query": query, "docs": sorted(doc_ids)})
        return await self._get(key)
    
    async def set_rerank_result(self, query: str, doc_ids: List[str], result: List[Dict], ttl: int = 1800) -> None:
        """리랭킹 결과 캐시 저장 (30분 TTL)"""
        key = self._generate_key("rerank", {"query": query, "docs": sorted(doc_ids)})
        await self._set(key, result, ttl)
    
    async def _get(self, key: str) -> Optional[Any]:
        """캐시에서 값 조회 (내부 메서드)"""
        if isinstance(self.cache, MemoryCache):
            return self.cache.get(key)
        else:
            return await self.cache.get(key)
    
    async def _set(self, key: str, value: Any, ttl: int) -> None:
        """캐시에 값 저장 (내부 메서드)"""
        if isinstance(self.cache, MemoryCache):
            self.cache.set(key, value, ttl)
        else:
            await self.cache.set(key, value, ttl)
    
    async def clear_all(self) -> None:
        """전체 캐시 삭제"""
        if isinstance(self.cache, MemoryCache):
            self.cache.clear()
        else:
            await self.cache.clear()
        logger.info("All cache cleared")
    
    def get_stats(self) -> Dict[str, Any]:
        """캐시 통계 조회"""
        if self.cache:
            return self.cache.stats()
        return {"type": "none", "status": "not_initialized"}


# 전역 캐시 매니저 인스턴스
cache_manager = CacheManager()


# 편의 함수들
async def get_cached_query_result(query: str, search_params: Dict = None) -> Optional[Dict]:
    """쿼리 결과 캐시 조회"""
    return await cache_manager.get_query_result(query, search_params or {})


async def cache_query_result(query: str, result: Dict, search_params: Dict = None, ttl: int = 1800) -> None:
    """쿼리 결과 캐시 저장"""
    await cache_manager.set_query_result(query, search_params or {}, result, ttl)


async def get_cached_embedding(text: str) -> Optional[List[float]]:
    """임베딩 캐시 조회"""
    return await cache_manager.get_embedding(text)


async def cache_embedding(text: str, embedding: List[float], ttl: int = 86400) -> None:
    """임베딩 캐시 저장"""
    await cache_manager.set_embedding(text, embedding, ttl)


async def get_cached_llm_response(prompt: str, model_params: Dict = None) -> Optional[str]:
    """LLM 응답 캐시 조회"""
    return await cache_manager.get_llm_response(prompt, model_params or {})


async def cache_llm_response(prompt: str, response: str, model_params: Dict = None, ttl: int = 3600) -> None:
    """LLM 응답 캐시 저장"""
    await cache_manager.set_llm_response(prompt, model_params or {}, response, ttl)


def get_cache_stats() -> Dict[str, Any]:
    """캐시 통계 조회"""
    return cache_manager.get_stats()


async def clear_cache() -> None:
    """전체 캐시 삭제"""
    await cache_manager.clear_all()