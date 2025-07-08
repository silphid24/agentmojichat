"""
모델 최적화 시스템
모델 예열, 경량화, 공유를 통한 성능 향상
"""

import asyncio
import time
import threading
from typing import Dict, List, Callable
from dataclasses import dataclass
from enum import Enum
import gc

from app.core.logging import logger


class ModelSize(Enum):
    """모델 크기 수준"""

    TINY = "tiny"
    SMALL = "small"
    MEDIUM = "medium"
    LARGE = "large"


class OptimizationLevel(Enum):
    """최적화 수준"""

    MINIMAL = "minimal"  # 기본 최적화만
    BALANCED = "balanced"  # 균형 잡힌 최적화
    AGGRESSIVE = "aggressive"  # 적극적 최적화


@dataclass
class ModelConfig:
    """모델 설정"""

    model_name: str
    model_size: ModelSize
    warm_up_queries: List[str]
    max_cache_size: int = 100
    cache_ttl: int = 3600  # 1시간
    enable_lightweight_mode: bool = True


@dataclass
class ModelStats:
    """모델 성능 통계"""

    total_requests: int = 0
    cache_hits: int = 0
    warm_up_time: float = 0.0
    avg_response_time: float = 0.0
    memory_usage: float = 0.0
    last_used: float = 0.0


class ModelManager:
    """모델 매니저 - 싱글톤 패턴으로 모델 인스턴스 공유"""

    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super(ModelManager, cls).__new__(cls)
        return cls._instance

    def __init__(self):
        if not hasattr(self, "initialized"):
            self.models = {}  # 모델 인스턴스 저장
            self.configs = {}  # 모델 설정
            self.stats = {}  # 모델 통계
            self.warm_up_status = {}  # 예열 상태
            self._model_locks = {}  # 모델별 락
            self.initialized = True

    def register_model(
        self, model_id: str, model_factory: Callable, config: ModelConfig
    ):
        """모델 등록"""
        with self._lock:
            self.configs[model_id] = config
            self.stats[model_id] = ModelStats()
            self.warm_up_status[model_id] = False
            self._model_locks[model_id] = threading.Lock()

            logger.info(f"Model registered: {model_id} ({config.model_size.value})")

    def get_model(self, model_id: str):
        """모델 인스턴스 반환 (싱글톤)"""
        if model_id not in self.models:
            with self._model_locks.get(model_id, self._lock):
                if model_id not in self.models:
                    # 모델 인스턴스 생성 (지연 로딩)
                    self._create_model_instance(model_id)

        # 사용 통계 업데이트
        self.stats[model_id].last_used = time.time()
        self.stats[model_id].total_requests += 1

        return self.models[model_id]

    def _create_model_instance(self, model_id: str):
        """모델 인스턴스 생성"""
        try:
            config = self.configs[model_id]

            # 모델 팩토리로 인스턴스 생성
            if model_id == "embedding":
                from app.core.cached_embeddings import CachedOpenAIEmbeddings
                from app.core.config import settings

                api_key = settings.openai_api_key or settings.llm_api_key

                # 경량화된 임베딩 모델 설정
                if config.enable_lightweight_mode:
                    model_name = "text-embedding-3-small"  # 더 작고 빠른 모델
                else:
                    model_name = "text-embedding-3-large"  # 더 정확하지만 느린 모델

                self.models[model_id] = CachedOpenAIEmbeddings(
                    openai_api_key=api_key, model=model_name
                )

            elif model_id == "reranker":
                from app.rag.reranker import CrossEncoderReranker

                # 경량화된 리랭커 설정
                if config.enable_lightweight_mode:
                    model_name = "cross-encoder/ms-marco-MiniLM-L-6-v2"  # 빠른 모델
                else:
                    model_name = "cross-encoder/ms-marco-electra-base"  # 정확한 모델

                self.models[model_id] = CrossEncoderReranker(model_name=model_name)

            logger.info(f"Model instance created: {model_id}")

        except Exception as e:
            logger.error(f"Failed to create model instance {model_id}: {e}")
            raise

    async def warm_up_model(self, model_id: str):
        """모델 예열"""
        if self.warm_up_status.get(model_id, False):
            return  # 이미 예열됨

        try:
            start_time = time.time()
            config = self.configs[model_id]
            model = self.get_model(model_id)

            logger.info(f"Starting warm-up for {model_id}...")

            # 예열 쿼리 실행
            if model_id == "embedding":
                # 임베딩 모델 예열
                for query in config.warm_up_queries:
                    await self._warm_up_embedding(model, query)

            elif model_id == "reranker":
                # 리랭커 모델 예열
                for query in config.warm_up_queries:
                    await self._warm_up_reranker(model, query)

            warm_up_time = time.time() - start_time
            self.stats[model_id].warm_up_time = warm_up_time
            self.warm_up_status[model_id] = True

            logger.info(f"Model {model_id} warmed up in {warm_up_time:.3f}s")

        except Exception as e:
            logger.error(f"Model warm-up failed for {model_id}: {e}")

    async def _warm_up_embedding(self, model, query: str):
        """임베딩 모델 예열"""
        try:
            # 임베딩 생성 (결과는 캐시됨)
            await asyncio.get_event_loop().run_in_executor(
                None, lambda: model.embed_query(query)
            )
        except Exception as e:
            logger.warning(f"Embedding warm-up query failed: {e}")

    async def _warm_up_reranker(self, model, query: str):
        """리랭커 모델 예열"""
        try:
            from langchain.schema import Document

            # 더미 문서 생성
            dummy_docs = [
                Document(page_content=f"Sample document {i} for warm-up", metadata={})
                for i in range(3)
            ]

            # 리랭킹 실행
            await asyncio.get_event_loop().run_in_executor(
                None, lambda: model.rerank(query, dummy_docs, [0.5, 0.6, 0.7])
            )
        except Exception as e:
            logger.warning(f"Reranker warm-up query failed: {e}")

    def cleanup_unused_models(self, max_idle_time: int = 3600):
        """사용하지 않는 모델 정리"""
        current_time = time.time()
        models_to_remove = []

        for model_id, stats in self.stats.items():
            if current_time - stats.last_used > max_idle_time:
                models_to_remove.append(model_id)

        for model_id in models_to_remove:
            self._unload_model(model_id)

    def _unload_model(self, model_id: str):
        """모델 언로드"""
        try:
            if model_id in self.models:
                del self.models[model_id]
                self.warm_up_status[model_id] = False

                # 가비지 컬렉션 강제 실행
                gc.collect()

                logger.info(f"Model unloaded: {model_id}")
        except Exception as e:
            logger.error(f"Failed to unload model {model_id}: {e}")

    def get_model_stats(self) -> Dict[str, ModelStats]:
        """모델 통계 반환"""
        return self.stats.copy()

    def optimize_memory(self):
        """메모리 최적화"""
        try:
            # 가비지 컬렉션 실행
            gc.collect()

            # 사용하지 않는 모델 정리
            self.cleanup_unused_models(max_idle_time=1800)  # 30분

            logger.info("Memory optimization completed")

        except Exception as e:
            logger.error(f"Memory optimization failed: {e}")


class AdaptiveModelSelector:
    """적응형 모델 선택기"""

    def __init__(self, model_manager: ModelManager):
        self.model_manager = model_manager
        self.performance_history = {}

    def select_optimal_model(
        self, task_type: str, complexity: str, performance_priority: str = "balanced"
    ) -> str:
        """최적 모델 선택"""
        try:
            if task_type == "embedding":
                return self._select_embedding_model(complexity, performance_priority)
            elif task_type == "reranking":
                return self._select_reranker_model(complexity, performance_priority)
            else:
                return "default"

        except Exception as e:
            logger.error(f"Model selection failed: {e}")
            return "default"

    def _select_embedding_model(self, complexity: str, priority: str) -> str:
        """임베딩 모델 선택"""
        if priority == "speed" or complexity == "simple":
            return "embedding_small"
        elif priority == "quality" or complexity == "complex":
            return "embedding_large"
        else:
            return "embedding"  # 기본값

    def _select_reranker_model(self, complexity: str, priority: str) -> str:
        """리랭커 모델 선택"""
        if priority == "speed" or complexity == "simple":
            return "reranker_fast"
        elif priority == "quality" or complexity == "complex":
            return "reranker_accurate"
        else:
            return "reranker"  # 기본값

    def record_performance(
        self,
        model_id: str,
        task_type: str,
        response_time: float,
        quality_score: float = None,
    ):
        """성능 기록"""
        try:
            if model_id not in self.performance_history:
                self.performance_history[model_id] = {
                    "response_times": [],
                    "quality_scores": [],
                    "usage_count": 0,
                }

            self.performance_history[model_id]["response_times"].append(response_time)
            self.performance_history[model_id]["usage_count"] += 1

            if quality_score is not None:
                self.performance_history[model_id]["quality_scores"].append(
                    quality_score
                )

            # 히스토리 크기 제한
            if len(self.performance_history[model_id]["response_times"]) > 100:
                self.performance_history[model_id]["response_times"].pop(0)

            if len(self.performance_history[model_id]["quality_scores"]) > 100:
                self.performance_history[model_id]["quality_scores"].pop(0)

        except Exception as e:
            logger.error(f"Failed to record performance for {model_id}: {e}")


# 전역 인스턴스들
model_manager = ModelManager()
adaptive_selector = AdaptiveModelSelector(model_manager)


# 모델 설정 등록
def initialize_model_configurations():
    """모델 설정 초기화"""
    try:
        # 임베딩 모델 설정
        embedding_config = ModelConfig(
            model_name="text-embedding-3-small",
            model_size=ModelSize.SMALL,
            warm_up_queries=[
                "MOJI AI 시스템",
                "프로젝트 관리",
                "사용자 가이드",
                "기능 설명",
            ],
            max_cache_size=1000,
            enable_lightweight_mode=True,
        )

        # 리랭커 모델 설정
        reranker_config = ModelConfig(
            model_name="cross-encoder/ms-marco-MiniLM-L-6-v2",
            model_size=ModelSize.SMALL,
            warm_up_queries=["테스트 쿼리 1", "테스트 쿼리 2", "성능 테스트"],
            max_cache_size=500,
            enable_lightweight_mode=True,
        )

        # 모델 등록
        model_manager.register_model("embedding", None, embedding_config)
        model_manager.register_model("reranker", None, reranker_config)

        logger.info("Model configurations initialized")

    except Exception as e:
        logger.error(f"Failed to initialize model configurations: {e}")


# 모델 예열 함수
async def warm_up_all_models():
    """모든 모델 예열"""
    try:
        # 중요한 모델들부터 예열
        priority_models = ["embedding", "reranker"]

        # 병렬 예열
        warm_up_tasks = []
        for model_id in priority_models:
            if model_id in model_manager.configs:
                task = model_manager.warm_up_model(model_id)
                warm_up_tasks.append(task)

        if warm_up_tasks:
            await asyncio.gather(*warm_up_tasks, return_exceptions=True)
            logger.info(f"Warmed up {len(warm_up_tasks)} models")

    except Exception as e:
        logger.error(f"Model warm-up failed: {e}")


# 편의 함수들
def get_optimized_embedding_model():
    """최적화된 임베딩 모델 반환"""
    return model_manager.get_model("embedding")


def get_optimized_reranker_model():
    """최적화된 리랭커 모델 반환"""
    return model_manager.get_model("reranker")


async def optimize_model_performance():
    """모델 성능 최적화 실행"""
    try:
        # 1. 메모리 최적화
        model_manager.optimize_memory()

        # 2. 모델 예열
        await warm_up_all_models()

        # 3. 성능 통계 로깅
        stats = model_manager.get_model_stats()
        for model_id, stat in stats.items():
            logger.info(
                f"Model {model_id}: {stat.total_requests} requests, "
                f"avg response: {stat.avg_response_time:.3f}s"
            )

        logger.info("Model performance optimization completed")

    except Exception as e:
        logger.error(f"Model performance optimization failed: {e}")


# 자동 초기화
initialize_model_configurations()
