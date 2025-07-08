"""
비동기 처리 유틸리티
병렬 처리, 배치 작업, 동시성 제어를 위한 헬퍼 함수들
"""

import asyncio
import time
from typing import List, Dict, Any, Callable, Optional, TypeVar
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from functools import wraps

from app.core.logging import logger

T = TypeVar("T")
R = TypeVar("R")


class AsyncBatchProcessor:
    """비동기 배치 처리기"""

    def __init__(self, batch_size: int = 10, max_concurrent: int = 5):
        self.batch_size = batch_size
        self.max_concurrent = max_concurrent
        self.semaphore = asyncio.Semaphore(max_concurrent)

    async def process_batch(
        self, items: List[T], processor: Callable[[List[T]], Any], is_async: bool = True
    ) -> List[Any]:
        """배치 단위로 아이템들을 처리"""
        if not items:
            return []

        # 배치로 분할
        batches = [
            items[i : i + self.batch_size]
            for i in range(0, len(items), self.batch_size)
        ]

        logger.info(f"Processing {len(items)} items in {len(batches)} batches")

        # 병렬 처리
        if is_async:
            tasks = [
                self._process_single_batch_async(batch, processor) for batch in batches
            ]
            batch_results = await asyncio.gather(*tasks, return_exceptions=True)
        else:
            tasks = [
                self._process_single_batch_sync(batch, processor) for batch in batches
            ]
            batch_results = await asyncio.gather(*tasks, return_exceptions=True)

        # 결과 평탄화
        results = []
        for batch_result in batch_results:
            if isinstance(batch_result, Exception):
                logger.error(f"Batch processing error: {batch_result}")
                continue

            if isinstance(batch_result, list):
                results.extend(batch_result)
            else:
                results.append(batch_result)

        return results

    async def _process_single_batch_async(
        self, batch: List[T], processor: Callable
    ) -> Any:
        """단일 배치 비동기 처리"""
        async with self.semaphore:
            try:
                return await processor(batch)
            except Exception as e:
                logger.error(f"Async batch processing error: {e}")
                raise

    async def _process_single_batch_sync(
        self, batch: List[T], processor: Callable
    ) -> Any:
        """단일 배치 동기 처리 (스레드풀 사용)"""
        async with self.semaphore:
            try:
                loop = asyncio.get_event_loop()
                return await loop.run_in_executor(None, processor, batch)
            except Exception as e:
                logger.error(f"Sync batch processing error: {e}")
                raise


class ParallelSearchManager:
    """병렬 검색 관리자"""

    def __init__(self, max_concurrent: int = 4):
        self.max_concurrent = max_concurrent
        self.semaphore = asyncio.Semaphore(max_concurrent)

    async def parallel_search(
        self,
        queries: List[str],
        search_functions: List[Callable],
        search_params: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """여러 검색 함수를 병렬로 실행"""
        if not queries or not search_functions:
            return []

        search_params = search_params or {}

        # 모든 쿼리-함수 조합에 대한 태스크 생성
        tasks = []
        for query in queries:
            for search_func in search_functions:
                task = self._execute_search(query, search_func, search_params)
                tasks.append(task)

        logger.info(f"Executing {len(tasks)} parallel searches")

        # 병렬 실행
        start_time = time.time()
        results = await asyncio.gather(*tasks, return_exceptions=True)
        elapsed_time = time.time() - start_time

        logger.info(f"Parallel searches completed in {elapsed_time:.3f}s")

        # 결과 정리
        valid_results = []
        for result in results:
            if isinstance(result, Exception):
                logger.error(f"Search error: {result}")
                continue

            if result and isinstance(result, dict):
                valid_results.append(result)

        return valid_results

    async def _execute_search(
        self, query: str, search_func: Callable, search_params: Dict[str, Any]
    ) -> Dict[str, Any]:
        """단일 검색 실행"""
        async with self.semaphore:
            try:
                if asyncio.iscoroutinefunction(search_func):
                    result = await search_func(query, **search_params)
                else:
                    loop = asyncio.get_event_loop()
                    result = await loop.run_in_executor(
                        None, search_func, query, **search_params
                    )

                return {
                    "query": query,
                    "function": search_func.__name__,
                    "result": result,
                    "timestamp": time.time(),
                }

            except Exception as e:
                logger.error(f"Search execution error for query '{query}': {e}")
                return {
                    "query": query,
                    "function": search_func.__name__,
                    "error": str(e),
                    "timestamp": time.time(),
                }


class ConcurrentTaskManager:
    """동시성 작업 관리자"""

    def __init__(self, max_workers: int = 4):
        self.max_workers = max_workers
        self.thread_pool = ThreadPoolExecutor(max_workers=max_workers)
        self.process_pool = None  # 필요시 생성

    async def run_concurrent_tasks(
        self,
        tasks: List[Callable],
        use_processes: bool = False,
        timeout: Optional[float] = None,
    ) -> List[Any]:
        """동시 작업 실행"""
        if not tasks:
            return []

        executor = self._get_executor(use_processes)

        try:
            loop = asyncio.get_event_loop()

            # 작업을 executor에 제출
            futures = [loop.run_in_executor(executor, task) for task in tasks]

            # 타임아웃과 함께 결과 대기
            if timeout:
                results = await asyncio.wait_for(
                    asyncio.gather(*futures, return_exceptions=True), timeout=timeout
                )
            else:
                results = await asyncio.gather(*futures, return_exceptions=True)

            return results

        except asyncio.TimeoutError:
            logger.error(f"Concurrent tasks timed out after {timeout}s")
            return []
        except Exception as e:
            logger.error(f"Concurrent task execution error: {e}")
            return []

    def _get_executor(self, use_processes: bool):
        """적절한 executor 반환"""
        if use_processes:
            if self.process_pool is None:
                self.process_pool = ProcessPoolExecutor(max_workers=self.max_workers)
            return self.process_pool
        else:
            return self.thread_pool

    def cleanup(self):
        """리소스 정리"""
        if self.thread_pool:
            self.thread_pool.shutdown(wait=True)
        if self.process_pool:
            self.process_pool.shutdown(wait=True)


def async_retry(
    max_attempts: int = 3,
    delay: float = 1.0,
    backoff_factor: float = 2.0,
    exceptions: tuple = (Exception,),
):
    """비동기 함수용 재시도 데코레이터"""

    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            last_exception = None

            for attempt in range(max_attempts):
                try:
                    return await func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e
                    if attempt == max_attempts - 1:
                        break

                    wait_time = delay * (backoff_factor**attempt)
                    logger.warning(
                        f"Attempt {attempt + 1} failed: {e}. Retrying in {wait_time}s..."
                    )
                    await asyncio.sleep(wait_time)

            logger.error(
                f"All {max_attempts} attempts failed. Last error: {last_exception}"
            )
            raise last_exception

        return wrapper

    return decorator


def async_timeout(seconds: float):
    """비동기 함수용 타임아웃 데코레이터"""

    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            try:
                return await asyncio.wait_for(func(*args, **kwargs), timeout=seconds)
            except asyncio.TimeoutError:
                logger.error(f"Function {func.__name__} timed out after {seconds}s")
                raise

        return wrapper

    return decorator


class AsyncRateLimiter:
    """비동기 속도 제한기"""

    def __init__(self, rate: int, per: float = 1.0):
        self.rate = rate  # 허용 요청 수
        self.per = per  # 시간 단위 (초)
        self.tokens = rate
        self.last_update = time.time()
        self.lock = asyncio.Lock()

    async def acquire(self):
        """토큰 획득 (속도 제한 적용)"""
        async with self.lock:
            now = time.time()
            elapsed = now - self.last_update

            # 토큰 보충
            self.tokens = min(self.rate, self.tokens + elapsed * (self.rate / self.per))
            self.last_update = now

            if self.tokens >= 1:
                self.tokens -= 1
                return

            # 토큰이 부족하면 대기
            wait_time = (1 - self.tokens) * (self.per / self.rate)
            await asyncio.sleep(wait_time)
            self.tokens = 0


# 전역 인스턴스들
batch_processor = AsyncBatchProcessor()
parallel_search_manager = ParallelSearchManager()
concurrent_task_manager = ConcurrentTaskManager()


# 편의 함수들
async def process_in_batches(
    items: List[T], processor: Callable, batch_size: int = 10, is_async: bool = True
) -> List[Any]:
    """배치 처리 편의 함수"""
    processor_instance = AsyncBatchProcessor(batch_size=batch_size)
    return await processor_instance.process_batch(items, processor, is_async)


async def run_parallel_searches(
    queries: List[str],
    search_functions: List[Callable],
    search_params: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    """병렬 검색 편의 함수"""
    return await parallel_search_manager.parallel_search(
        queries, search_functions, search_params
    )


async def execute_concurrent_tasks(
    tasks: List[Callable], timeout: Optional[float] = None
) -> List[Any]:
    """동시 작업 실행 편의 함수"""
    return await concurrent_task_manager.run_concurrent_tasks(tasks, timeout=timeout)
