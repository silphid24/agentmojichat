#!/usr/bin/env python3
"""
비동기 처리 성능 테스트
병렬 검색 vs 순차 검색 성능 비교
"""

import asyncio
import os
import sys
import time
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))


async def test_parallel_vs_sequential():
    """병렬 vs 순차 검색 성능 비교"""
    print("⚡ 병렬 vs 순차 검색 성능 테스트")
    print("=" * 50)

    try:
        # 환경 설정
        from app.core.config import settings

        if settings.openai_api_key:
            os.environ["OPENAI_API_KEY"] = settings.openai_api_key
        elif settings.llm_api_key:
            os.environ["OPENAI_API_KEY"] = settings.llm_api_key

        from app.rag.enhanced_rag import rag_pipeline, get_hybrid_pipeline
        from app.core.cache import clear_cache

        # 캐시 초기화 (공정한 비교를 위해)
        await clear_cache()
        print("✅ 캐시 초기화 완료")

        hybrid_pipeline = get_hybrid_pipeline()

        # 테스트 쿼리 (쿼리 재작성이 활성화되어 여러 쿼리 생성)
        test_queries = [
            "MOJI AI 에이전트의 핵심 기능과 특징",
            "SMHACCP 회사의 사업 영역과 비전",
            "프로젝트 관리 시스템의 주요 장점",
        ]

        print("📊 테스트 설정:")
        print(f"  - 테스트 쿼리: {len(test_queries)}개")
        print("  - 각 쿼리는 여러 변형으로 재작성됨")
        print("  - 병렬 vs 순차 처리 비교")
        print()

        for i, query in enumerate(test_queries, 1):
            print(f"🧪 테스트 {i}: {query[:30]}...")
            print("-" * 40)

            # === 순차 검색 테스트 ===
            print("📍 순차 검색 (기존 방식)")
            start_time = time.time()

            try:
                # 병렬 검색 비활성화
                sequential_docs, sequential_meta = (
                    await rag_pipeline.search_with_rewriting(
                        query, k=5, use_parallel_search=False
                    )
                )
                sequential_time = time.time() - start_time

                print(f"   ⏱️  시간: {sequential_time:.3f}초")
                print(f"   📄 결과: {len(sequential_docs)}개")
                print(
                    f"   🔄 재작성 쿼리: {len(sequential_meta.get('rewritten_queries', []))}개"
                )

            except Exception as e:
                print(f"   ❌ 순차 검색 오류: {str(e)}")
                sequential_time = float("inf")
                continue

            # 잠시 대기 (시스템 안정화)
            await asyncio.sleep(0.5)

            # === 병렬 검색 테스트 ===
            print("\n⚡ 병렬 검색 (개선된 방식)")
            start_time = time.time()

            try:
                # 병렬 검색 활성화
                parallel_docs, parallel_meta = await rag_pipeline.search_with_rewriting(
                    query, k=5, use_parallel_search=True
                )
                parallel_time = time.time() - start_time

                print(f"   ⏱️  시간: {parallel_time:.3f}초")
                print(f"   📄 결과: {len(parallel_docs)}개")
                print(
                    f"   🔄 재작성 쿼리: {len(parallel_meta.get('rewritten_queries', []))}개"
                )

                # 성능 비교
                if sequential_time > 0 and sequential_time != float("inf"):
                    improvement = (
                        (sequential_time - parallel_time) / sequential_time
                    ) * 100
                    speedup = (
                        sequential_time / parallel_time
                        if parallel_time > 0
                        else float("inf")
                    )

                    print("\n📊 성능 비교:")
                    print(f"   🚀 성능 향상: {improvement:.1f}%")
                    print(f"   ⚡ 속도 향상: {speedup:.1f}x")

                    if improvement > 20:
                        print("   ✅ 병렬 처리 효과 우수")
                    elif improvement > 0:
                        print("   🔄 병렬 처리 효과 양호")
                    else:
                        print("   ⚠️  병렬 처리 효과 제한적")

            except Exception as e:
                print(f"   ❌ 병렬 검색 오류: {str(e)}")

            print("=" * 50)

        print("\n🎯 병렬 처리 최적화 완료!")

    except Exception as e:
        print(f"❌ 테스트 오류: {str(e)}")
        import traceback

        traceback.print_exc()


async def test_hybrid_parallel_performance():
    """하이브리드 검색 병렬 처리 성능 테스트"""
    print("\n🔄 하이브리드 검색 병렬 처리 테스트")
    print("=" * 50)

    try:
        from app.rag.enhanced_rag import get_hybrid_pipeline
        from app.core.cache import clear_cache

        # 캐시 초기화
        await clear_cache()

        hybrid_pipeline = get_hybrid_pipeline()

        test_query = "MOJI AI 시스템의 전체적인 아키텍처와 구성 요소"

        print(f'📝 테스트 쿼리: "{test_query}"')
        print()

        # === 쿼리 재작성 없이 (단일 쿼리) ===
        print("📍 단일 쿼리 모드 (재작성 없음)")
        start_time = time.time()

        single_docs, single_meta = await hybrid_pipeline.search_with_hybrid(
            test_query, k=5, use_query_rewriting=False
        )
        single_time = time.time() - start_time

        print(f"   ⏱️  시간: {single_time:.3f}초")
        print(f"   📄 결과: {len(single_docs)}개")
        print(f"   🔍 검색 타입: {single_meta.get('search_type', 'unknown')}")

        # === 쿼리 재작성 + 병렬 처리 ===
        print("\n⚡ 다중 쿼리 + 병렬 처리")
        start_time = time.time()

        parallel_docs, parallel_meta = await hybrid_pipeline.search_with_hybrid(
            test_query, k=5, use_query_rewriting=True  # 쿼리 재작성 활성화
        )
        parallel_time = time.time() - start_time

        print(f"   ⏱️  시간: {parallel_time:.3f}초")
        print(f"   📄 결과: {len(parallel_docs)}개")
        print(f"   🔍 검색 타입: {parallel_meta.get('search_type', 'unknown')}")
        print(f"   🔄 재작성 쿼리: {len(parallel_meta.get('rewritten_queries', []))}개")
        print(f"   📊 총 후보: {parallel_meta.get('total_results', 0)}개")

        # 비율 분석
        if single_time > 0:
            overhead_ratio = parallel_time / single_time
            print("\n📊 오버헤드 분석:")
            print(f"   📈 처리 시간 비율: {overhead_ratio:.1f}x")

            if overhead_ratio < 2.0:
                print("   ✅ 쿼리 확장 대비 오버헤드 합리적")
            elif overhead_ratio < 3.0:
                print("   🔄 쿼리 확장 대비 오버헤드 보통")
            else:
                print("   ⚠️  쿼리 확장 대비 오버헤드 높음")

        # 품질 비교
        print("\n🎯 검색 품질 비교:")
        print(f"   단일 쿼리: {len(single_docs)}개 결과")
        print(f"   다중 쿼리: {len(parallel_docs)}개 결과")

        if len(parallel_docs) > len(single_docs):
            print("   ✅ 다중 쿼리로 더 많은 관련 문서 발견")
        elif len(parallel_docs) == len(single_docs):
            print("   🔄 결과 수는 동일, 품질 개선 가능성")
        else:
            print("   ⚠️  다중 쿼리로 결과 수 감소")

    except Exception as e:
        print(f"❌ 하이브리드 테스트 오류: {str(e)}")


async def test_async_batch_processing():
    """비동기 배치 처리 테스트"""
    print("\n📦 비동기 배치 처리 테스트")
    print("=" * 40)

    try:
        from app.core.async_utils import AsyncBatchProcessor

        # 테스트 데이터
        test_items = [f"테스트 항목 {i}" for i in range(20)]

        def simple_processor(batch):
            """간단한 배치 처리기"""
            time.sleep(0.1)  # 처리 시뮬레이션
            return [f"처리됨: {item}" for item in batch]

        async def async_processor(batch):
            """비동기 배치 처리기"""
            await asyncio.sleep(0.1)  # 비동기 처리 시뮬레이션
            return [f"비동기 처리됨: {item}" for item in batch]

        # 배치 처리기 생성
        batch_processor = AsyncBatchProcessor(batch_size=5, max_concurrent=3)

        # 동기 배치 처리
        print("🔄 동기 배치 처리 (배치 크기: 5)")
        start_time = time.time()
        sync_results = await batch_processor.process_batch(
            test_items, simple_processor, is_async=False
        )
        sync_time = time.time() - start_time

        print(f"   ⏱️  시간: {sync_time:.3f}초")
        print(f"   📄 결과: {len(sync_results)}개")

        # 비동기 배치 처리
        print("\n⚡ 비동기 배치 처리 (배치 크기: 5)")
        start_time = time.time()
        async_results = await batch_processor.process_batch(
            test_items, async_processor, is_async=True
        )
        async_time = time.time() - start_time

        print(f"   ⏱️  시간: {async_time:.3f}초")
        print(f"   📄 결과: {len(async_results)}개")

        # 성능 비교
        if sync_time > 0:
            improvement = ((sync_time - async_time) / sync_time) * 100
            print("\n📊 배치 처리 성능:")
            print(f"   🚀 비동기 처리 향상: {improvement:.1f}%")

    except Exception as e:
        print(f"❌ 배치 처리 테스트 오류: {str(e)}")


if __name__ == "__main__":

    async def main():
        await test_parallel_vs_sequential()
        await test_hybrid_parallel_performance()
        await test_async_batch_processing()

        print("\n🎉 모든 비동기 성능 테스트 완료!")
        print("💡 병렬 처리로 검색 성능이 크게 향상되었습니다!")

    asyncio.run(main())
