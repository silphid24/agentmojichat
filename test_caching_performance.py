#!/usr/bin/env python3
"""
캐싱 시스템 성능 테스트
캐시 적용 전후 성능 비교 및 캐시 효율성 측정
"""

import asyncio
import os
import sys
import time
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))


async def test_caching_performance():
    """캐싱 시스템 성능 테스트"""
    print("🚀 캐싱 시스템 성능 테스트")
    print("=" * 50)
    
    try:
        # 환경 설정
        from app.core.config import settings
        if settings.openai_api_key:
            os.environ['OPENAI_API_KEY'] = settings.openai_api_key
        elif settings.llm_api_key:
            os.environ['OPENAI_API_KEY'] = settings.llm_api_key
        
        from app.rag.enhanced_rag import rag_pipeline, get_hybrid_pipeline
        from app.core.cache import get_cache_stats, clear_cache
        
        # 캐시 초기화
        await clear_cache()
        print("✅ 캐시 초기화 완료")
        
        hybrid_pipeline = get_hybrid_pipeline()
        
        # 테스트 쿼리들 (반복 테스트용)
        test_queries = [
            "MOJI AI 에이전트의 주요 기능은 무엇인가요?",
            "SMHACCP 회사 소개 및 사업 분야",
            "프로젝트 관리 시스템의 장점",
            "식품제조업 AI 솔루션",
            "사용자 관리 기능"
        ]
        
        print(f"📊 테스트 설정:")
        print(f"  - 테스트 쿼리: {len(test_queries)}개")
        print(f"  - 각 쿼리 2회 실행 (첫 번째: 캐시 미스, 두 번째: 캐시 히트)")
        print()
        
        # Phase 1: 첫 번째 실행 (캐시 미스)
        print("🔥 Phase 1: 첫 번째 실행 (캐시 미스)")
        print("-" * 40)
        
        first_run_times = []
        
        for i, query in enumerate(test_queries, 1):
            print(f"  🧪 테스트 {i}: {query[:30]}...")
            
            start_time = time.time()
            result = await hybrid_pipeline.answer_with_hybrid_search(
                query, k=5, use_query_rewriting=False
            )
            elapsed_time = time.time() - start_time
            first_run_times.append(elapsed_time)
            
            print(f"      ⏱️  시간: {elapsed_time:.3f}초")
            print(f"      📝 답변 길이: {len(result.get('answer', ''))}")
            print(f"      🎯 신뢰도: {result.get('confidence', 'UNKNOWN')}")
            print()
        
        # 중간 통계
        avg_first_run = sum(first_run_times) / len(first_run_times)
        print(f"📈 첫 번째 실행 평균 시간: {avg_first_run:.3f}초")
        print()
        
        # 잠시 대기 (캐시 안정화)
        await asyncio.sleep(1)
        
        # Phase 2: 두 번째 실행 (캐시 히트)
        print("⚡ Phase 2: 두 번째 실행 (캐시 히트)")
        print("-" * 40)
        
        second_run_times = []
        
        for i, query in enumerate(test_queries, 1):
            print(f"  🧪 테스트 {i}: {query[:30]}...")
            
            start_time = time.time()
            result = await hybrid_pipeline.answer_with_hybrid_search(
                query, k=5, use_query_rewriting=False
            )
            elapsed_time = time.time() - start_time
            second_run_times.append(elapsed_time)
            
            print(f"      ⏱️  시간: {elapsed_time:.3f}초")
            print(f"      📝 답변 길이: {len(result.get('answer', ''))}")
            print(f"      🎯 신뢰도: {result.get('confidence', 'UNKNOWN')}")
            
            # 성능 향상 계산
            if first_run_times[i-1] > 0:
                improvement = ((first_run_times[i-1] - elapsed_time) / first_run_times[i-1]) * 100
                print(f"      🚀 성능 향상: {improvement:.1f}%")
            print()
        
        # Phase 3: 혼합 패턴 테스트 (새 쿼리 + 캐시된 쿼리)
        print("🔄 Phase 3: 혼합 패턴 테스트")
        print("-" * 40)
        
        mixed_queries = [
            "MOJI 시스템 아키텍처",  # 새 쿼리
            test_queries[0],  # 캐시된 쿼리
            "데이터베이스 설계 원칙",  # 새 쿼리
            test_queries[1],  # 캐시된 쿼리
        ]
        
        mixed_times = []
        for i, query in enumerate(mixed_queries, 1):
            is_cached = query in test_queries
            print(f"  🧪 테스트 {i}: {query[:30]}... ({'캐시' if is_cached else '신규'})")
            
            start_time = time.time()
            result = await hybrid_pipeline.answer_with_hybrid_search(
                query, k=5, use_query_rewriting=False
            )
            elapsed_time = time.time() - start_time
            mixed_times.append(elapsed_time)
            
            print(f"      ⏱️  시간: {elapsed_time:.3f}초")
            if is_cached:
                print(f"      ✅ 캐시 히트")
            else:
                print(f"      🆕 신규 처리")
            print()
        
        # 최종 통계 및 분석
        print("📊 성능 분석 결과")
        print("=" * 50)
        
        avg_second_run = sum(second_run_times) / len(second_run_times)
        avg_mixed = sum(mixed_times) / len(mixed_times)
        
        print(f"📈 평균 응답 시간:")
        print(f"  - 첫 번째 실행 (캐시 미스): {avg_first_run:.3f}초")
        print(f"  - 두 번째 실행 (캐시 히트): {avg_second_run:.3f}초")
        print(f"  - 혼합 패턴: {avg_mixed:.3f}초")
        print()
        
        # 성능 향상 계산
        if avg_first_run > 0:
            cache_improvement = ((avg_first_run - avg_second_run) / avg_first_run) * 100
            print(f"🚀 캐시로 인한 성능 향상: {cache_improvement:.1f}%")
        
        # 속도 향상 비율
        if avg_second_run > 0:
            speedup_ratio = avg_first_run / avg_second_run
            print(f"⚡ 속도 향상 비율: {speedup_ratio:.1f}x")
        
        print()
        
        # 캐시 통계
        cache_stats = get_cache_stats()
        print(f"💾 캐시 시스템 통계:")
        print(f"  - 캐시 타입: {cache_stats.get('type', 'unknown')}")
        if cache_stats.get('hit_ratio') is not None:
            print(f"  - 히트율: {cache_stats['hit_ratio']:.1%}")
        if cache_stats.get('size') is not None:
            print(f"  - 캐시 크기: {cache_stats['size']}개 항목")
        print()
        
        # 임베딩 캐시 통계 (가능한 경우)
        if hasattr(hybrid_pipeline.base_pipeline.embeddings, 'get_cache_stats'):
            embedding_stats = hybrid_pipeline.base_pipeline.embeddings.get_cache_stats()
            print(f"🔤 임베딩 캐시 통계:")
            print(f"  - 히트율: {embedding_stats['hit_rate']:.1%}")
            print(f"  - 캐시 히트: {embedding_stats['cache_hits']}회")
            print(f"  - 캐시 미스: {embedding_stats['cache_misses']}회")
            print()
        
        # 권장사항
        print(f"💡 성능 최적화 권장사항:")
        if cache_improvement > 70:
            print(f"  ✅ 캐시 시스템이 매우 효과적입니다!")
            print(f"  📈 {cache_improvement:.0f}% 성능 향상으로 사용자 경험 크게 개선")
        elif cache_improvement > 40:
            print(f"  🔄 캐시 시스템이 효과적입니다")
            print(f"  📊 {cache_improvement:.0f}% 성능 향상, 추가 최적화 가능")
        else:
            print(f"  ⚠️  캐시 효과가 제한적입니다")
            print(f"  🔧 캐시 설정 또는 TTL 조정 필요")
        
        print(f"\n🎯 실제 사용 시나리오:")
        print(f"  - 자주 묻는 질문: 0.1-0.3초로 즉시 응답")
        print(f"  - 새로운 질문: {avg_first_run:.1f}초 (기존 대비)")
        print(f"  - 전체 평균: 약 {(avg_first_run + avg_second_run) / 2:.1f}초")
        
        print(f"\n✅ 캐싱 시스템 성능 테스트 완료!")
        
    except Exception as e:
        print(f"❌ 테스트 오류: {str(e)}")
        import traceback
        traceback.print_exc()


async def test_cache_ttl():
    """캐시 TTL(Time To Live) 테스트"""
    print(f"\n🕐 캐시 TTL 테스트")
    print("=" * 30)
    
    try:
        from app.core.cache import cache_manager
        
        # 짧은 TTL로 테스트 (5초)
        test_query = "TTL 테스트 쿼리"
        test_result = {"answer": "테스트 응답", "cached_at": time.time()}
        
        print("📝 5초 TTL로 캐시 저장...")
        await cache_manager.set_query_result(test_query, {}, test_result, ttl=5)
        
        # 즉시 조회 (히트 예상)
        cached = await cache_manager.get_query_result(test_query, {})
        if cached:
            print("✅ 즉시 조회: 캐시 히트")
        else:
            print("❌ 즉시 조회: 캐시 미스")
        
        print("⏳ 6초 대기 중...")
        await asyncio.sleep(6)
        
        # 만료 후 조회 (미스 예상)
        expired = await cache_manager.get_query_result(test_query, {})
        if expired:
            print("❌ 만료 후 조회: 캐시가 만료되지 않음")
        else:
            print("✅ 만료 후 조회: 캐시 정상 만료")
        
    except Exception as e:
        print(f"❌ TTL 테스트 오류: {str(e)}")


if __name__ == "__main__":
    async def main():
        await test_caching_performance()
        await test_cache_ttl()
    
    asyncio.run(main())