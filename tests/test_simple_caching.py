#!/usr/bin/env python3
"""
간단한 캐싱 테스트
기본 RAG 파이프라인으로 캐시 기능 검증
"""

import asyncio
import os
import sys
import time
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))


async def test_basic_caching():
    """기본 캐싱 기능 테스트"""
    print("🧪 기본 캐싱 기능 테스트")
    print("=" * 40)
    
    try:
        # 환경 설정
        from app.core.config import settings
        if settings.openai_api_key:
            os.environ['OPENAI_API_KEY'] = settings.openai_api_key
        elif settings.llm_api_key:
            os.environ['OPENAI_API_KEY'] = settings.llm_api_key
        
        from app.rag.enhanced_rag import rag_pipeline
        from app.core.cache import get_cache_stats, clear_cache
        
        # 캐시 초기화
        await clear_cache()
        print("✅ 캐시 초기화 완료")
        
        # 테스트 쿼리
        test_query = "MOJI AI 에이전트의 주요 기능"
        
        print(f"\n📝 테스트 쿼리: \"{test_query}\"")
        print("-" * 40)
        
        # 첫 번째 실행 (캐시 미스)
        print("🔥 첫 번째 실행 (캐시 미스 예상)")
        start_time = time.time()
        
        result1 = await rag_pipeline.answer_with_confidence(
            test_query, k=3, score_threshold=0.1
        )
        
        first_time = time.time() - start_time
        print(f"   ⏱️  시간: {first_time:.3f}초")
        print(f"   📝 답변 길이: {len(result1.get('answer', ''))}")
        print(f"   🎯 신뢰도: {result1.get('confidence', 'UNKNOWN')}")
        print(f"   📊 처리 시간 (메타): {result1.get('processing_time', 'N/A')}")
        
        # 잠시 대기
        await asyncio.sleep(1)
        
        # 두 번째 실행 (캐시 히트)
        print("\n⚡ 두 번째 실행 (캐시 히트 예상)")
        start_time = time.time()
        
        result2 = await rag_pipeline.answer_with_confidence(
            test_query, k=3, score_threshold=0.1
        )
        
        second_time = time.time() - start_time
        print(f"   ⏱️  시간: {second_time:.3f}초")
        print(f"   📝 답변 길이: {len(result2.get('answer', ''))}")
        print(f"   🎯 신뢰도: {result2.get('confidence', 'UNKNOWN')}")
        print(f"   📊 처리 시간 (메타): {result2.get('processing_time', 'N/A')}")
        
        # 결과 비교
        print(f"\n📊 성능 비교:")
        if first_time > 0:
            improvement = ((first_time - second_time) / first_time) * 100
            speedup = first_time / second_time if second_time > 0 else float('inf')
            
            print(f"   🚀 성능 향상: {improvement:.1f}%")
            print(f"   ⚡ 속도 향상: {speedup:.1f}x")
        
        # 답변 동일성 확인
        answer1 = result1.get('answer', '')
        answer2 = result2.get('answer', '')
        
        if answer1 == answer2:
            print(f"   ✅ 답변 일관성: 동일한 답변")
        else:
            print(f"   ⚠️  답변 일관성: 다른 답변 (캐시 문제 가능)")
            print(f"      첫 번째: {answer1[:50]}...")
            print(f"      두 번째: {answer2[:50]}...")
        
        # 캐시 통계
        cache_stats = get_cache_stats()
        print(f"\n💾 캐시 통계:")
        print(f"   - 타입: {cache_stats.get('type', 'unknown')}")
        print(f"   - 크기: {cache_stats.get('size', 0)}개 항목")
        if cache_stats.get('hit_ratio') is not None:
            print(f"   - 히트율: {cache_stats['hit_ratio']:.1%}")
        
        # 세 번째 실행으로 캐시 안정성 확인
        print(f"\n🔄 세 번째 실행 (캐시 안정성 확인)")
        start_time = time.time()
        
        result3 = await rag_pipeline.answer_with_confidence(
            test_query, k=3, score_threshold=0.1
        )
        
        third_time = time.time() - start_time
        print(f"   ⏱️  시간: {third_time:.3f}초")
        
        # 평균 캐시 히트 시간
        avg_cache_time = (second_time + third_time) / 2
        print(f"   📈 평균 캐시 시간: {avg_cache_time:.3f}초")
        
        print(f"\n✅ 기본 캐싱 테스트 완료!")
        
        return {
            "cache_miss_time": first_time,
            "cache_hit_time": avg_cache_time,
            "improvement": improvement if first_time > 0 else 0,
            "cache_working": answer1 == answer2
        }
        
    except Exception as e:
        print(f"❌ 테스트 오류: {str(e)}")
        import traceback
        traceback.print_exc()
        return None


async def test_different_queries():
    """다양한 쿼리로 캐시 패턴 테스트"""
    print(f"\n🎯 다양한 쿼리 캐시 패턴 테스트")
    print("=" * 40)
    
    try:
        from app.rag.enhanced_rag import rag_pipeline
        
        queries = [
            "SMHACCP 회사 소개",
            "프로젝트 관리 기능",
            "AI 솔루션 특징"
        ]
        
        total_first_run = 0
        total_cache_hit = 0
        
        for i, query in enumerate(queries, 1):
            print(f"\n🧪 쿼리 {i}: {query}")
            
            # 첫 실행
            start = time.time()
            await rag_pipeline.answer_with_confidence(query, k=3)
            first_time = time.time() - start
            total_first_run += first_time
            
            # 캐시된 실행
            start = time.time()
            await rag_pipeline.answer_with_confidence(query, k=3)
            cache_time = time.time() - start
            total_cache_hit += cache_time
            
            improvement = ((first_time - cache_time) / first_time) * 100 if first_time > 0 else 0
            print(f"   첫 실행: {first_time:.3f}초, 캐시: {cache_time:.3f}초, 향상: {improvement:.1f}%")
        
        # 전체 결과
        avg_first = total_first_run / len(queries)
        avg_cache = total_cache_hit / len(queries)
        overall_improvement = ((avg_first - avg_cache) / avg_first) * 100 if avg_first > 0 else 0
        
        print(f"\n📊 전체 평균:")
        print(f"   첫 실행: {avg_first:.3f}초")
        print(f"   캐시 히트: {avg_cache:.3f}초")
        print(f"   전체 향상: {overall_improvement:.1f}%")
        
    except Exception as e:
        print(f"❌ 다양한 쿼리 테스트 오류: {str(e)}")


if __name__ == "__main__":
    async def main():
        result = await test_basic_caching()
        await test_different_queries()
        
        if result and result["cache_working"]:
            print(f"\n🎉 캐싱 시스템이 정상 작동합니다!")
            print(f"💡 예상 성능 향상: {result['improvement']:.0f}%")
        else:
            print(f"\n⚠️  캐싱 시스템 점검이 필요합니다.")
    
    asyncio.run(main())