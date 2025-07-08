#!/usr/bin/env python3
"""
최적화된 하이브리드 검색 테스트 (쿼리 재작성 비활성화)
"""

import asyncio
import os
import sys
import time
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))


async def test_optimized_hybrid():
    """최적화된 하이브리드 검색 테스트"""
    print("⚡ 최적화된 하이브리드 검색 테스트")
    print("=" * 50)

    try:
        # 환경 설정
        from app.core.config import settings

        if settings.openai_api_key:
            os.environ["OPENAI_API_KEY"] = settings.openai_api_key
        elif settings.llm_api_key:
            os.environ["OPENAI_API_KEY"] = settings.llm_api_key

        from app.rag.enhanced_rag import rag_pipeline, get_hybrid_pipeline

        hybrid_pipeline = get_hybrid_pipeline()

        test_queries = [
            "MOJI AI 에이전트 기능",
            "SMHACCP 회사 소개",
            "식품제조업 AI 솔루션",
        ]

        for query in test_queries:
            print(f'🧪 테스트 쿼리: "{query}"')

            # 기본 검색
            start = time.time()
            basic_results = rag_pipeline.vectorstore.similarity_search_with_score(
                query, k=5
            )
            basic_time = time.time() - start

            # 최적화된 하이브리드 검색 (쿼리 재작성 없이)
            start = time.time()
            hybrid_docs, hybrid_meta = await hybrid_pipeline.search_with_hybrid(
                query, k=5, use_query_rewriting=False
            )
            hybrid_time = time.time() - start

            print(f"   📍 기본 검색: {basic_time:.3f}초, {len(basic_results)}개 결과")
            print(f"   🔍 하이브리드: {hybrid_time:.3f}초, {len(hybrid_docs)}개 결과")

            improvement = (
                ((basic_time - hybrid_time) / basic_time) * 100 if basic_time > 0 else 0
            )
            print(f"   ⚡ 성능 차이: {improvement:+.1f}%")

            # 검색 품질 비교
            if hybrid_meta.get("result_details"):
                avg_score = sum(
                    d.get("combined_score", 0) for d in hybrid_meta["result_details"]
                ) / len(hybrid_meta["result_details"])
                print(f"   📊 하이브리드 평균 점수: {avg_score:.3f}")

            print()

        print("✅ 최적화 완료! 이제 훨씬 빨라야 합니다.")

    except Exception as e:
        print(f"❌ 오류: {str(e)}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(test_optimized_hybrid())
