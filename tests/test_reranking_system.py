#!/usr/bin/env python3
"""
리랭킹 시스템 테스트 스크립트
교차 인코더를 사용한 검색 결과 재정렬의 효과 검증
"""

import asyncio
import os
import sys
import time
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))


async def test_reranking_system():
    """리랭킹 시스템 종합 테스트"""
    print("🔄 리랭킹 시스템 테스트")
    print("=" * 50)

    try:
        # 환경 설정
        from app.core.config import settings

        if settings.openai_api_key:
            os.environ["OPENAI_API_KEY"] = settings.openai_api_key
        elif settings.llm_api_key:
            os.environ["OPENAI_API_KEY"] = settings.llm_api_key

        from app.rag.enhanced_rag import rag_pipeline, get_hybrid_pipeline
        from app.rag.reranker import get_global_reranker

        # 하이브리드/리랭킹 파이프라인 초기화
        hybrid_pipeline = get_hybrid_pipeline()
        reranker = get_global_reranker()

        print("✅ 시스템 초기화 완료")
        print(f"   - 리랭커 모델: {reranker.cross_encoder.model_name}")
        print(f"   - 로컬 모델 사용: {reranker.cross_encoder.use_local_model}")
        print()

        # 테스트 쿼리들 (다양한 난이도와 타입)
        test_queries = [
            {
                "query": "MOJI AI 에이전트의 핵심 기능",
                "description": "명확한 키워드 매칭",
                "expected_improvement": True,
            },
            {
                "query": "프로젝트 관리 시스템의 장점과 특징",
                "description": "복합 개념 검색",
                "expected_improvement": True,
            },
            {
                "query": "식품제조업 문제점 해결 방안",
                "description": "문제-해결 관계 검색",
                "expected_improvement": True,
            },
            {
                "query": "사용자 인터페이스 디자인",
                "description": "일반적인 용어",
                "expected_improvement": False,
            },
            {
                "query": "SMHACCP 회사 연혁과 비전",
                "description": "구체적 정보 검색",
                "expected_improvement": True,
            },
        ]

        print("📊 테스트 설정:")
        print(
            f"   - 벡터 DB 문서 수: {rag_pipeline.get_collection_stats().get('total_documents', 0)}개"
        )
        print(f"   - 테스트 쿼리 수: {len(test_queries)}개")
        print()

        total_improvement_count = 0
        total_rank_changes = []

        for i, test_case in enumerate(test_queries, 1):
            query = test_case["query"]
            description = test_case["description"]
            expected_improvement = test_case["expected_improvement"]

            print(f"🧪 테스트 {i}: {description}")
            print(f'   쿼리: "{query}"')
            print()

            # === 하이브리드 검색 (리랭킹 없음) ===
            print("   📍 하이브리드 검색 (리랭킹 없음)")
            start_time = time.time()

            try:
                no_rerank_docs, no_rerank_meta = (
                    await hybrid_pipeline.search_with_hybrid(
                        query, k=10, use_reranking=False
                    )
                )
                no_rerank_time = time.time() - start_time

                print(f"      ⏱️  시간: {no_rerank_time:.3f}초")
                print(f"      📄 결과: {len(no_rerank_docs)}개")

                # 상위 3개 결과 표시
                if no_rerank_meta.get("result_details"):
                    print("      📋 상위 3개:")
                    for j, detail in enumerate(no_rerank_meta["result_details"][:3], 1):
                        source = Path(detail.get("source", "Unknown")).name
                        score = detail.get("combined_score", 0)
                        print(f"         {j}. {source} (점수: {score:.4f})")

            except Exception as e:
                print(f"      ❌ 오류: {str(e)}")
                continue

            print()

            # === 하이브리드 검색 + 리랭킹 ===
            print("   🔄 하이브리드 검색 + 리랭킹")
            start_time = time.time()

            try:
                rerank_docs, rerank_meta = await hybrid_pipeline.search_with_hybrid(
                    query, k=10, use_reranking=True
                )
                rerank_time = time.time() - start_time

                print(f"      ⏱️  시간: {rerank_time:.3f}초")
                print(f"      📄 결과: {len(rerank_docs)}개")
                print(
                    f"      🔎 검색 타입: {rerank_meta.get('search_type', 'unknown')}"
                )

                # 상위 3개 결과 표시 (리랭킹 정보 포함)
                if rerank_meta.get("result_details"):
                    print("      📋 상위 3개 (리랭킹 후):")
                    rank_changes = []

                    for j, detail in enumerate(rerank_meta["result_details"][:3], 1):
                        source = Path(detail.get("source", "Unknown")).name
                        breakdown = detail.get("score_breakdown", {})

                        final_score = breakdown.get(
                            "final_score", breakdown.get("combined_score", 0)
                        )
                        rerank_score = breakdown.get("rerank_score", 0)
                        rank_change = breakdown.get("rank_change", 0)

                        rank_changes.append(rank_change)

                        change_indicator = ""
                        if rank_change > 0:
                            change_indicator = f" ⬆️+{rank_change}"
                        elif rank_change < 0:
                            change_indicator = f" ⬇️{rank_change}"

                        print(
                            f"         {j}. {source} (점수: {final_score:.4f}){change_indicator}"
                        )
                        if rerank_score > 0:
                            print(f"            리랭크: {rerank_score:.3f}")

                    # 순위 변화 분석
                    positive_changes = sum(1 for change in rank_changes if change > 0)
                    negative_changes = sum(1 for change in rank_changes if change < 0)
                    avg_rank_change = sum(abs(change) for change in rank_changes) / len(
                        rank_changes
                    )

                    print(
                        f"      📈 순위 변화: 상승 {positive_changes}개, 하락 {negative_changes}개"
                    )
                    print(f"      📊 평균 순위 변화: {avg_rank_change:.1f}")

                    total_rank_changes.extend(rank_changes)

                    # 개선 여부 판단
                    improvement_detected = (
                        avg_rank_change > 0.5 or positive_changes > negative_changes
                    )
                    if improvement_detected:
                        total_improvement_count += 1
                        print("      ✅ 리랭킹 효과 감지됨")
                    else:
                        print("      ➖ 리랭킹 효과 미미")

            except Exception as e:
                print(f"      ❌ 오류: {str(e)}")
                continue

            # 성능 비교
            performance_overhead = (
                ((rerank_time - no_rerank_time) / no_rerank_time) * 100
                if no_rerank_time > 0
                else 0
            )
            print(
                f"   ⏱️  성능 오버헤드: +{performance_overhead:.1f}% ({rerank_time:.3f}초 vs {no_rerank_time:.3f}초)"
            )

            # 예상 vs 실제 비교
            if expected_improvement:
                if improvement_detected:
                    print("   🎯 예상대로 개선됨")
                else:
                    print("   ⚠️  예상과 달리 개선 미미")
            else:
                if not improvement_detected:
                    print("   🎯 예상대로 변화 없음")
                else:
                    print("   🌟 예상보다 좋은 개선")

            print("-" * 50)

        # 전체 결과 요약
        print("\n📈 리랭킹 시스템 평가 결과:")
        print(f"   🔄 총 테스트 수: {len(test_queries)}개")
        print(f"   ✅ 개선 감지된 쿼리: {total_improvement_count}개")
        print(
            f"   📊 전체 개선율: {(total_improvement_count / len(test_queries)) * 100:.1f}%"
        )

        if total_rank_changes:
            positive_total = sum(1 for change in total_rank_changes if change > 0)
            negative_total = sum(1 for change in total_rank_changes if change < 0)
            avg_total_change = sum(abs(change) for change in total_rank_changes) / len(
                total_rank_changes
            )

            print("   📈 전체 순위 변화:")
            print(f"      - 상승: {positive_total}개 문서")
            print(f"      - 하락: {negative_total}개 문서")
            print(f"      - 평균 변화: {avg_total_change:.1f}")

        # 시스템 성능 정보
        print("\n💾 시스템 성능:")
        try:
            import psutil

            process = psutil.Process()
            memory_mb = process.memory_info().rss / 1024 / 1024
            print(f"   📊 메모리 사용량: {memory_mb:.1f} MB")
        except ImportError:
            print("   ⚠️  psutil 없음 - 메모리 정보 확인 불가")

        # 권장사항
        print("\n💡 권장사항:")
        if total_improvement_count >= len(test_queries) * 0.6:
            print("   🌟 리랭킹 시스템이 효과적으로 작동하고 있습니다!")
            print("   ✅ 프로덕션 환경에서 리랭킹 활성화 권장")
        elif total_improvement_count >= len(test_queries) * 0.3:
            print("   🔄 리랭킹 시스템이 부분적으로 효과적입니다")
            print("   🎯 특정 쿼리 타입에 대해 선택적 활용 고려")
        else:
            print("   ⚠️  현재 데이터셋에서는 리랭킹 효과가 제한적입니다")
            print("   🔧 모델 튜닝이나 다른 리랭킹 전략 검토 필요")

        print("\n✅ 리랭킹 시스템 테스트 완료!")

    except Exception as e:
        print(f"❌ 테스트 오류: {str(e)}")
        import traceback

        traceback.print_exc()


async def test_reranker_models():
    """다양한 리랭커 모델 성능 비교"""
    print("\n🔬 리랭커 모델 성능 비교")
    print("=" * 40)

    try:
        from app.rag.reranker import CrossEncoderReranker
        from app.rag.enhanced_rag import rag_pipeline

        # 테스트용 간단한 문서와 쿼리
        test_query = "MOJI AI 에이전트 기능"

        # 기본 검색으로 문서 가져오기
        docs_with_scores = rag_pipeline.vectorstore.similarity_search_with_score(
            test_query, k=5
        )
        docs = [doc for doc, _ in docs_with_scores]
        original_scores = [
            1.0 / (1.0 + score) for _, score in docs_with_scores
        ]  # 거리를 유사도로 변환

        if not docs:
            print("⚠️  테스트할 문서가 없습니다")
            return

        print(f"📄 테스트 문서: {len(docs)}개")
        print(f'🔍 테스트 쿼리: "{test_query}"')
        print()

        # 모델별 테스트
        model_configs = [
            {
                "name": "MS Marco MiniLM",
                "model_name": "ms-marco-MiniLM-L-6-v2",
                "description": "경량 영어 특화 모델",
            },
            {
                "name": "Cross-encoder Official",
                "model_name": "cross-encoder/ms-marco-MiniLM-L-6-v2",
                "description": "공식 교차 인코더",
            },
            {
                "name": "Fallback Similarity",
                "model_name": "fallback",
                "description": "텍스트 유사도 폴백",
                "use_local_model": False,
            },
        ]

        for config in model_configs:
            print(f"🧪 테스트: {config['name']}")
            print(f"   설명: {config['description']}")

            try:
                start_time = time.time()

                # 리랭커 생성
                if config.get("use_local_model", True):
                    reranker = CrossEncoderReranker(
                        model_name=config["model_name"], use_local_model=True
                    )
                else:
                    reranker = CrossEncoderReranker(use_local_model=False)

                # 리랭킹 실행
                results = reranker.rerank(test_query, docs, original_scores, top_k=3)
                elapsed_time = time.time() - start_time

                print(f"   ⏱️  처리 시간: {elapsed_time:.3f}초")
                print(f"   📊 결과 수: {len(results)}개")

                if results:
                    print("   📋 상위 3개 결과:")
                    for i, result in enumerate(results[:3], 1):
                        source = Path(
                            result.document.metadata.get("source", "Unknown")
                        ).name
                        change_indicator = ""
                        if result.rank_change > 0:
                            change_indicator = f" ⬆️+{result.rank_change}"
                        elif result.rank_change < 0:
                            change_indicator = f" ⬇️{result.rank_change}"

                        print(f"      {i}. {source}")
                        print(
                            f"         원본: {result.original_score:.3f}, 리랭크: {result.rerank_score:.3f}"
                        )
                        print(
                            f"         최종: {result.combined_score:.3f}{change_indicator}"
                        )

                # 순위 변화 통계
                rank_changes = [r.rank_change for r in results]
                avg_change = (
                    sum(abs(change) for change in rank_changes) / len(rank_changes)
                    if rank_changes
                    else 0
                )
                print(f"   📈 평균 순위 변화: {avg_change:.1f}")

            except Exception as e:
                print(f"   ❌ 오류: {str(e)}")

            print()

    except Exception as e:
        print(f"❌ 모델 비교 테스트 오류: {str(e)}")


if __name__ == "__main__":

    async def main():
        await test_reranking_system()
        await test_reranker_models()

    asyncio.run(main())
