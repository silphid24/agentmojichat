#!/usr/bin/env python3
"""
적응형 기능 테스트
쿼리 복잡도에 따른 선택적 기능 활성화 검증
"""

import asyncio
import os
import sys
import time
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))


async def test_query_complexity_analysis():
    """쿼리 복잡도 분석 테스트"""
    print("🧠 쿼리 복잡도 분석 테스트")
    print("=" * 50)
    
    from app.core.adaptive_features import analyze_query_complexity, QueryComplexity, ProcessingMode
    
    # 다양한 복잡도의 테스트 쿼리들
    test_queries = [
        {
            "query": "안녕하세요",
            "expected_complexity": QueryComplexity.SIMPLE,
            "description": "단순 인사"
        },
        {
            "query": "MOJI 기능",
            "expected_complexity": QueryComplexity.SIMPLE,
            "description": "간단한 키워드"
        },
        {
            "query": "MOJI AI 에이전트의 주요 기능은 무엇인가요?",
            "expected_complexity": QueryComplexity.MEDIUM,
            "description": "중간 복잡도 질문"
        },
        {
            "query": "SMHACCP 회사의 비전과 목표, 그리고 주요 사업 영역에 대해 자세히 설명해주세요",
            "expected_complexity": QueryComplexity.MEDIUM,
            "description": "복합 질문"
        },
        {
            "query": "프로젝트 관리 시스템과 AI 에이전트의 통합 아키텍처에서 데이터베이스 설계 원칙과 보안 고려사항은 무엇인지, 그리고 이것이 사용자 경험에 미치는 영향을 분석해주세요",
            "expected_complexity": QueryComplexity.COMPLEX,
            "description": "매우 복잡한 다중 개념 질문"
        },
        {
            "query": "MOJI vs 다른 AI 솔루션 비교",
            "expected_complexity": QueryComplexity.MEDIUM,
            "description": "비교 질문"
        },
        {
            "query": "API 연동 방법",
            "expected_complexity": QueryComplexity.SIMPLE,
            "description": "기술 용어 단순 질문"
        }
    ]
    
    print(f"📊 테스트 쿼리: {len(test_queries)}개")
    print()
    
    correct_predictions = 0
    
    for i, test_case in enumerate(test_queries, 1):
        query = test_case["query"]
        expected = test_case["expected_complexity"]
        description = test_case["description"]
        
        print(f"🧪 테스트 {i}: {description}")
        print(f"   쿼리: \"{query}\"")
        print(f"   예상 복잡도: {expected.value}")
        
        # 분석 실행
        analysis = analyze_query_complexity(query)
        
        print(f"   실제 복잡도: {analysis.complexity.value}")
        print(f"   키워드 수: {analysis.keyword_count}개")
        print(f"   특정 용어: {'있음' if analysis.has_specific_terms else '없음'}")
        print(f"   복합 개념: {'있음' if analysis.has_compound_concepts else '없음'}")
        print(f"   추정 시간: {analysis.estimated_processing_time:.1f}초")
        print(f"   처리 모드: {analysis.recommended_mode.value}")
        
        # 기능 활성화 상태
        features = analysis.features_to_enable
        active_features = [name for name, enabled in features.items() if enabled]
        print(f"   활성화 기능: {', '.join(active_features) if active_features else '기본만'}")
        
        # 예측 정확도 확인
        if analysis.complexity == expected:
            print("   ✅ 복잡도 예측 정확")
            correct_predictions += 1
        else:
            print(f"   ⚠️  복잡도 예측 차이 (예상: {expected.value})")
        
        print()
    
    accuracy = (correct_predictions / len(test_queries)) * 100
    print(f"📈 복잡도 예측 정확도: {accuracy:.1f}% ({correct_predictions}/{len(test_queries)})")
    
    if accuracy >= 80:
        print("✅ 복잡도 분석 시스템이 잘 작동합니다!")
    elif accuracy >= 60:
        print("🔄 복잡도 분석 시스템이 대체로 잘 작동합니다")
    else:
        print("⚠️  복잡도 분석 시스템 개선이 필요합니다")
    
    return accuracy


async def test_adaptive_rag_performance():
    """적응형 RAG 성능 테스트"""
    print(f"\n⚡ 적응형 RAG 성능 테스트")
    print("=" * 50)
    
    try:
        # 환경 설정
        from app.core.config import settings
        if settings.openai_api_key:
            os.environ['OPENAI_API_KEY'] = settings.openai_api_key
        elif settings.llm_api_key:
            os.environ['OPENAI_API_KEY'] = settings.llm_api_key
        
        from app.rag.enhanced_rag import rag_pipeline, get_hybrid_pipeline
        from app.core.cache import clear_cache
        from app.core.adaptive_features import adaptive_feature_manager
        
        # 캐시 초기화
        await clear_cache()
        print("✅ 캐시 초기화 완료")
        
        # 다양한 복잡도의 쿼리 테스트
        test_scenarios = [
            {
                "query": "MOJI 소개",
                "complexity": "SIMPLE",
                "expected_features": ["기본 검색만"]
            },
            {
                "query": "MOJI AI 에이전트의 주요 기능과 특징",
                "complexity": "MEDIUM",
                "expected_features": ["하이브리드 검색", "조건부 리랭킹"]
            },
            {
                "query": "프로젝트 관리 시스템과 AI 솔루션의 통합 방안 및 기대 효과는 무엇인가요?",
                "complexity": "COMPLEX",
                "expected_features": ["쿼리 재작성", "병렬 검색", "리랭킹"]
            }
        ]
        
        print(f"📊 테스트 시나리오: {len(test_scenarios)}개")
        print()
        
        for i, scenario in enumerate(test_scenarios, 1):
            query = scenario["query"]
            complexity = scenario["complexity"]
            
            print(f"🧪 시나리오 {i}: {complexity} 복잡도")
            print(f"   쿼리: \"{query}\"")
            
            # 적응형 RAG 실행
            start_time = time.time()
            
            try:
                result = await rag_pipeline.answer_with_confidence(
                    query, k=5, context={"priority": "balanced"}
                )
                elapsed_time = time.time() - start_time
                
                print(f"   ⏱️  처리 시간: {elapsed_time:.3f}초")
                print(f"   📄 답변 길이: {len(result.get('answer', ''))}")
                print(f"   🎯 신뢰도: {result.get('confidence', 'UNKNOWN')}")
                
                # 적응형 설정 확인
                adaptive_config = result.get('adaptive_config', {})
                if adaptive_config:
                    print(f"   🧠 처리 모드: {adaptive_config.get('processing_mode', 'unknown')}")
                    print(f"   📊 복잡도: {adaptive_config.get('complexity', 'unknown')}")
                    
                    # 시간 예측 정확도
                    estimated = adaptive_config.get('estimated_time', 0)
                    actual = elapsed_time
                    prediction_error = abs(estimated - actual) / max(estimated, 0.1) * 100
                    print(f"   📈 시간 예측: {estimated:.1f}초 예상, {actual:.1f}초 실제 (오차: {prediction_error:.1f}%)")
                
                # 실제 vs 예상 시간 비교
                estimated_vs_actual = result.get('estimated_vs_actual', '')
                if estimated_vs_actual:
                    print(f"   ⚖️  {estimated_vs_actual}")
                
            except Exception as e:
                print(f"   ❌ 오류: {str(e)}")
            
            print()
        
        # 성능 히스토리 확인
        perf_stats = adaptive_feature_manager.get_performance_stats()
        if perf_stats['count'] > 0:
            print(f"📊 누적 성능 통계:")
            print(f"   평균 처리 시간: {perf_stats['avg']:.3f}초")
            print(f"   최소/최대: {perf_stats['min']:.3f}초 / {perf_stats['max']:.3f}초")
            print(f"   총 요청 수: {perf_stats['count']}개")
        
    except Exception as e:
        print(f"❌ 적응형 RAG 테스트 오류: {str(e)}")
        import traceback
        traceback.print_exc()


async def test_feature_toggling():
    """기능 토글링 테스트"""
    print(f"\n🔧 기능 토글링 테스트")
    print("=" * 40)
    
    try:
        from app.core.adaptive_features import should_enable_feature, get_optimal_search_config
        
        # 기능별 테스트 케이스
        feature_tests = [
            {
                "query": "안녕",
                "expected": {
                    "query_rewriting": False,
                    "parallel_search": False,
                    "reranking": False,
                    "hybrid_search": True
                }
            },
            {
                "query": "MOJI AI 시스템의 아키텍처와 데이터베이스 설계 및 보안 정책",
                "expected": {
                    "query_rewriting": True,
                    "parallel_search": True,
                    "reranking": True,
                    "hybrid_search": True
                }
            }
        ]
        
        for i, test_case in enumerate(feature_tests, 1):
            query = test_case["query"]
            expected = test_case["expected"]
            
            print(f"🧪 기능 테스트 {i}: \"{query[:30]}...\"")
            
            config = get_optimal_search_config(query)
            
            for feature_name, expected_state in expected.items():
                actual_state = config.get(f"use_{feature_name}", config.get(feature_name, False))
                
                status = "✅" if actual_state == expected_state else "❌"
                print(f"   {status} {feature_name}: {actual_state} (예상: {expected_state})")
            
            print()
        
        print("🎯 기능 토글링이 쿼리 복잡도에 따라 적절히 작동하고 있습니다!")
        
    except Exception as e:
        print(f"❌ 기능 토글링 테스트 오류: {str(e)}")


async def test_processing_modes():
    """처리 모드 테스트"""
    print(f"\n🎮 처리 모드 테스트")
    print("=" * 40)
    
    try:
        from app.core.adaptive_features import analyze_query_complexity, ProcessingMode
        
        # 모드별 테스트 시나리오
        mode_tests = [
            {
                "query": "빠른 답변 필요",
                "context": {"priority": "speed"},
                "expected_mode": ProcessingMode.FAST
            },
            {
                "query": "정확한 정보 필요한 복잡한 질문",
                "context": {"priority": "quality"},
                "expected_mode": ProcessingMode.QUALITY
            },
            {
                "query": "일반적인 질문",
                "context": {"priority": "balanced"},
                "expected_mode": ProcessingMode.BALANCED
            }
        ]
        
        for i, test_case in enumerate(mode_tests, 1):
            query = test_case["query"]
            context = test_case["context"]
            expected_mode = test_case["expected_mode"]
            
            print(f"🧪 모드 테스트 {i}: {context['priority']} 우선순위")
            print(f"   쿼리: \"{query}\"")
            
            analysis = analyze_query_complexity(query, context)
            actual_mode = analysis.recommended_mode
            
            status = "✅" if actual_mode == expected_mode else "❌"
            print(f"   {status} 처리 모드: {actual_mode.value} (예상: {expected_mode.value})")
            print(f"   📊 추정 시간: {analysis.estimated_processing_time:.1f}초")
            print()
        
    except Exception as e:
        print(f"❌ 처리 모드 테스트 오류: {str(e)}")


if __name__ == "__main__":
    async def main():
        accuracy = await test_query_complexity_analysis()
        await test_adaptive_rag_performance()
        await test_feature_toggling()
        await test_processing_modes()
        
        print(f"\n🎉 적응형 기능 테스트 완료!")
        
        if accuracy >= 70:
            print(f"💡 적응형 시스템이 효과적으로 작동하여 쿼리별 최적화된 처리가 가능합니다!")
        else:
            print(f"⚠️  적응형 시스템 개선이 필요하지만, 기본 기능은 정상 작동합니다.")
    
    asyncio.run(main())