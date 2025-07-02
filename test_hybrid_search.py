#!/usr/bin/env python3
"""
하이브리드 검색 테스트 스크립트
벡터 + 키워드 + BM25 결합 검색의 성능을 기존 검색과 비교
"""

import asyncio
import os
import sys
import time
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

async def test_hybrid_vs_basic_search():
    """하이브리드 검색 vs 기본 검색 비교 테스트"""
    print("🔍 하이브리드 검색 vs 기본 검색 비교 테스트")
    print("=" * 60)
    
    try:
        # 환경 설정
        from app.core.config import settings
        if settings.openai_api_key:
            os.environ['OPENAI_API_KEY'] = settings.openai_api_key
        elif settings.llm_api_key:
            os.environ['OPENAI_API_KEY'] = settings.llm_api_key
        
        from app.rag.enhanced_rag import rag_pipeline, get_hybrid_pipeline
        
        # 하이브리드 파이프라인 초기화
        hybrid_pipeline = get_hybrid_pipeline()
        
        # 테스트 쿼리들
        test_queries = [
            {
                "query": "MOJI AI 에이전트의 주요 기능은 무엇인가요?",
                "description": "일반적인 기능 문의"
            },
            {
                "query": "SMHACCP 회사 소개",
                "description": "회사 정보 검색"
            },
            {
                "query": "프로젝트 관리 플랫폼 기능",
                "description": "특정 기능 검색"
            },
            {
                "query": "식품제조업 문제점과 AI 솔루션",
                "description": "복합 키워드 검색"
            },
            {
                "query": "문서 업로드 방법",
                "description": "절차 관련 검색"
            }
        ]
        
        print(f"📊 테스트 설정:")
        print(f"  - 벡터 DB 문서 수: {rag_pipeline.get_collection_stats().get('total_documents', 0)}개")
        print(f"  - 테스트 쿼리 수: {len(test_queries)}개")
        print()
        
        total_basic_time = 0
        total_hybrid_time = 0
        
        for i, test_case in enumerate(test_queries, 1):
            query = test_case["query"]
            description = test_case["description"]
            
            print(f"🧪 테스트 {i}: {description}")
            print(f"   쿼리: \"{query}\"")
            print()
            
            # === 기본 검색 테스트 ===
            print("   📍 기본 검색 (벡터 유사도)")
            start_time = time.time()
            
            try:
                basic_results = rag_pipeline.vectorstore.similarity_search_with_score(query, k=5)
                basic_time = time.time() - start_time
                total_basic_time += basic_time
                
                print(f"      ⏱️  시간: {basic_time:.3f}초")
                print(f"      📄 결과: {len(basic_results)}개")
                
                if basic_results:
                    print("      📋 상위 3개 결과:")
                    for j, (doc, score) in enumerate(basic_results[:3], 1):
                        source = doc.metadata.get('file_name', 'Unknown')
                        content_preview = doc.page_content[:80].replace('\n', ' ')
                        print(f"         {j}. {source} (점수: {score:.4f})")
                        print(f"            \"{content_preview}...\"")
                
            except Exception as e:
                print(f"      ❌ 오류: {str(e)}")
                basic_time = 0
            
            print()
            
            # === 하이브리드 검색 테스트 ===
            print("   🔍 하이브리드 검색 (벡터+키워드+BM25)")
            start_time = time.time()
            
            try:
                hybrid_docs, hybrid_metadata = await hybrid_pipeline.search_with_hybrid(query, k=5)
                hybrid_time = time.time() - start_time
                total_hybrid_time += hybrid_time
                
                print(f"      ⏱️  시간: {hybrid_time:.3f}초")
                print(f"      📄 결과: {len(hybrid_docs)}개")
                print(f"      🔎 검색 타입: {hybrid_metadata.get('search_type', 'unknown')}")
                print(f"      📊 총 후보: {hybrid_metadata.get('total_results', 0)}개")
                
                if hybrid_docs and hybrid_metadata.get('result_details'):
                    print("      📋 상위 3개 결과:")
                    for j, detail in enumerate(hybrid_metadata['result_details'][:3], 1):
                        source = Path(detail.get('source', 'Unknown')).name
                        combined_score = detail.get('combined_score', 0)
                        breakdown = detail.get('score_breakdown', {})
                        
                        print(f"         {j}. {source} (종합: {combined_score:.4f})")
                        print(f"            벡터: {breakdown.get('vector_score', 0):.3f}, "
                              f"키워드: {breakdown.get('keyword_score', 0):.3f}, "
                              f"BM25: {breakdown.get('bm25_score', 0):.3f}")
                
            except Exception as e:
                print(f"      ❌ 오류: {str(e)}")
                hybrid_time = 0
            
            # 성능 비교
            if basic_time > 0 and hybrid_time > 0:
                improvement = ((basic_time - hybrid_time) / basic_time) * 100
                print(f"   ⚡ 속도 차이: {improvement:+.1f}% ({hybrid_time:.3f}초 vs {basic_time:.3f}초)")
            
            print("-" * 60)
        
        # 전체 성능 요약
        print(f"\n📈 전체 성능 요약:")
        print(f"   🔵 기본 검색 평균: {total_basic_time/len(test_queries):.3f}초")
        print(f"   🟢 하이브리드 검색 평균: {total_hybrid_time/len(test_queries):.3f}초")
        
        if total_basic_time > 0:
            overall_improvement = ((total_basic_time - total_hybrid_time) / total_basic_time) * 100
            print(f"   ⚡ 전체 성능 차이: {overall_improvement:+.1f}%")
        
        # 메모리 사용량 체크
        print(f"\n💾 메모리 사용량:")
        try:
            import psutil
            process = psutil.Process()
            memory_mb = process.memory_info().rss / 1024 / 1024
            print(f"   📊 현재 메모리: {memory_mb:.1f} MB")
        except ImportError:
            print("   ⚠️  psutil 없음 - 메모리 정보 확인 불가")
        
        print(f"\n✅ 하이브리드 검색 테스트 완료!")
        print(f"💡 다음 단계: 웹챗에서 실제 테스트해보세요")
        print(f"   1. uvicorn app.main:app --reload")
        print(f"   2. http://localhost:8000/static/webchat-test.html")
        print(f"   3. RAG 토글 ON 후 위 쿼리들로 테스트")
        
    except Exception as e:
        print(f"❌ 테스트 오류: {str(e)}")
        import traceback
        traceback.print_exc()


async def test_confidence_scoring():
    """신뢰도 평가 시스템 테스트"""
    print(f"\n🎯 신뢰도 평가 시스템 테스트")
    print("=" * 40)
    
    try:
        from app.rag.enhanced_rag import get_hybrid_pipeline
        hybrid_pipeline = get_hybrid_pipeline()
        
        confidence_queries = [
            {
                "query": "MOJI는 무엇인가요?",
                "expected": "HIGH",
                "reason": "명확한 주제"
            },
            {
                "query": "이 회사의 미래 전망은?",
                "expected": "MEDIUM",
                "reason": "추상적 질문"
            },
            {
                "query": "점심 메뉴 추천해주세요",
                "expected": "LOW", 
                "reason": "무관한 질문"
            }
        ]
        
        for i, test_case in enumerate(confidence_queries, 1):
            query = test_case["query"]
            expected = test_case["expected"]
            reason = test_case["reason"]
            
            print(f"🧪 신뢰도 테스트 {i}: {reason}")
            print(f"   쿼리: \"{query}\"")
            print(f"   예상 신뢰도: {expected}")
            
            try:
                result = await hybrid_pipeline.answer_with_hybrid_search(query, k=3)
                actual_confidence = result.get('confidence', 'UNKNOWN')
                reasoning = result.get('reasoning', '')
                
                print(f"   실제 신뢰도: {actual_confidence}")
                print(f"   근거: {reasoning}")
                
                if actual_confidence == expected:
                    print("   ✅ 예상과 일치")
                else:
                    print(f"   ⚠️  예상과 다름 (예상: {expected}, 실제: {actual_confidence})")
                
            except Exception as e:
                print(f"   ❌ 오류: {str(e)}")
            
            print()
        
    except Exception as e:
        print(f"❌ 신뢰도 테스트 오류: {str(e)}")


if __name__ == "__main__":
    async def main():
        await test_hybrid_vs_basic_search()
        await test_confidence_scoring()
    
    asyncio.run(main())