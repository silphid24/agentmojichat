#!/usr/bin/env python3
"""
벡터 검색 테스트 (LLM 없이)
DOCX 파일 내용이 벡터 DB에 올바르게 저장되었는지 확인
"""

import os
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

def test_vector_search():
    """벡터 검색 테스트"""
    print("🔍 벡터 검색 테스트 (LLM 없이)")
    print("=" * 50)
    
    try:
        # 환경 설정
        from app.core.config import settings
        if settings.openai_api_key:
            os.environ['OPENAI_API_KEY'] = settings.openai_api_key
        elif settings.llm_api_key:
            os.environ['OPENAI_API_KEY'] = settings.llm_api_key
        
        from app.rag.enhanced_rag import rag_pipeline
        
        # 테스트 쿼리들
        test_queries = [
            "MOJI 테스트",
            "테스트 문서",
            "AI 어시스턴트",
            "SMHACCP",
            "회사소개"
        ]
        
        print(f"📊 벡터 DB 상태:")
        stats = rag_pipeline.get_collection_stats()
        print(f"  - 총 문서 수: {stats.get('total_documents', 0)}")
        print(f"  - 임베딩 모델: {stats.get('embedding_model', 'Unknown')}")
        print()
        
        for query in test_queries:
            print(f"🔎 검색어: '{query}'")
            
            try:
                # 직접 벡터 검색 (LLM 없이)
                results = rag_pipeline.vectorstore.similarity_search_with_score(query, k=3)
                
                if results:
                    print(f"  ✅ 발견된 결과: {len(results)}개")
                    for i, (doc, score) in enumerate(results, 1):
                        source = doc.metadata.get('file_name', 'Unknown')
                        content_preview = doc.page_content[:100].replace('\n', ' ')
                        print(f"    {i}. 출처: {source}")
                        print(f"       점수: {score:.4f}")
                        print(f"       내용: {content_preview}...")
                        print()
                else:
                    print(f"  ❌ 결과 없음")
                    
            except Exception as e:
                print(f"  ❌ 검색 오류: {str(e)}")
            
            print("-" * 40)
        
        # DOCX 파일 특별 확인
        print(f"\n📋 DOCX 파일 확인:")
        docx_query = "MOJI 테스트 문서"
        results = rag_pipeline.vectorstore.similarity_search_with_score(docx_query, k=5)
        
        docx_results = []
        for doc, score in results:
            source = doc.metadata.get('source', '')
            if source.endswith('.docx'):
                docx_results.append((doc, score))
        
        if docx_results:
            print(f"  ✅ DOCX 파일에서 발견: {len(docx_results)}개")
            for doc, score in docx_results:
                source = doc.metadata.get('file_name', 'Unknown')
                print(f"    - {source} (점수: {score:.4f})")
                print(f"      내용: {doc.page_content[:200]}...")
                print()
        else:
            print(f"  ⚠️  DOCX 파일에서 직접 매치 없음")
            print(f"  💡 전체 결과:")
            for i, (doc, score) in enumerate(results[:3], 1):
                source = doc.metadata.get('file_name', 'Unknown')
                print(f"    {i}. {source} (점수: {score:.4f})")
        
    except Exception as e:
        print(f"❌ 오류: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_vector_search()