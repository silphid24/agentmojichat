#!/usr/bin/env python3
"""
문서 업로드 도구
data/documents 폴더의 문서를 벡터 DB에 업로드합니다.
"""

import asyncio
import os
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

async def upload_documents():
    """Upload documents to RAG system"""
    try:
        # Set environment variable for OpenAI API key
        from app.core.config import settings
        if settings.llm_api_key:
            os.environ['OPENAI_API_KEY'] = settings.llm_api_key
        
        from app.rag.enhanced_rag import rag_pipeline
        
        print("📁 문서 업로드 시작...")
        print(f"문서 폴더: {rag_pipeline.documents_dir}")
        
        # Create directory if not exists
        rag_pipeline.documents_dir.mkdir(parents=True, exist_ok=True)
        
        # Check for documents
        doc_files = list(rag_pipeline.documents_dir.glob("**/*"))
        doc_files = [f for f in doc_files if f.is_file() and f.suffix in ['.txt', '.md', '.docx']]
        
        if not doc_files:
            print("⚠️  문서가 없습니다!")
            print(f"\n다음 폴더에 문서를 추가하세요: {rag_pipeline.documents_dir}")
            print("지원 형식: .txt, .md, .docx")
            return
        
        print(f"\n발견된 문서: {len(doc_files)}개")
        for f in doc_files:
            print(f"  - {f.name}")
        
        # Load documents
        print("\n🔄 문서 처리 중...")
        result = await rag_pipeline.load_documents()
        
        if result["success"]:
            print(f"\n✅ 업로드 완료!")
            print(f"  - 처리된 파일: {len(result['processed_files'])}개")
            print(f"  - 생성된 청크: {result['total_chunks']}개")
            
            # Show stats
            stats = rag_pipeline.get_collection_stats()
            print(f"\n📊 벡터 DB 통계:")
            print(f"  - 총 문서 수: {stats['total_documents']}")
            print(f"  - 청크 크기: {stats['chunk_size']}")
            print(f"  - 임베딩 모델: {stats['embedding_model']}")
        else:
            print(f"❌ 업로드 실패: {result.get('error', 'Unknown error')}")
            
    except Exception as e:
        print(f"❌ 오류 발생: {str(e)}")
        import traceback
        traceback.print_exc()

async def test_rag_query():
    """Test RAG query"""
    try:
        # Ensure API key is set
        from app.core.config import settings
        if settings.llm_api_key:
            os.environ['OPENAI_API_KEY'] = settings.llm_api_key
            
        from app.rag.enhanced_rag import rag_pipeline
        
        print("\n\n🔍 RAG 테스트 쿼리")
        print("-" * 50)
        
        # Test queries
        test_queries = [
            "프로젝트의 주요 목표는 무엇인가요?",
            "시스템 아키텍처에 대해 설명해주세요",
            "사용된 기술 스택은 무엇인가요?"
        ]
        
        for query in test_queries[:1]:  # Test with first query
            print(f"\n📝 질문: {query}")
            
            result = await rag_pipeline.answer_with_confidence(query, k=3)
            
            print(f"\n💬 답변: {result['answer']}")
            print(f"🎯 신뢰도: {result['confidence']}")
            print(f"💡 근거: {result['reasoning']}")
            
            if result['sources']:
                print(f"\n📚 출처:")
                for source in result['sources']:
                    print(f"  - {os.path.basename(source)}")
            
            if 'search_metadata' in result:
                print(f"\n🔎 검색 메타데이터:")
                print(f"  - 재작성된 쿼리 수: {len(result['search_metadata'].get('rewritten_queries', []))}")
                print(f"  - 검색된 문서 수: {result['search_metadata'].get('total_results', 0)}")
                
    except Exception as e:
        print(f"❌ 테스트 실패: {str(e)}")

async def main():
    """Main function"""
    print("🤖 MOJI RAG 문서 업로드 도구")
    print("=" * 50)
    
    # Upload documents
    await upload_documents()
    
    # Test query
    response = input("\n\nRAG 테스트를 실행하시겠습니까? (y/N): ")
    if response.lower() == 'y':
        await test_rag_query()
    
    print("\n\n💡 웹챗에서 RAG 사용하기:")
    print("1. 서버 실행: uvicorn app.main:app --reload")
    print("2. 웹챗 접속: http://localhost:8000/static/webchat-test.html")
    print("3. 명령어:")
    print("   - /rag-help - 도움말")
    print("   - /rag [질문] - RAG 질의")
    print("   - /rag-stats - 통계 보기")

if __name__ == "__main__":
    asyncio.run(main())