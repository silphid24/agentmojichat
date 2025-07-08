#!/usr/bin/env python3
"""
벡터 스토어 초기화 및 문서 재인덱싱 도구
모든 기존 인덱스를 삭제하고 문서를 처음부터 다시 인덱싱합니다.

사용법:
  python clear_and_reload_docs.py              # 전체 초기화 및 재인덱싱
  python clear_and_reload_docs.py --clear-only # 벡터 스토어만 초기화 (문서 재인덱싱 없이)
"""

import asyncio
import os
import sys
import shutil
import argparse
from pathlib import Path
from typing import List, Dict, Any

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))


async def clear_vector_store(include_faiss: bool = True):
    """벡터 스토어 완전 초기화"""
    try:
        from app.rag.enhanced_rag import rag_pipeline

        print("🗑️  기존 벡터 스토어 삭제 중...")

        # Remove ChromaDB vector database directory
        if rag_pipeline.vectordb_dir.exists():
            shutil.rmtree(rag_pipeline.vectordb_dir)
            print(f"   ✅ ChromaDB 삭제됨: {rag_pipeline.vectordb_dir}")
        else:
            print(f"   ℹ️  ChromaDB 이미 비어있음: {rag_pipeline.vectordb_dir}")

        # Remove FAISS index if requested
        if include_faiss:
            faiss_index_dir = rag_pipeline.vectordb_dir.parent / "faiss_index"
            vector_index_dir = Path("data/vector_index")

            for faiss_dir in [faiss_index_dir, vector_index_dir]:
                if faiss_dir.exists():
                    shutil.rmtree(faiss_dir)
                    print(f"   ✅ FAISS 인덱스 삭제됨: {faiss_dir}")
                else:
                    print(f"   ℹ️  FAISS 인덱스 이미 비어있음: {faiss_dir}")

        # Recreate ChromaDB directory
        rag_pipeline.vectordb_dir.mkdir(parents=True, exist_ok=True)
        print(f"   ✅ ChromaDB 디렉토리 재생성: {rag_pipeline.vectordb_dir}")

        return True

    except Exception as e:
        print(f"❌ 벡터 스토어 삭제 실패: {str(e)}")
        return False


async def scan_documents() -> List[Path]:
    """문서 폴더 스캔"""
    try:
        from app.rag.enhanced_rag import rag_pipeline

        print(f"📁 문서 폴더 스캔: {rag_pipeline.documents_dir}")

        # Create directory if not exists
        rag_pipeline.documents_dir.mkdir(parents=True, exist_ok=True)

        # Supported file extensions
        supported_extensions = {".txt", ".md", ".docx", ".pdf"}

        # Find all documents recursively
        doc_files = []
        for ext in supported_extensions:
            doc_files.extend(rag_pipeline.documents_dir.rglob(f"*{ext}"))

        # Filter out non-files
        doc_files = [f for f in doc_files if f.is_file()]

        print(f"   📊 발견된 문서: {len(doc_files)}개")

        # Group by extension
        by_ext = {}
        for doc in doc_files:
            ext = doc.suffix
            if ext not in by_ext:
                by_ext[ext] = []
            by_ext[ext].append(doc)

        for ext, files in by_ext.items():
            print(f"   {ext}: {len(files)}개")

        return doc_files

    except Exception as e:
        print(f"❌ 문서 스캔 실패: {str(e)}")
        return []


async def reload_all_documents(doc_files: List[Path]) -> Dict[str, Any]:
    """모든 문서 재인덱싱"""
    try:
        # Set environment variable for OpenAI API key
        from app.core.config import settings

        if settings.openai_api_key:
            os.environ["OPENAI_API_KEY"] = settings.openai_api_key
        elif settings.llm_api_key:
            os.environ["OPENAI_API_KEY"] = settings.llm_api_key

        from app.rag.enhanced_rag import rag_pipeline

        print("\n🔄 문서 재인덱싱 시작...")
        print(f"   📁 총 {len(doc_files)}개 파일 처리 예정")

        # Progress tracking
        processed_count = 0
        error_count = 0
        total_chunks = 0

        # Process documents
        result = await rag_pipeline.load_documents()

        if result["success"]:
            processed_count = len(result.get("processed_files", []))
            total_chunks = result.get("total_chunks", 0)

            print("\n✅ 재인덱싱 완료!")
            print(f"   📄 처리된 파일: {processed_count}개")
            print(f"   📝 생성된 청크: {total_chunks}개")

            if result.get("errors"):
                error_count = len(result["errors"])
                print(f"   ⚠️  오류 발생: {error_count}개")
                for error in result["errors"][:3]:  # Show first 3 errors
                    print(f"      - {error}")
                if len(result["errors"]) > 3:
                    print(f"      ... 및 {len(result['errors']) - 3}개 더")
        else:
            print(f"❌ 재인덱싱 실패: {result.get('error', 'Unknown error')}")
            return result

        # Get final stats
        stats = rag_pipeline.get_collection_stats()
        print("\n📊 최종 통계:")
        print(f"   📚 총 문서 수: {stats.get('total_documents', 0)}")
        print(f"   📦 청크 크기: {stats.get('chunk_size', 0)}")
        print(f"   🔄 청크 중복: {stats.get('chunk_overlap', 0)}")
        print(f"   🧩 의미론적 청킹: {'활성화' if stats.get('use_semantic_chunking') else '비활성화'}")
        print(f"   🤖 임베딩 모델: {stats.get('embedding_model', 'Unknown')}")

        return {
            "success": True,
            "processed_files": processed_count,
            "total_chunks": total_chunks,
            "errors": error_count,
            "stats": stats,
        }

    except Exception as e:
        print(f"❌ 재인덱싱 실패: {str(e)}")
        import traceback

        traceback.print_exc()
        return {"success": False, "error": str(e)}


async def verify_indexing():
    """인덱싱 결과 검증"""
    try:
        from app.rag.enhanced_rag import rag_pipeline

        print("\n🔍 인덱싱 결과 검증...")

        # Test query
        test_query = "시스템에 대해 설명해주세요"

        print(f"   📝 테스트 쿼리: {test_query}")

        result = await rag_pipeline.answer_with_confidence(test_query, k=3)

        if result and result.get("answer"):
            print("   ✅ 쿼리 성공")
            print(f"   🎯 신뢰도: {result.get('confidence', 'Unknown')}")
            print(f"   📚 출처: {len(result.get('sources', []))}개")

            if result.get("sources"):
                print("   📄 출처 파일:")
                for source in result["sources"][:3]:
                    print(f"      - {os.path.basename(source)}")
        else:
            print("   ⚠️  쿼리 응답 없음 (문서가 없거나 관련성 낮음)")

        return True

    except Exception as e:
        print(f"❌ 검증 실패: {str(e)}")
        return False


def parse_arguments():
    """명령행 인수 파싱"""
    parser = argparse.ArgumentParser(
        description="벡터 스토어 초기화 및 문서 재인덱싱 도구",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
사용 예시:
  python clear_and_reload_docs.py              # 전체 초기화 및 재인덱싱
  python clear_and_reload_docs.py -c # 벡터 스토어만 초기화
  python clear_and_reload_docs.py -y           # 확인 없이 전체 작업 실행
  python clear_and_reload_docs.py --clear-only -y  # 확인 없이 초기화만 실행
        """,
    )

    parser.add_argument(
        "-c", "--clear-only",
        action="store_true",
        help="벡터 스토어만 초기화하고 문서 재인덱싱은 건너뛰기",
    )

    parser.add_argument(
        "-y", "--yes", action="store_true", help="확인 프롬프트 없이 자동으로 진행"
    )

    parser.add_argument(
        "--no-faiss",
        action="store_true",
        help="FAISS 인덱스는 삭제하지 않고 ChromaDB만 초기화",
    )

    return parser.parse_args()


async def clear_only_mode():
    """벡터 스토어만 초기화하는 모드"""
    print("🧹 MOJI RAG 벡터 스토어 초기화 (초기화만)")
    print("=" * 60)

    try:
        # Get args to check FAISS deletion option
        args = parse_arguments()
        include_faiss = not args.no_faiss

        print("\n📍 벡터 스토어 초기화 시작")
        if not await clear_vector_store(include_faiss=include_faiss):
            print("❌ 벡터 스토어 초기화 실패")
            return False

        print("\n✅ 벡터 스토어 초기화 완료!")
        print("\n💡 다음 단계:")
        print("   1. 문서 재인덱싱: python clear_and_reload_docs.py")
        print("   2. 또는 서버에서 문서 자동 로드 대기")

        return True

    except Exception as e:
        print(f"\n❌ 초기화 중 오류 발생: {str(e)}")
        import traceback

        traceback.print_exc()
        return False


async def main():
    """메인 함수"""
    args = parse_arguments()

    if args.clear_only:
        await clear_only_mode()
        return

    print("🧹 MOJI RAG 벡터 스토어 초기화 및 재인덱싱")
    print("=" * 60)

    # Warning message (skip if --yes flag is used)
    if not args.yes:
        print("⚠️  경고: 이 작업은 기존의 모든 인덱스를 삭제합니다!")
        response = input("계속하시겠습니까? (y/N): ")

        if response.lower() != "y":
            print("❌ 작업이 취소되었습니다.")
            return
    else:
        print("⚡ 자동 실행 모드 (--yes 플래그 사용)")

    start_time = asyncio.get_event_loop().time()

    try:
        # Step 1: Clear vector store
        print("\n📍 1단계: 벡터 스토어 초기화")
        include_faiss = not args.no_faiss
        if not await clear_vector_store(include_faiss=include_faiss):
            print("❌ 벡터 스토어 초기화 실패")
            return

        # Step 2: Scan documents
        print("\n📍 2단계: 문서 스캔")
        doc_files = await scan_documents()

        if not doc_files:
            print("⚠️  인덱싱할 문서가 없습니다!")
            print("   data/documents/ 폴더에 문서를 추가한 후 다시 시도하세요.")
            return

        # Step 3: Reload documents
        print("\n📍 3단계: 문서 재인덱싱")
        result = await reload_all_documents(doc_files)

        if not result.get("success"):
            print("❌ 문서 재인덱싱 실패")
            return

        # Step 4: Verify
        print("\n📍 4단계: 검증")
        await verify_indexing()

        # Summary
        end_time = asyncio.get_event_loop().time()
        duration = end_time - start_time

        print("\n🎉 재인덱싱 완료!")
        print(f"   ⏱️  소요 시간: {duration:.1f}초")
        print(f"   📄 처리된 파일: {result.get('processed_files', 0)}개")
        print(f"   📝 생성된 청크: {result.get('total_chunks', 0)}개")

        print("\n💡 다음 단계:")
        print("   1. 서버 시작: uvicorn app.main:app --reload")
        print("   2. 웹챗 접속: http://localhost:8001/static/moji-webchat-v2.html")
        print("   3. RAG 테스트: 토글을 ON으로 설정하고 질문하기")

    except Exception as e:
        print(f"\n❌ 작업 중 오류 발생: {str(e)}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())
