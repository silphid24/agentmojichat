#!/usr/bin/env python3
"""
문서 업로드 도구
data/documents 폴더의 문서를 벡터 DB에 업로드합니다.

사용법:
    python upload_docs.py                           # 모든 문서 업로드
    python upload_docs.py --folder policies/       # 특정 폴더만 업로드
    python upload_docs.py --file guide.txt         # 특정 파일만 업로드
    python upload_docs.py --incremental            # 증분 업데이트
    python upload_docs.py --batch-size 10          # 배치 크기 설정
"""

import asyncio
import os
import sys
import argparse
import hashlib
import json
from pathlib import Path
from typing import List, Dict
from datetime import datetime

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))


class DocumentMetadata:
    """문서 메타데이터 관리"""

    def __init__(self, metadata_file: Path = None):
        self.metadata_file = metadata_file or Path("data/.doc_metadata.json")
        self.metadata = self.load_metadata()

    def load_metadata(self) -> Dict[str, Dict]:
        """메타데이터 로드"""
        if self.metadata_file.exists():
            try:
                with open(self.metadata_file, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception:
                pass
        return {}

    def save_metadata(self):
        """메타데이터 저장"""
        self.metadata_file.parent.mkdir(parents=True, exist_ok=True)
        with open(self.metadata_file, "w", encoding="utf-8") as f:
            json.dump(self.metadata, f, ensure_ascii=False, indent=2)

    def get_file_hash(self, file_path: Path) -> str:
        """파일 해시 계산"""
        hash_md5 = hashlib.md5()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
        return hash_md5.hexdigest()

    def is_file_changed(self, file_path: Path) -> bool:
        """파일 변경 여부 확인"""
        file_str = str(file_path)
        current_hash = self.get_file_hash(file_path)
        current_mtime = file_path.stat().st_mtime

        if file_str not in self.metadata:
            return True

        stored_data = self.metadata[file_str]
        return (
            stored_data.get("hash") != current_hash
            or stored_data.get("mtime") != current_mtime
        )

    def update_file_metadata(self, file_path: Path):
        """파일 메타데이터 업데이트"""
        file_str = str(file_path)
        self.metadata[file_str] = {
            "hash": self.get_file_hash(file_path),
            "mtime": file_path.stat().st_mtime,
            "size": file_path.stat().st_size,
            "indexed_at": datetime.now().isoformat(),
        }


def get_document_files(
    base_dir: Path, folder: str = None, file_path: str = None
) -> List[Path]:
    """문서 파일 목록 가져오기"""
    supported_extensions = {".txt", ".md", ".docx", ".pdf"}

    if file_path:
        # 특정 파일
        target_file = base_dir / file_path
        if target_file.exists() and target_file.suffix in supported_extensions:
            if not _is_excluded_file(target_file):
                return [target_file]
        else:
            print(f"❌ 파일을 찾을 수 없거나 지원되지 않는 형식: {target_file}")
            return []

    if folder:
        # 특정 폴더
        target_dir = base_dir / folder
        if not target_dir.exists():
            print(f"❌ 폴더를 찾을 수 없음: {target_dir}")
            return []
        search_path = target_dir
    else:
        # 전체 문서 폴더
        search_path = base_dir

    # 재귀적으로 파일 검색
    doc_files = []
    for ext in supported_extensions:
        doc_files.extend(search_path.rglob(f"*{ext}"))

    # 파일 필터링 (존재하고, 제외 목록에 없는 것만)
    filtered_files = []
    for f in doc_files:
        if f.is_file() and not _is_excluded_file(f):
            # 파일 크기 체크 (너무 작거나 큰 파일 제외)
            try:
                file_size = f.stat().st_size
                if file_size < 10:  # 10바이트 미만 파일 제외
                    print(f"⚠️  파일이 너무 작음 (제외): {f.name} ({file_size} bytes)")
                    continue
                if file_size > 10 * 1024 * 1024:  # 10MB 초과 파일 제외
                    print(
                        f"⚠️  파일이 너무 큼 (제외): {f.name} ({file_size / 1024 / 1024:.1f} MB)"
                    )
                    continue
                filtered_files.append(f)
            except Exception as e:
                print(f"⚠️  파일 접근 오류 (제외): {f.name} - {e}")
                continue

    return filtered_files


def _is_excluded_file(file_path: Path) -> bool:
    """제외할 파일인지 확인"""
    filename = file_path.name

    # Word 임시 파일 제외
    if filename.startswith("~$"):
        return True

    # 숨김 파일 제외
    if filename.startswith("."):
        return True

    # 백업 파일 제외
    if filename.endswith(".bak") or filename.endswith(".backup"):
        return True

    # 빈 파일명 제외
    if not filename.strip():
        return True

    return False


async def upload_documents(
    folder: str = None,
    file_path: str = None,
    incremental: bool = False,
    batch_size: int = 5,
    force: bool = False,
):
    """문서 업로드 (고급 옵션 지원)"""
    try:
        # Set environment variable for OpenAI API key
        from app.core.config import settings

        if settings.openai_api_key:
            os.environ["OPENAI_API_KEY"] = settings.openai_api_key
        elif settings.llm_api_key:
            os.environ["OPENAI_API_KEY"] = settings.llm_api_key

        from app.rag.enhanced_rag import rag_pipeline

        print("📁 문서 업로드 시작...")
        print(f"문서 폴더: {rag_pipeline.documents_dir}")

        # Create directory if not exists
        rag_pipeline.documents_dir.mkdir(parents=True, exist_ok=True)

        # Get document files
        doc_files = get_document_files(rag_pipeline.documents_dir, folder, file_path)

        if not doc_files:
            print("⚠️  문서가 없습니다!")
            if folder:
                print(f"폴더 '{folder}'에서 문서를 찾을 수 없습니다.")
            elif file_path:
                print(f"파일 '{file_path}'를 찾을 수 없습니다.")
            else:
                print(f"다음 폴더에 문서를 추가하세요: {rag_pipeline.documents_dir}")
            print("지원 형식: .txt, .md, .docx, .pdf")
            return

        # 증분 업데이트 처리
        metadata_manager = DocumentMetadata()
        files_to_process = doc_files

        if incremental and not force:
            print("🔍 변경된 파일 확인 중...")
            changed_files = []
            unchanged_files = []

            for doc_file in doc_files:
                if metadata_manager.is_file_changed(doc_file):
                    changed_files.append(doc_file)
                else:
                    unchanged_files.append(doc_file)

            files_to_process = changed_files

            print(f"   📊 전체 파일: {len(doc_files)}개")
            print(f"   🔄 변경된 파일: {len(changed_files)}개")
            print(f"   ✅ 변경 없는 파일: {len(unchanged_files)}개")

            if not changed_files:
                print("✅ 변경된 파일이 없습니다. 업로드를 건너뜁니다.")
                return

            print("\n변경된 파일 목록:")
            for f in changed_files:
                relative_path = f.relative_to(rag_pipeline.documents_dir)
                print(f"  - {relative_path}")
        else:
            print(f"\n발견된 문서: {len(doc_files)}개")
            for f in doc_files[:10]:  # Show first 10
                relative_path = f.relative_to(rag_pipeline.documents_dir)
                print(f"  - {relative_path}")
            if len(doc_files) > 10:
                print(f"  ... 및 {len(doc_files) - 10}개 더")

        # 배치 처리
        print(f"\n🔄 문서 처리 중... (배치 크기: {batch_size})")

        # For now, process all at once (could be enhanced for true batch processing)
        result = await rag_pipeline.load_documents()

        if result["success"]:
            print("\n✅ 업로드 완료!")
            print(f"  - 처리된 파일: {len(result['processed_files'])}개")
            print(f"  - 생성된 청크: {result['total_chunks']}개")

            # 메타데이터 업데이트
            if incremental:
                print("📝 메타데이터 업데이트 중...")
                for doc_file in files_to_process:
                    metadata_manager.update_file_metadata(doc_file)
                metadata_manager.save_metadata()
                print("   ✅ 메타데이터 저장됨")

            # Show stats
            stats = rag_pipeline.get_collection_stats()
            print("\n📊 벡터 DB 통계:")
            print(f"  - 총 문서 수: {stats['total_documents']}")
            print(f"  - 청크 크기: {stats['chunk_size']}")
            print(f"  - 청크 중복: {stats.get('chunk_overlap', 'N/A')}")
            print(f"  - 의미론적 청킹: {'활성화' if stats.get('use_semantic_chunking') else '비활성화'}")
            print(f"  - 임베딩 모델: {stats['embedding_model']}")

            if result.get("errors"):
                print(f"\n⚠️  오류 발생: {len(result['errors'])}개")
                for error in result["errors"][:3]:
                    print(f"  - {error}")
                if len(result["errors"]) > 3:
                    print(f"  ... 및 {len(result['errors']) - 3}개 더")
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
            os.environ["OPENAI_API_KEY"] = settings.llm_api_key

        from app.rag.enhanced_rag import rag_pipeline

        print("\n\n🔍 RAG 테스트 쿼리")
        print("-" * 50)

        # Test queries
        test_queries = [
            "프로젝트의 주요 목표는 무엇인가요?",
            "시스템 아키텍처에 대해 설명해주세요",
            "사용된 기술 스택은 무엇인가요?",
        ]

        for query in test_queries[:1]:  # Test with first query
            print(f"\n📝 질문: {query}")

            try:
                result = await rag_pipeline.answer_with_confidence(query, k=3)

                if not result or not isinstance(result, dict):
                    print("\n❌ RAG 파이프라인에서 결과를 반환하지 않았습니다.")
                    print(
                        "💡 원인: 벡터 DB가 비어있거나 초기화되지 않았을 수 있습니다."
                    )
                    print(
                        "🔧 해결: python3 clear_and_reload_docs.py 를 실행하여 문서를 다시 인덱싱하세요."
                    )
                    continue

                print(f"\n💬 답변: {result.get('answer', '답변이 없습니다')}")
                print(f"🎯 신뢰도: {result.get('confidence', 'UNKNOWN')}")
                print(f"💡 근거: {result.get('reasoning', '근거 정보 없음')}")

                sources = result.get("sources", [])
                if sources:
                    print("\n📚 출처:")
                    for source in sources:
                        print(f"  - {os.path.basename(source)}")
                else:
                    print("\n📚 출처: 없음 (문서가 인덱싱되지 않았을 수 있습니다)")

                if "search_metadata" in result:
                    search_metadata = result["search_metadata"]
                    print("\n🔎 검색 메타데이터:")
                    print(
                        f"  - 재작성된 쿼리 수: {len(search_metadata.get('rewritten_queries', []))}"
                    )
                    print(
                        f"  - 검색된 문서 수: {search_metadata.get('total_results', 0)}"
                    )
                    if "error" in search_metadata:
                        print(f"  - 검색 오류: {search_metadata['error']}")

            except Exception as e:
                print("\n❌ RAG 테스트 실행 중 오류 발생:")
                print(f"   오류 메시지: {str(e)}")
                print(f"   오류 타입: {type(e).__name__}")

                # 일반적인 해결책 제시
                print("\n🔧 문제 해결 방법:")
                print(
                    "  1. 벡터 DB 초기화: python3 clear_and_reload_docs.py --clear-only -y"
                )
                print("  2. 문서 재인덱싱: python3 clear_and_reload_docs.py -y")
                print("  3. 환경 변수 확인: LLM_API_KEY, OPENAI_API_KEY 설정")

                # 더 자세한 디버깅 정보
                import traceback

                print("\n🐛 상세 오류 정보:")
                traceback.print_exc()

    except Exception as e:
        print(f"❌ 테스트 실패: {str(e)}")


def parse_arguments():
    """명령행 인수 파싱"""
    parser = argparse.ArgumentParser(
        description="MOJI RAG 문서 업로드 도구",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
사용 예시:
  python upload_docs.py                           # 모든 문서 업로드
  python upload_docs.py --folder policies/       # 특정 폴더만 업로드
  python upload_docs.py --file guide.txt         # 특정 파일만 업로드
  python upload_docs.py --incremental            # 증분 업데이트
  python upload_docs.py --batch-size 10          # 배치 크기 설정
  python upload_docs.py --incremental --force    # 강제 전체 재인덱싱
        """,
    )

    parser.add_argument("--folder", type=str, help="특정 폴더만 인덱싱 (예: policies/)")

    parser.add_argument("--file", type=str, help="특정 파일만 인덱싱 (예: guide.txt)")

    parser.add_argument(
        "--incremental", action="store_true", help="증분 업데이트 (변경된 파일만 처리)"
    )

    parser.add_argument(
        "--batch-size", type=int, default=5, help="배치 처리 크기 (기본값: 5)"
    )

    parser.add_argument(
        "--force",
        action="store_true",
        help="강제 실행 (증분 모드에서도 모든 파일 처리)",
    )

    parser.add_argument(
        "--no-test", action="store_true", help="테스트 쿼리 실행하지 않음"
    )

    return parser.parse_args()


async def main():
    """Main function"""
    args = parse_arguments()

    print("🤖 MOJI RAG 문서 업로드 도구")
    print("=" * 50)

    # 설정 정보 출력
    if args.folder:
        print(f"📁 대상 폴더: {args.folder}")
    elif args.file:
        print(f"📄 대상 파일: {args.file}")
    else:
        print("📁 대상: 전체 문서 폴더")

    if args.incremental:
        print("🔄 모드: 증분 업데이트")
        if args.force:
            print("⚡ 강제 모드: 모든 파일 재처리")
    else:
        print("🔄 모드: 전체 업로드")

    print(f"📦 배치 크기: {args.batch_size}")
    print()

    # Upload documents
    await upload_documents(
        folder=args.folder,
        file_path=args.file,
        incremental=args.incremental,
        batch_size=args.batch_size,
        force=args.force,
    )

    # Test query (unless disabled)
    if not args.no_test:
        response = input("\n\nRAG 테스트를 실행하시겠습니까? (y/N): ")
        if response.lower() == "y":
            await test_rag_query()

    print("\n\n💡 웹챗에서 RAG 사용하기:")
    print("1. 서버 실행: uvicorn app.main:app --reload")
    print("2. 웹챗 접속: http://localhost:8000/api/v1/adapters/webchat/page")
    print("3. 명령어:")
    print("   - RAG 토글을 ON으로 설정")
    print("   - 일반 질문 입력")
    print("   - /rag <질문> - RAG 전용 검색")
    print("   - /help - 도움말")
    print("   - /model - 현재 모델 정보")


if __name__ == "__main__":
    asyncio.run(main())
