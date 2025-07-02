#!/usr/bin/env python3
"""
MOJI RAG 문서 관리 도구
문서 인덱싱, 검색, 통계, 백업/복원 등 통합 관리 기능을 제공합니다.
"""

import asyncio
import os
import sys
import argparse
import json
import shutil
from pathlib import Path
from typing import List, Dict, Any
from datetime import datetime

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

class DocumentManager:
    """통합 문서 관리 클래스"""
    
    def __init__(self):
        self.setup_env()
        
    def setup_env(self):
        """환경 설정"""
        from app.core.config import settings
        if settings.openai_api_key:
            os.environ['OPENAI_API_KEY'] = settings.openai_api_key
        elif settings.llm_api_key:
            os.environ['OPENAI_API_KEY'] = settings.llm_api_key
    
    async def list_documents(self, folder: str = None, show_metadata: bool = False):
        """문서 목록 조회"""
        try:
            from app.rag.enhanced_rag import rag_pipeline
            
            print("📚 문서 목록")
            print("=" * 50)
            
            # 문서 폴더 확인
            docs_dir = rag_pipeline.documents_dir
            if not docs_dir.exists():
                print(f"❌ 문서 폴더가 존재하지 않음: {docs_dir}")
                return
            
            # 파일 검색
            supported_extensions = {'.txt', '.md', '.docx', '.pdf'}
            search_path = docs_dir / folder if folder else docs_dir
            
            if not search_path.exists():
                print(f"❌ 폴더가 존재하지 않음: {search_path}")
                return
            
            doc_files = []
            for ext in supported_extensions:
                doc_files.extend(search_path.rglob(f"*{ext}"))
            
            doc_files = [f for f in doc_files if f.is_file()]
            doc_files.sort()
            
            if not doc_files:
                print("📭 문서가 없습니다.")
                return
            
            print(f"📊 총 {len(doc_files)}개 문서")
            if folder:
                print(f"📁 폴더: {folder}")
            print()
            
            # 파일 목록 출력
            for i, doc_file in enumerate(doc_files, 1):
                relative_path = doc_file.relative_to(docs_dir)
                file_size = doc_file.stat().st_size
                file_mtime = datetime.fromtimestamp(doc_file.stat().st_mtime)
                
                print(f"{i:3d}. {relative_path}")
                print(f"     📏 크기: {file_size:,} bytes")
                print(f"     📅 수정: {file_mtime.strftime('%Y-%m-%d %H:%M:%S')}")
                
                if show_metadata:
                    # 메타데이터 정보 추가
                    metadata_file = Path("data/.doc_metadata.json")
                    if metadata_file.exists():
                        try:
                            with open(metadata_file, 'r') as f:
                                metadata = json.load(f)
                            
                            file_str = str(doc_file)
                            if file_str in metadata:
                                meta = metadata[file_str]
                                indexed_at = meta.get('indexed_at', 'Unknown')
                                print(f"     🗂️  인덱싱: {indexed_at}")
                        except Exception:
                            pass
                print()
                
        except Exception as e:
            print(f"❌ 오류: {str(e)}")
    
    async def search_documents(self, query: str, max_results: int = 5):
        """문서 검색"""
        try:
            from app.rag.enhanced_rag import rag_pipeline
            
            print(f"🔍 문서 검색: '{query}'")
            print("=" * 50)
            
            result = await rag_pipeline.answer_with_confidence(query, k=max_results)
            
            if not result or not result.get('answer'):
                print("❌ 검색 결과 없음")
                return
            
            print(f"💬 답변:")
            print(f"{result['answer']}")
            print()
            
            print(f"🎯 신뢰도: {result.get('confidence', 'Unknown')}")
            print(f"💡 근거: {result.get('reasoning', 'None')}")
            print()
            
            if result.get('sources'):
                print(f"📚 출처 ({len(result['sources'])}개):")
                for i, source in enumerate(result['sources'], 1):
                    print(f"  {i}. {os.path.basename(source)}")
                print()
            
            if result.get('search_metadata'):
                metadata = result['search_metadata']
                print(f"🔎 검색 메타데이터:")
                print(f"  - 재작성된 쿼리: {len(metadata.get('rewritten_queries', []))}개")
                print(f"  - 검색된 문서: {metadata.get('total_results', 0)}개")
                
        except Exception as e:
            print(f"❌ 검색 오류: {str(e)}")
    
    async def show_stats(self):
        """통계 정보 표시"""
        try:
            from app.rag.enhanced_rag import rag_pipeline
            
            print("📊 RAG 시스템 통계")
            print("=" * 50)
            
            # 기본 통계
            stats = rag_pipeline.get_collection_stats()
            
            print(f"📚 벡터 DB 정보:")
            print(f"  - 총 문서 수: {stats.get('total_documents', 0)}")
            print(f"  - 청크 크기: {stats.get('chunk_size', 0)}")
            print(f"  - 청크 중복: {stats.get('chunk_overlap', 0)}")
            print(f"  - 임베딩 모델: {stats.get('embedding_model', 'Unknown')}")
            print()
            
            # 파일 시스템 통계
            docs_dir = rag_pipeline.documents_dir
            if docs_dir.exists():
                supported_extensions = {'.txt', '.md', '.docx', '.pdf'}
                file_counts = {}
                total_size = 0
                
                for ext in supported_extensions:
                    files = list(docs_dir.rglob(f"*{ext}"))
                    files = [f for f in files if f.is_file()]
                    file_counts[ext] = len(files)
                    
                    for file in files:
                        total_size += file.stat().st_size
                
                print(f"📁 파일 시스템 정보:")
                print(f"  - 문서 폴더: {docs_dir}")
                print(f"  - 총 파일 수: {sum(file_counts.values())}")
                for ext, count in file_counts.items():
                    if count > 0:
                        print(f"  - {ext} 파일: {count}개")
                print(f"  - 총 크기: {total_size:,} bytes ({total_size/1024/1024:.1f} MB)")
                print()
            
            # 메타데이터 정보
            metadata_file = Path("data/.doc_metadata.json")
            if metadata_file.exists():
                try:
                    with open(metadata_file, 'r') as f:
                        metadata = json.load(f)
                    
                    print(f"🗂️  메타데이터 정보:")
                    print(f"  - 추적 중인 파일: {len(metadata)}개")
                    
                    if metadata:
                        indexed_files = [f for f in metadata.values() if 'indexed_at' in f]
                        print(f"  - 인덱싱된 파일: {len(indexed_files)}개")
                        
                        if indexed_files:
                            latest = max(indexed_files, key=lambda x: x['indexed_at'])
                            print(f"  - 최근 인덱싱: {latest['indexed_at']}")
                except Exception:
                    print(f"  - 메타데이터 파일 읽기 오류")
            else:
                print(f"🗂️  메타데이터: 없음")
                
        except Exception as e:
            print(f"❌ 통계 조회 오류: {str(e)}")
    
    async def cleanup_orphaned(self):
        """고아 파일 정리"""
        try:
            from app.rag.enhanced_rag import rag_pipeline
            
            print("🧹 고아 파일 정리")
            print("=" * 50)
            
            # 메타데이터에서 삭제된 파일 찾기
            metadata_file = Path("data/.doc_metadata.json")
            if not metadata_file.exists():
                print("📭 메타데이터 파일이 없습니다.")
                return
            
            with open(metadata_file, 'r') as f:
                metadata = json.load(f)
            
            orphaned_files = []
            for file_path in list(metadata.keys()):
                if not Path(file_path).exists():
                    orphaned_files.append(file_path)
            
            if not orphaned_files:
                print("✅ 고아 파일이 없습니다.")
                return
            
            print(f"🗑️  발견된 고아 파일: {len(orphaned_files)}개")
            for file_path in orphaned_files:
                print(f"  - {file_path}")
            
            response = input(f"\n메타데이터에서 제거하시겠습니까? (y/N): ")
            if response.lower() == 'y':
                for file_path in orphaned_files:
                    del metadata[file_path]
                
                with open(metadata_file, 'w') as f:
                    json.dump(metadata, f, ensure_ascii=False, indent=2)
                
                print(f"✅ {len(orphaned_files)}개 고아 파일이 메타데이터에서 제거되었습니다.")
            else:
                print("❌ 정리가 취소되었습니다.")
                
        except Exception as e:
            print(f"❌ 정리 오류: {str(e)}")
    
    async def backup_index(self, backup_path: str):
        """인덱스 백업"""
        try:
            from app.rag.enhanced_rag import rag_pipeline
            
            print(f"💾 인덱스 백업: {backup_path}")
            print("=" * 50)
            
            backup_dir = Path(backup_path)
            backup_dir.mkdir(parents=True, exist_ok=True)
            
            # 벡터 DB 백업
            vectordb_dir = rag_pipeline.vectordb_dir
            if vectordb_dir.exists():
                backup_vectordb = backup_dir / "vectordb"
                if backup_vectordb.exists():
                    shutil.rmtree(backup_vectordb)
                shutil.copytree(vectordb_dir, backup_vectordb)
                print(f"✅ 벡터 DB 백업됨: {backup_vectordb}")
            
            # 메타데이터 백업
            metadata_file = Path("data/.doc_metadata.json")
            if metadata_file.exists():
                backup_metadata = backup_dir / "doc_metadata.json"
                shutil.copy2(metadata_file, backup_metadata)
                print(f"✅ 메타데이터 백업됨: {backup_metadata}")
            
            # 백업 정보 저장
            backup_info = {
                "created_at": datetime.now().isoformat(),
                "vectordb_included": vectordb_dir.exists(),
                "metadata_included": metadata_file.exists(),
                "stats": rag_pipeline.get_collection_stats()
            }
            
            backup_info_file = backup_dir / "backup_info.json"
            with open(backup_info_file, 'w') as f:
                json.dump(backup_info, f, ensure_ascii=False, indent=2)
            
            print(f"✅ 백업 완료: {backup_dir}")
            
        except Exception as e:
            print(f"❌ 백업 오류: {str(e)}")
    
    async def restore_index(self, backup_path: str):
        """인덱스 복원"""
        try:
            from app.rag.enhanced_rag import rag_pipeline
            
            print(f"📂 인덱스 복원: {backup_path}")
            print("=" * 50)
            
            backup_dir = Path(backup_path)
            if not backup_dir.exists():
                print(f"❌ 백업 디렉토리가 존재하지 않음: {backup_dir}")
                return
            
            # 백업 정보 확인
            backup_info_file = backup_dir / "backup_info.json"
            if backup_info_file.exists():
                with open(backup_info_file, 'r') as f:
                    backup_info = json.load(f)
                print(f"📅 백업 생성일: {backup_info.get('created_at', 'Unknown')}")
                print()
            
            response = input("⚠️  현재 인덱스를 덮어쓰시겠습니까? (y/N): ")
            if response.lower() != 'y':
                print("❌ 복원이 취소되었습니다.")
                return
            
            # 벡터 DB 복원
            backup_vectordb = backup_dir / "vectordb"
            if backup_vectordb.exists():
                vectordb_dir = rag_pipeline.vectordb_dir
                if vectordb_dir.exists():
                    shutil.rmtree(vectordb_dir)
                shutil.copytree(backup_vectordb, vectordb_dir)
                print(f"✅ 벡터 DB 복원됨")
            
            # 메타데이터 복원
            backup_metadata = backup_dir / "doc_metadata.json"
            if backup_metadata.exists():
                metadata_file = Path("data/.doc_metadata.json")
                metadata_file.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(backup_metadata, metadata_file)
                print(f"✅ 메타데이터 복원됨")
            
            print(f"✅ 복원 완료!")
            
        except Exception as e:
            print(f"❌ 복원 오류: {str(e)}")

def parse_arguments():
    """명령행 인수 파싱"""
    parser = argparse.ArgumentParser(
        description="MOJI RAG 문서 관리 도구",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    subparsers = parser.add_subparsers(dest='command', help='사용 가능한 명령어')
    
    # list 명령어
    list_parser = subparsers.add_parser('list', help='문서 목록 조회')
    list_parser.add_argument('--folder', type=str, help='특정 폴더만 조회')
    list_parser.add_argument('--metadata', action='store_true', help='메타데이터 정보 포함')
    
    # search 명령어
    search_parser = subparsers.add_parser('search', help='문서 검색')
    search_parser.add_argument('query', type=str, help='검색 쿼리')
    search_parser.add_argument('--max-results', type=int, default=5, help='최대 결과 수')
    
    # stats 명령어
    subparsers.add_parser('stats', help='통계 정보 표시')
    
    # cleanup 명령어
    subparsers.add_parser('cleanup', help='고아 파일 정리')
    
    # backup 명령어
    backup_parser = subparsers.add_parser('backup', help='인덱스 백업')
    backup_parser.add_argument('path', type=str, help='백업 경로')
    
    # restore 명령어
    restore_parser = subparsers.add_parser('restore', help='인덱스 복원')
    restore_parser.add_argument('path', type=str, help='백업 경로')
    
    return parser.parse_args()

async def main():
    """메인 함수"""
    args = parse_arguments()
    
    if not args.command:
        print("❌ 명령어를 지정해주세요. --help를 참조하세요.")
        return
    
    manager = DocumentManager()
    
    print("🛠️  MOJI RAG 문서 관리 도구")
    print("=" * 50)
    
    try:
        if args.command == 'list':
            await manager.list_documents(args.folder, args.metadata)
        
        elif args.command == 'search':
            await manager.search_documents(args.query, args.max_results)
        
        elif args.command == 'stats':
            await manager.show_stats()
        
        elif args.command == 'cleanup':
            await manager.cleanup_orphaned()
        
        elif args.command == 'backup':
            await manager.backup_index(args.path)
        
        elif args.command == 'restore':
            await manager.restore_index(args.path)
        
        else:
            print(f"❌ 알 수 없는 명령어: {args.command}")
    
    except Exception as e:
        print(f"❌ 오류: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(main())