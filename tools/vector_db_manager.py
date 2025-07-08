#!/usr/bin/env python3
"""
벡터 데이터베이스 관리 도구
ChromaDB와 FAISS 벡터 데이터를 쉽게 관리할 수 있는 유틸리티
"""

import asyncio
import os
import sys
import shutil
from pathlib import Path
from typing import Optional, Dict, Any
import argparse

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))


async def clear_chroma_db(persist_dir: str = "data/vectordb") -> bool:
    """ChromaDB 데이터 완전 삭제"""
    try:
        print(f"🗑️  ChromaDB 데이터 삭제 중... ({persist_dir})")

        persist_path = Path(persist_dir)
        if persist_path.exists():
            shutil.rmtree(persist_path)
            print(f"   ✅ 삭제 완료: {persist_path}")
        else:
            print(f"   ℹ️  이미 비어있음: {persist_path}")

        # 디렉토리 재생성
        persist_path.mkdir(parents=True, exist_ok=True)
        os.chmod(str(persist_path), 0o755)
        print(f"   ✅ 디렉토리 재생성: {persist_path}")

        return True

    except Exception as e:
        print(f"❌ ChromaDB 삭제 실패: {str(e)}")
        return False


async def clear_faiss_index(index_dir: str = "data/vector_index") -> bool:
    """FAISS 인덱스 삭제"""
    try:
        print(f"🗑️  FAISS 인덱스 삭제 중... ({index_dir})")

        index_path = Path(index_dir)
        if index_path.exists():
            shutil.rmtree(index_path)
            print(f"   ✅ 삭제 완료: {index_path}")
        else:
            print(f"   ℹ️  이미 비어있음: {index_path}")

        # 디렉토리 재생성
        index_path.mkdir(parents=True, exist_ok=True)
        print(f"   ✅ 디렉토리 재생성: {index_path}")

        return True

    except Exception as e:
        print(f"❌ FAISS 인덱스 삭제 실패: {str(e)}")
        return False


async def get_vector_db_stats() -> Dict[str, Any]:
    """벡터 DB 상태 정보 조회"""
    try:
        from app.rag.enhanced_rag import rag_pipeline

        print("📊 벡터 DB 상태 조회 중...")

        # ChromaDB 상태
        stats = rag_pipeline.get_collection_stats()

        # 디렉토리 크기 계산
        vectordb_path = Path("data/vectordb")
        if vectordb_path.exists():
            total_size = sum(
                f.stat().st_size for f in vectordb_path.rglob("*") if f.is_file()
            )
            stats["vectordb_size_mb"] = round(total_size / (1024 * 1024), 2)
        else:
            stats["vectordb_size_mb"] = 0

        # FAISS 인덱스 상태
        faiss_path = Path("data/vector_index")
        if faiss_path.exists():
            faiss_size = sum(
                f.stat().st_size for f in faiss_path.rglob("*") if f.is_file()
            )
            stats["faiss_size_mb"] = round(faiss_size / (1024 * 1024), 2)
        else:
            stats["faiss_size_mb"] = 0

        # 영구 저장소 여부
        stats["is_persistent"] = getattr(rag_pipeline, "_is_persistent", False)

        return stats

    except Exception as e:
        print(f"❌ 상태 조회 실패: {str(e)}")
        return {"error": str(e)}


async def backup_vector_db(backup_name: Optional[str] = None) -> bool:
    """벡터 DB 백업"""
    try:
        if not backup_name:
            from datetime import datetime

            backup_name = f"vectordb_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        print(f"💾 벡터 DB 백업 중... ({backup_name})")

        backup_dir = Path(f"data/backups/{backup_name}")
        backup_dir.mkdir(parents=True, exist_ok=True)

        # ChromaDB 백업
        vectordb_path = Path("data/vectordb")
        if vectordb_path.exists():
            shutil.copytree(vectordb_path, backup_dir / "vectordb")
            print("   ✅ ChromaDB 백업 완료")

        # FAISS 백업
        faiss_path = Path("data/vector_index")
        if faiss_path.exists():
            shutil.copytree(faiss_path, backup_dir / "vector_index")
            print("   ✅ FAISS 백업 완료")

        print(f"   📁 백업 위치: {backup_dir}")
        return True

    except Exception as e:
        print(f"❌ 백업 실패: {str(e)}")
        return False


async def restore_vector_db(backup_name: str) -> bool:
    """백업에서 벡터 DB 복원"""
    try:
        print(f"♻️  벡터 DB 복원 중... ({backup_name})")

        backup_dir = Path(f"data/backups/{backup_name}")
        if not backup_dir.exists():
            print(f"❌ 백업을 찾을 수 없습니다: {backup_dir}")
            return False

        # 기존 데이터 백업
        await backup_vector_db("before_restore")

        # ChromaDB 복원
        if (backup_dir / "vectordb").exists():
            shutil.rmtree("data/vectordb", ignore_errors=True)
            shutil.copytree(backup_dir / "vectordb", "data/vectordb")
            print("   ✅ ChromaDB 복원 완료")

        # FAISS 복원
        if (backup_dir / "vector_index").exists():
            shutil.rmtree("data/vector_index", ignore_errors=True)
            shutil.copytree(backup_dir / "vector_index", "data/vector_index")
            print("   ✅ FAISS 복원 완료")

        return True

    except Exception as e:
        print(f"❌ 복원 실패: {str(e)}")
        return False


async def list_backups():
    """백업 목록 조회"""
    try:
        backup_dir = Path("data/backups")
        if not backup_dir.exists():
            print("📁 백업 파일이 없습니다.")
            return

        backups = [d for d in backup_dir.iterdir() if d.is_dir()]
        if not backups:
            print("📁 백업 파일이 없습니다.")
            return

        print("📁 백업 목록:")
        for backup in sorted(backups):
            # 백업 크기 계산
            total_size = sum(f.stat().st_size for f in backup.rglob("*") if f.is_file())
            size_mb = round(total_size / (1024 * 1024), 2)

            print(f"   - {backup.name} ({size_mb} MB)")

    except Exception as e:
        print(f"❌ 백업 목록 조회 실패: {str(e)}")


async def main():
    """메인 함수"""
    parser = argparse.ArgumentParser(description="벡터 데이터베이스 관리 도구")
    parser.add_argument(
        "action",
        choices=[
            "clear",
            "clear-chroma",
            "clear-faiss",
            "stats",
            "backup",
            "restore",
            "list-backups",
        ],
        help="수행할 작업",
    )
    parser.add_argument("--name", help="백업/복원 이름")
    parser.add_argument("--force", action="store_true", help="확인 없이 실행")

    args = parser.parse_args()

    print("🔧 MOJI 벡터 데이터베이스 관리 도구")
    print("=" * 50)

    if args.action == "stats":
        print("\n📊 벡터 DB 상태:")
        stats = await get_vector_db_stats()
        
        # Format specific stats for better readability
        formatted_stats = {}
        for key, value in stats.items():
            if key == "total_documents":
                formatted_stats["총 문서 수"] = f"{value}개"
            elif key == "chunk_size":
                formatted_stats["청크 크기"] = str(value)
            elif key == "chunk_overlap":
                formatted_stats["청크 중복"] = str(value)
            elif key == "use_semantic_chunking":
                formatted_stats["의미론적 청킹"] = "활성화" if value else "비활성화"
            elif key == "embedding_model":
                formatted_stats["임베딩 모델"] = value
            elif key == "vectordb_size_mb":
                formatted_stats["ChromaDB 크기"] = f"{value} MB"
            elif key == "faiss_size_mb":
                formatted_stats["FAISS 크기"] = f"{value} MB"
            elif key == "is_persistent":
                formatted_stats["영구 저장소"] = "활성화" if value else "비활성화"
            else:
                formatted_stats[key] = value
        
        for key, value in formatted_stats.items():
            print(f"   {key}: {value}")

    elif args.action == "list-backups":
        await list_backups()

    elif args.action in ["clear", "clear-chroma", "clear-faiss"]:
        if not args.force:
            if args.action == "clear":
                response = input("⚠️  모든 벡터 데이터를 삭제하시겠습니까? (y/N): ")
            elif args.action == "clear-chroma":
                response = input("⚠️  ChromaDB 데이터를 삭제하시겠습니까? (y/N): ")
            else:
                response = input("⚠️  FAISS 인덱스를 삭제하시겠습니까? (y/N): ")

            if response.lower() != "y":
                print("❌ 작업이 취소되었습니다.")
                return

        success = True
        if args.action in ["clear", "clear-chroma"]:
            success &= await clear_chroma_db()

        if args.action in ["clear", "clear-faiss"]:
            success &= await clear_faiss_index()

        if success:
            print("\n✅ 벡터 데이터 삭제 완료!")
        else:
            print("\n❌ 일부 작업이 실패했습니다.")

    elif args.action == "backup":
        success = await backup_vector_db(args.name)
        if success:
            print("\n✅ 백업 완료!")
        else:
            print("\n❌ 백업 실패!")

    elif args.action == "restore":
        if not args.name:
            print("❌ 복원할 백업 이름을 지정해주세요: --name backup_name")
            return

        success = await restore_vector_db(args.name)
        if success:
            print("\n✅ 복원 완료!")
        else:
            print("\n❌ 복원 실패!")


if __name__ == "__main__":
    asyncio.run(main())
