#!/usr/bin/env python3
"""
MOJI RAG 시스템 상태 점검 도구
RAG 시스템의 전반적인 건강 상태를 진단하고 문제를 감지합니다.
"""

import asyncio
import os
import sys
import json
import time
from pathlib import Path
from typing import List, Dict, Any
from datetime import datetime

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))


class RAGHealthChecker:
    """RAG 시스템 상태 점검 클래스"""

    def __init__(self):
        self.setup_env()
        self.results = {}

    def setup_env(self):
        """환경 설정"""
        from app.core.config import settings

        if settings.openai_api_key:
            os.environ["OPENAI_API_KEY"] = settings.openai_api_key
        elif settings.llm_api_key:
            os.environ["OPENAI_API_KEY"] = settings.llm_api_key

    def print_header(self, title: str):
        """섹션 헤더 출력"""
        print(f"\n{'='*60}")
        print(f"📋 {title}")
        print("=" * 60)

    def print_check(self, name: str, status: str, details: str = ""):
        """체크 결과 출력"""
        status_icon = {"PASS": "✅", "WARN": "⚠️ ", "FAIL": "❌"}.get(status, "❓")

        print(f"{status_icon} {name}: {status}")
        if details:
            print(f"   {details}")

    async def check_environment(self) -> Dict[str, Any]:
        """환경 설정 점검"""
        self.print_header("환경 설정 점검")

        checks = {}

        # API Key 확인
        api_key = os.environ.get("OPENAI_API_KEY")
        if api_key:
            self.print_check("OpenAI API Key", "PASS", f"키 길이: {len(api_key)} 문자")
            checks["api_key"] = True
        else:
            self.print_check("OpenAI API Key", "FAIL", "API 키가 설정되지 않음")
            checks["api_key"] = False

        # 필수 라이브러리 확인
        required_packages = ["langchain", "chromadb", "openai", "sentence_transformers"]

        missing_packages = []
        for package in required_packages:
            try:
                __import__(package)
                self.print_check(f"Package: {package}", "PASS")
            except ImportError:
                self.print_check(f"Package: {package}", "FAIL", "설치되지 않음")
                missing_packages.append(package)

        checks["packages"] = {
            "missing": missing_packages,
            "all_installed": len(missing_packages) == 0,
        }

        return checks

    async def check_directories(self) -> Dict[str, Any]:
        """디렉토리 구조 점검"""
        self.print_header("디렉토리 구조 점검")

        try:
            from app.rag.enhanced_rag import rag_pipeline

            checks = {}

            # 문서 디렉토리
            docs_dir = rag_pipeline.documents_dir
            if docs_dir.exists():
                doc_count = len(list(docs_dir.rglob("*.*")))
                self.print_check(
                    "문서 디렉토리", "PASS", f"{docs_dir} ({doc_count}개 파일)"
                )
                checks["docs_dir"] = {"exists": True, "file_count": doc_count}
            else:
                self.print_check("문서 디렉토리", "WARN", f"{docs_dir} 존재하지 않음")
                checks["docs_dir"] = {"exists": False, "file_count": 0}

            # 벡터 DB 디렉토리
            vectordb_dir = rag_pipeline.vectordb_dir
            if vectordb_dir.exists():
                db_size = sum(
                    f.stat().st_size for f in vectordb_dir.rglob("*") if f.is_file()
                )
                self.print_check(
                    "벡터 DB 디렉토리",
                    "PASS",
                    f"{vectordb_dir} ({db_size/1024/1024:.1f} MB)",
                )
                checks["vectordb_dir"] = {
                    "exists": True,
                    "size_mb": db_size / 1024 / 1024,
                }
            else:
                self.print_check(
                    "벡터 DB 디렉토리", "WARN", f"{vectordb_dir} 존재하지 않음"
                )
                checks["vectordb_dir"] = {"exists": False, "size_mb": 0}

            # 메타데이터 파일
            metadata_file = Path("data/.doc_metadata.json")
            if metadata_file.exists():
                with open(metadata_file, "r") as f:
                    metadata = json.load(f)
                self.print_check(
                    "메타데이터 파일", "PASS", f"{len(metadata)}개 파일 추적 중"
                )
                checks["metadata"] = {"exists": True, "tracked_files": len(metadata)}
            else:
                self.print_check("메타데이터 파일", "WARN", "메타데이터 파일 없음")
                checks["metadata"] = {"exists": False, "tracked_files": 0}

            return checks

        except Exception as e:
            self.print_check("디렉토리 점검", "FAIL", f"오류: {str(e)}")
            return {"error": str(e)}

    async def check_vector_store(self) -> Dict[str, Any]:
        """벡터 스토어 상태 점검"""
        self.print_header("벡터 스토어 상태 점검")

        try:
            from app.rag.enhanced_rag import rag_pipeline

            checks = {}

            # 컬렉션 통계
            stats = rag_pipeline.get_collection_stats()

            if "error" in stats:
                self.print_check("벡터 스토어 접근", "FAIL", stats["error"])
                checks["accessible"] = False
                return checks

            # 문서 수 확인
            doc_count = stats.get("total_documents", 0)
            if doc_count > 0:
                self.print_check("인덱싱된 문서", "PASS", f"{doc_count}개")
                checks["document_count"] = doc_count
            else:
                self.print_check("인덱싱된 문서", "WARN", "인덱싱된 문서 없음")
                checks["document_count"] = 0

            # 설정 정보
            chunk_size = stats.get("chunk_size", 0)
            chunk_overlap = stats.get("chunk_overlap", 0)
            embedding_model = stats.get("embedding_model", "Unknown")
            use_semantic_chunking = stats.get("use_semantic_chunking", False)

            self.print_check(
                "청크 설정", "PASS", f"크기: {chunk_size}, 중복: {chunk_overlap}"
            )
            self.print_check(
                "의미론적 청킹", "PASS" if use_semantic_chunking else "INFO", 
                "활성화됨" if use_semantic_chunking else "비활성화됨"
            )
            self.print_check("임베딩 모델", "PASS", embedding_model)

            checks.update(
                {
                    "accessible": True,
                    "chunk_size": chunk_size,
                    "chunk_overlap": chunk_overlap,
                    "embedding_model": embedding_model,
                    "use_semantic_chunking": use_semantic_chunking,
                }
            )

            return checks

        except Exception as e:
            self.print_check("벡터 스토어 점검", "FAIL", f"오류: {str(e)}")
            return {"error": str(e)}

    async def check_search_performance(self) -> Dict[str, Any]:
        """검색 성능 점검"""
        self.print_header("검색 성능 점검")

        try:
            from app.rag.enhanced_rag import rag_pipeline

            checks = {}
            test_queries = [
                "시스템에 대해 설명해주세요",
                "기능은 무엇인가요",
                "사용법을 알려주세요",
            ]

            performance_results = []

            for query in test_queries:
                start_time = time.time()

                try:
                    # 단순 검색 테스트
                    results = rag_pipeline.vectorstore.similarity_search(query, k=3)
                    search_time = time.time() - start_time

                    result_count = len(results)

                    if search_time < 1.0:
                        status = "PASS"
                    elif search_time < 3.0:
                        status = "WARN"
                    else:
                        status = "FAIL"

                    self.print_check(
                        f"검색: '{query[:20]}...'",
                        status,
                        f"{search_time:.2f}초, {result_count}개 결과",
                    )

                    performance_results.append(
                        {
                            "query": query,
                            "time": search_time,
                            "result_count": result_count,
                            "status": status,
                        }
                    )

                except Exception as e:
                    self.print_check(
                        f"검색: '{query[:20]}...'", "FAIL", f"오류: {str(e)}"
                    )
                    performance_results.append(
                        {"query": query, "error": str(e), "status": "FAIL"}
                    )

            # 전체 성능 평가
            successful_searches = [r for r in performance_results if "time" in r]
            if successful_searches:
                avg_time = sum(r["time"] for r in successful_searches) / len(
                    successful_searches
                )
                checks["average_search_time"] = avg_time
                checks["successful_searches"] = len(successful_searches)
                checks["total_searches"] = len(test_queries)

                self.print_check("전체 검색 성능", "PASS", f"평균 {avg_time:.2f}초")
            else:
                checks["average_search_time"] = None
                checks["successful_searches"] = 0
                checks["total_searches"] = len(test_queries)

                self.print_check("전체 검색 성능", "FAIL", "모든 검색 실패")

            checks["detailed_results"] = performance_results
            return checks

        except Exception as e:
            self.print_check("검색 성능 점검", "FAIL", f"오류: {str(e)}")
            return {"error": str(e)}

    async def check_rag_pipeline(self) -> Dict[str, Any]:
        """RAG 파이프라인 점검"""
        self.print_header("RAG 파이프라인 점검")

        try:
            from app.rag.enhanced_rag import rag_pipeline

            checks = {}

            # 단순 RAG 쿼리 테스트
            test_query = "시스템에 대해 간단히 설명해주세요"

            start_time = time.time()
            # Use the new adaptive method
            result = await rag_pipeline.generate_answer_adaptive(test_query)
            total_time = time.time() - start_time

            if result and result.get("answer"):
                answer_length = len(result["answer"])
                confidence = result.get("confidence", "Unknown")
                source_count = len(result.get("sources", []))

                self.print_check("RAG 쿼리 실행", "PASS", f"{total_time:.2f}초")
                self.print_check("답변 생성", "PASS", f"{answer_length} 문자")
                self.print_check("신뢰도 평가", "PASS", confidence)
                self.print_check("출처 참조", "PASS", f"{source_count}개 출처")

                checks.update(
                    {
                        "query_successful": True,
                        "response_time": total_time,
                        "answer_length": answer_length,
                        "confidence": confidence,
                        "source_count": source_count,
                    }
                )

                # 상세 결과
                if "search_metadata" in result:
                    metadata = result["search_metadata"]
                    rewritten_queries = len(metadata.get("rewritten_queries", []))
                    total_results = metadata.get("total_results", 0)

                    self.print_check(
                        "쿼리 재작성", "PASS", f"{rewritten_queries}개 변형"
                    )
                    self.print_check("검색 결과", "PASS", f"{total_results}개 후보")

                    checks.update(
                        {
                            "rewritten_queries": rewritten_queries,
                            "search_results": total_results,
                        }
                    )

            else:
                self.print_check("RAG 쿼리 실행", "FAIL", "응답 없음")
                checks["query_successful"] = False

                if "error" in result:
                    checks["error"] = result["error"]

            return checks

        except Exception as e:
            self.print_check("RAG 파이프라인 점검", "FAIL", f"오류: {str(e)}")
            return {"error": str(e)}

    async def check_llm_integration(self) -> Dict[str, Any]:
        """LLM 통합 점검"""
        self.print_header("LLM 통합 점검")

        try:
            from app.llm.router import llm_router

            checks = {}

            # LLM 라우터 상태
            current_provider = llm_router.current_provider
            current_model = llm_router.current_model

            self.print_check("LLM 라우터", "PASS", f"프로바이더: {current_provider}")
            self.print_check("현재 모델", "PASS", current_model or "기본값")

            # LangChain 모델 테스트
            try:
                llm = await llm_router.get_langchain_model()
                self.print_check("LangChain 통합", "PASS", "모델 로드됨")
                checks["langchain_model"] = True
            except Exception as e:
                self.print_check("LangChain 통합", "FAIL", f"오류: {str(e)}")
                checks["langchain_model"] = False
                checks["langchain_error"] = str(e)

            checks.update(
                {"current_provider": current_provider, "current_model": current_model}
            )

            return checks

        except Exception as e:
            self.print_check("LLM 통합 점검", "FAIL", f"오류: {str(e)}")
            return {"error": str(e)}

    async def generate_recommendations(self, all_results: Dict[str, Any]) -> List[str]:
        """점검 결과를 바탕으로 추천사항 생성"""
        recommendations = []

        # 환경 설정 문제
        env_checks = all_results.get("environment", {})
        if not env_checks.get("api_key"):
            recommendations.append(
                "🔑 OpenAI API 키를 설정하세요 (.env 파일의 OPENAI_API_KEY)"
            )

        if env_checks.get("packages", {}).get("missing"):
            missing = env_checks["packages"]["missing"]
            recommendations.append(
                f"📦 누락된 패키지를 설치하세요: pip install {' '.join(missing)}"
            )

        # 디렉토리 문제
        dir_checks = all_results.get("directories", {})
        if not dir_checks.get("docs_dir", {}).get("exists"):
            recommendations.append(
                "📁 문서 디렉토리를 생성하고 문서를 추가하세요: mkdir -p data/documents"
            )

        if dir_checks.get("docs_dir", {}).get("file_count", 0) == 0:
            recommendations.append(
                "📄 data/documents/ 폴더에 문서를 추가한 후 python upload_docs.py를 실행하세요"
            )

        # 벡터 스토어 문제
        vector_checks = all_results.get("vector_store", {})
        if vector_checks.get("document_count", 0) == 0:
            recommendations.append("🔄 문서를 인덱싱하세요: python upload_docs.py")

        # 성능 문제
        perf_checks = all_results.get("search_performance", {})
        avg_time = perf_checks.get("average_search_time")
        if avg_time and avg_time > 3.0:  # Updated threshold for quality-first approach
            recommendations.append(
                "⚡ 검색 성능이 느립니다. 하이브리드 검색 가중치나 청크 크기를 조정하세요"
            )
        
        # 의미론적 청킹 권장
        vector_checks = all_results.get("vector_store", {})
        if not vector_checks.get("use_semantic_chunking"):
            recommendations.append(
                "🔧 의미론적 청킹이 비활성화되어 있습니다. 품질 향상을 위해 활성화를 권장합니다"
            )

        # RAG 파이프라인 문제
        rag_checks = all_results.get("rag_pipeline", {})
        if not rag_checks.get("query_successful"):
            recommendations.append(
                "🔧 RAG 파이프라인에 문제가 있습니다. 로그를 확인하세요"
            )

        # LLM 통합 문제
        llm_checks = all_results.get("llm_integration", {})
        if not llm_checks.get("langchain_model"):
            recommendations.append(
                "🤖 LLM 통합에 문제가 있습니다. API 키와 프로바이더 설정을 확인하세요"
            )

        if not recommendations:
            recommendations.append("✅ 모든 시스템이 정상 작동 중입니다!")

        return recommendations

    async def run_full_check(self, verbose: bool = True) -> Dict[str, Any]:
        """전체 상태 점검 실행"""
        if verbose:
            print("🏥 MOJI RAG 시스템 상태 점검")
            print("=" * 60)
            print(f"📅 점검 시작: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

        results = {}

        # 순차적으로 점검 실행
        try:
            results["environment"] = await self.check_environment()
            results["directories"] = await self.check_directories()
            results["vector_store"] = await self.check_vector_store()
            results["search_performance"] = await self.check_search_performance()
            results["rag_pipeline"] = await self.check_rag_pipeline()
            results["llm_integration"] = await self.check_llm_integration()

            # 추천사항 생성
            recommendations = await self.generate_recommendations(results)

            if verbose:
                self.print_header("추천사항")
                for i, rec in enumerate(recommendations, 1):
                    print(f"{i}. {rec}")

                self.print_header("요약")

                # 전체 상태 요약
                total_checks = 0
                passed_checks = 0

                for category, checks in results.items():
                    if isinstance(checks, dict) and "error" not in checks:
                        for key, value in checks.items():
                            if isinstance(value, bool):
                                total_checks += 1
                                if value:
                                    passed_checks += 1

                success_rate = (
                    (passed_checks / total_checks * 100) if total_checks > 0 else 0
                )

                print(
                    f"📊 전체 점검: {passed_checks}/{total_checks} 통과 ({success_rate:.1f}%)"
                )

                if success_rate >= 90:
                    print("🟢 시스템 상태: 우수")
                elif success_rate >= 70:
                    print("🟡 시스템 상태: 양호 (일부 개선 필요)")
                else:
                    print("🔴 시스템 상태: 주의 (즉시 조치 필요)")

                print(f"📅 점검 완료: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

            results["recommendations"] = recommendations
            results["summary"] = {
                "total_checks": total_checks,
                "passed_checks": passed_checks,
                "success_rate": success_rate,
                "timestamp": datetime.now().isoformat(),
            }

        except Exception as e:
            if verbose:
                print(f"❌ 점검 중 오류 발생: {str(e)}")
            results["error"] = str(e)

        return results


async def main():
    """메인 함수"""
    import argparse

    parser = argparse.ArgumentParser(description="MOJI RAG 시스템 상태 점검")
    parser.add_argument("--json", action="store_true", help="JSON 형식으로 결과 출력")
    parser.add_argument("--save", type=str, help="결과를 파일로 저장")
    parser.add_argument("--quiet", action="store_true", help="상세 출력 비활성화")

    args = parser.parse_args()

    checker = RAGHealthChecker()

    try:
        results = await checker.run_full_check(verbose=not args.quiet)

        if args.json or args.save:
            output = json.dumps(results, ensure_ascii=False, indent=2)

            if args.save:
                with open(args.save, "w", encoding="utf-8") as f:
                    f.write(output)
                print(f"\n📄 결과가 저장됨: {args.save}")

            if args.json:
                print("\n" + output)

    except Exception as e:
        print(f"❌ 점검 실행 오류: {str(e)}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())
