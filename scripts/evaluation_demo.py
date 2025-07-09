#!/usr/bin/env python3
"""
RAG 평가 시스템 데모 스크립트
실제 인덱싱된 데이터를 기반으로 RAG 시스템 성능을 평가합니다.
"""

import os
import asyncio
import sys
import time
from pathlib import Path
from datetime import datetime

# 환경 변수 설정 - 텔레메트리 및 연결 안정성 개선
os.environ["CHROMA_DISABLE_TELEMETRY"] = "true"
os.environ["CHROMA_TELEMETRY_DISABLED"] = "true"
os.environ["ANONYMIZED_TELEMETRY"] = "false"
os.environ["CHROMA_TELEMETRY"] = "false"
os.environ["CHROMA_CLIENT_TIMEOUT"] = "10"
os.environ["HTTPX_TIMEOUT"] = "30"
os.environ["OPENAI_TIMEOUT"] = "30"

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from app.rag.enhanced_rag import rag_pipeline
from app.evaluation.ragas_evaluator import RAGASEvaluator
from app.evaluation.metrics_dashboard import MetricsDashboard
from app.evaluation.chunk_quality_evaluator import ChunkQualityEvaluator
from app.evaluation.vectordb_performance_evaluator import VectorDBPerformanceEvaluator
from app.core.logging import logger


async def quick_performance_test():
    """빠른 성능 테스트 - 실제 데이터 기반"""
    
    try:
        logger.info("=== 빠른 성능 테스트 시작 ===")
        
        # Vector DB 초기화 확인
        rag_pipeline._initialize_vectorstore()
        
        if not rag_pipeline.vectorstore:
            logger.error("Vector store가 초기화되지 않았습니다.")
            return None
            
        # 성능 테스트용 쿼리
        test_queries = [
            "회사 이름이 무엇인가요?",
            "급여 지급일은 언제인가요?",
            "점심값 지원은 얼마인가요?",
            "회사의 주요 사업은 무엇인가요?",
            "업무 메신저는 무엇을 사용하나요?"
        ]
        
        # 성능 메트릭 수집
        performance_metrics = {
            "query_times": [],
            "total_time": 0,
            "successful_queries": 0,
            "failed_queries": 0
        }
        
        start_time = time.time()
        
        for query in test_queries:
            query_start = time.time()
            try:
                # Use hybrid search for better performance
                from app.rag.hybrid_search import HybridRAGPipeline
                hybrid_pipeline = HybridRAGPipeline(rag_pipeline)
                result = await hybrid_pipeline.answer_with_hybrid_search(query)
                query_time = time.time() - query_start
                
                performance_metrics["query_times"].append(query_time)
                performance_metrics["successful_queries"] += 1
                
                logger.info(f"Query: {query[:30]}... - Time: {query_time:.3f}s")
                
            except Exception as e:
                performance_metrics["failed_queries"] += 1
                logger.error(f"Query failed: {query[:30]}... - Error: {e}")
        
        performance_metrics["total_time"] = time.time() - start_time
        performance_metrics["avg_query_time"] = sum(performance_metrics["query_times"]) / len(performance_metrics["query_times"]) if performance_metrics["query_times"] else 0
        
        # 결과 출력
        logger.info("=== 성능 테스트 결과 ===")
        logger.info(f"총 테스트 시간: {performance_metrics['total_time']:.3f}초")
        logger.info(f"평균 쿼리 응답 시간: {performance_metrics['avg_query_time']:.3f}초")
        logger.info(f"성공 쿼리: {performance_metrics['successful_queries']}/{len(test_queries)}")
        logger.info(f"실패 쿼리: {performance_metrics['failed_queries']}/{len(test_queries)}")
        
        return performance_metrics
        
    except Exception as e:
        logger.error(f"성능 테스트 중 오류 발생: {e}")
        return None


async def run_real_data_evaluation():
    """실제 데이터를 사용한 RAG 평가 시스템 실행 (메인 평가 함수)"""
    
    try:
        logger.info("=== 실제 데이터 기반 RAG 평가 시스템 시작 ===")
        
        # 1. 기존 Vector DB 상태 확인
        logger.info("1. 기존 Vector DB 상태 확인 중...")
        rag_pipeline._initialize_vectorstore()
        
        # Vector store 통계 가져오기
        vectorstore_stats = {}
        total_docs = 0
        
        if rag_pipeline.vectorstore:
            try:
                collection = rag_pipeline.vectorstore._collection
                total_docs = collection.count()
                vectorstore_stats = {
                    "total_documents": total_docs,
                    "collection_name": collection.name if hasattr(collection, 'name') else "unknown"
                }
                logger.info(f"Vector DB 상태: {total_docs}개 문서 인덱싱됨")
            except Exception as e:
                logger.warning(f"Vector DB 상태 확인 실패: {e}")
        
        if total_docs == 0:
            logger.warning("Vector DB에 문서가 없습니다. 먼저 문서를 로드해주세요.")
            logger.info("문서 로드 방법: python upload_docs.py")
            return {"success": False, "error": "No documents in vector DB"}
        
        # 2. 실제 데이터 기반 테스트 쿼리 생성
        logger.info("2. 실제 데이터 기반 테스트 쿼리 준비 중...")
        
        # 다양한 유형의 실제 쿼리들
        real_test_queries = [
            # 회사 정보
            "회사 이름이 무엇인가요?",
            "회사 설립일은 언제인가요?",
            "회사 본사 위치는 어디인가요?",
            "대구 지사 주소를 알려주세요",
            "회사의 주요 사업은 무엇인가요?",
            
            # 기술 관련
            "주요 기술 스택은 무엇인가요?",
            "시스템 아키텍처는 어떻게 구성되어 있나요?",
            "사용하는 데이터베이스는 무엇인가요?",
            "모니터링 도구로는 무엇을 사용하나요?",
            "배포 프로세스는 어떻게 진행되나요?",
            
            # 조직 및 인사
            "DX기획팀 구성원을 알려주세요",
            "AIoT개발팀에는 누가 있나요?",
            "급여 지급일은 언제인가요?",
            "연차는 어떻게 계산되나요?",
            "퇴사 통보는 언제까지 해야 하나요?",
            
            # 복지 및 혜택
            "유연근무제는 어떻게 운영되나요?",
            "점심값 지원은 얼마인가요?",
            "생일축하금은 얼마인가요?",
            "건강검진 지원 내용을 알려주세요",
            "회식비 지원은 어떻게 되나요?",
            
            # 업무 도구 및 프로세스
            "사내 메일 시스템은 무엇을 사용하나요?",
            "업무 메신저는 어떤 것을 사용하나요?",
            "코드 리뷰 프로세스는 어떻게 되나요?",
            "Git 워크플로우는 어떻게 되나요?",
            "테스트 정책은 무엇인가요?",
            
            # 복합 질문
            "신입 개발자가 알아야 할 주요 사항은 무엇인가요?",
            "회사의 개발 문화와 프로세스를 설명해주세요",
            "온보딩 프로세스는 어떻게 진행되나요?",
            "회사 복리후생 제도에 대해 전반적으로 설명해주세요",
            "HACCP 컨설팅과 스마트팩토리 사업에 대해 설명해주세요"
        ]
        
        # 3. RAGAS 평가 실행
        logger.info("3. RAGAS 평가 시스템 초기화 중...")
        evaluator = RAGASEvaluator(
            rag_pipeline=rag_pipeline,
            results_dir="data/evaluation/real_data",
            use_ragas=True
        )
        
        # 평가할 쿼리 수 선택 (성능 고려)
        selected_queries = real_test_queries[:20]  # 상위 20개 쿼리 평가
        
        logger.info(f"4. {len(selected_queries)}개 실제 쿼리로 평가 실행 중...")
        
        # 평가 실행
        evaluation_results_list, evaluation_summary = await evaluator.evaluate_dataset(selected_queries)
        
        # 5. 청킹 품질 평가 (실제 데이터 기반)
        logger.info("5. 실제 데이터 청킹 품질 평가 중...")
        
        chunk_quality_metrics = None
        chunk_quality_report = {}
        
        try:
            # 실제 청크 데이터 가져오기
            chunk_documents = []
            if rag_pipeline.vectorstore and hasattr(rag_pipeline.vectorstore, '_collection'):
                chunks = rag_pipeline.vectorstore._collection.get(include=['documents', 'metadatas'])
                if chunks and chunks.get('documents') and isinstance(chunks['documents'], list):
                    from langchain.schema import Document
                    documents = chunks['documents']
                    metadatas = chunks.get('metadatas', [])
                    
                    # 샘플링 (성능을 위해 일부만 평가)
                    sample_size = min(200, len(documents))
                    import random
                    sample_indices = random.sample(range(len(documents)), sample_size)
                    
                    for idx in sample_indices:
                        if idx < len(documents) and isinstance(documents[idx], str):
                            metadata = metadatas[idx] if metadatas and idx < len(metadatas) else {}
                            chunk_documents.append(Document(page_content=documents[idx], metadata=metadata))
            
            if chunk_documents:
                chunk_evaluator = ChunkQualityEvaluator()
                chunk_quality_metrics = chunk_evaluator.evaluate_chunks(chunks=chunk_documents)
                chunk_quality_report = chunk_evaluator.generate_quality_report(chunk_quality_metrics, chunk_documents)
                logger.info(f"청킹 품질 평가 완료: 전체 품질 점수 {chunk_quality_metrics.overall_quality:.3f}")
            else:
                logger.warning("청크 데이터를 찾을 수 없어 청킹 품질 평가를 건너뜁니다.")
                
        except Exception as e:
            logger.error(f"청킹 품질 평가 중 오류 발생: {e}")
        
        # 6. Vector DB 성능 평가 (실제 데이터 기반)
        logger.info("6. Vector DB 성능 평가 중...")
        
        vectordb_metrics = None
        vectordb_report = {}
        
        try:
            if rag_pipeline.vectorstore:
                # 성능 테스트용 쿼리
                performance_test_queries = [
                    "회사 이름이 무엇인가요?",
                    "주요 기술 스택은 무엇인가요?",
                    "직원 복리후생은 어떻게 되나요?",
                    "개발 가이드의 주요 내용은?",
                    "시스템 아키텍처는 어떻게 구성되어 있나요?",
                    "온보딩 프로세스는 어떻게 되나요?",
                    "테스트 정책은 무엇인가요?",
                    "배포 프로세스는 어떻게 진행되나요?",
                    "모니터링 시스템은 무엇을 사용하나요?",
                    "데이터베이스 구성은 어떻게 되나요?"
                ]
                
                vectordb_evaluator = VectorDBPerformanceEvaluator(
                    vectorstore=rag_pipeline.vectorstore,
                    test_queries=performance_test_queries
                )
                
                vectordb_metrics = vectordb_evaluator.evaluate_performance()
                vectordb_report = vectordb_evaluator.generate_performance_report(vectordb_metrics)
                logger.info(f"Vector DB 성능 평가 완료: 전체 성능 점수 {vectordb_metrics.overall_performance:.3f}")
            else:
                logger.warning("Vector store를 찾을 수 없어 Vector DB 성능 평가를 건너뜁니다.")
                
        except Exception as e:
            logger.error(f"Vector DB 성능 평가 중 오류 발생: {e}")
        
        # 7. 종합 결과 리포트 생성
        logger.info("7. 종합 결과 리포트 생성 중...")
        
        dashboard = MetricsDashboard(results_dir="data/evaluation/real_data")
        
        # 기본 리포트 생성
        report = dashboard.generate_report(evaluation_results_list, evaluation_summary, save_plots=True)
        
        # 종합 HTML 리포트 생성
        enhanced_report = {
            **report,
            "chunk_quality": {
                "metrics": chunk_quality_metrics.to_dict() if chunk_quality_metrics else {},
                "report": chunk_quality_report if chunk_quality_metrics else {},
                "grade": chunk_quality_report.get("summary", {}).get("quality_grade", "Unknown") if chunk_quality_metrics else "N/A"
            },
            "vectordb_performance": {
                "metrics": vectordb_metrics.to_dict() if vectordb_metrics else {},
                "report": vectordb_report,
                "grade": vectordb_report.get("summary", {}).get("performance_grade", "Unknown") if vectordb_metrics else "N/A"
            },
            "evaluation_timestamp": datetime.now().isoformat(),
            "vectorstore_stats": vectorstore_stats
        }
        
        html_report_path = dashboard.create_comprehensive_html_report(
            results=evaluation_results_list,
            summary=evaluation_summary,
            enhanced_report=enhanced_report
        )
        
        # 8. 결과 요약 출력
        logger.info("=== 실제 데이터 기반 RAG 평가 결과 요약 ===")
        
        # RAGAS 평가 결과
        logger.info(f"📊 RAGAS 평가 결과:")
        logger.info(f"  - 총 쿼리 수: {evaluation_summary.total_queries}")
        logger.info(f"  - 평균 신뢰도: {evaluation_summary.avg_faithfulness:.3f}")
        logger.info(f"  - 평균 답변 관련성: {evaluation_summary.avg_answer_relevancy:.3f}")
        logger.info(f"  - 평균 컨텍스트 정밀도: {evaluation_summary.avg_context_precision:.3f}")
        logger.info(f"  - 평균 응답 시간: {evaluation_summary.avg_response_time:.3f}초")
        
        if chunk_quality_metrics:
            logger.info(f"📦 청킹 품질 평가:")
            logger.info(f"  - 의미적 일관성: {chunk_quality_metrics.semantic_coherence:.3f}")
            logger.info(f"  - 경계 품질: {chunk_quality_metrics.boundary_quality:.3f}")
            logger.info(f"  - 정보 커버리지: {chunk_quality_metrics.information_coverage:.3f}")
            logger.info(f"  - 전체 품질: {chunk_quality_metrics.overall_quality:.3f}")
        
        if vectordb_metrics:
            logger.info(f"🗃️ Vector DB 성능:")
            logger.info(f"  - 인덱스 품질: {vectordb_metrics.index_quality:.3f}")
            logger.info(f"  - 검색 정확도: {vectordb_metrics.search_accuracy:.3f}")
            logger.info(f"  - 메모리 효율성: {vectordb_metrics.memory_efficiency:.3f}")
            logger.info(f"  - 전체 성능: {vectordb_metrics.overall_performance:.3f}")
        
        logger.info(f"📄 상세 HTML 리포트: {html_report_path}")
        
        # 9. 실제 데이터 기반 추천사항
        logger.info("=== 실제 데이터 기반 개선 추천사항 ===")
        
        recommendations = []
        
        # RAGAS 결과 기반 추천
        if evaluation_summary.avg_faithfulness < 0.8:
            recommendations.append("📝 답변의 정확성을 높이기 위해 더 정확한 문서 청킹이 필요합니다.")
        if evaluation_summary.avg_answer_relevancy < 0.8:
            recommendations.append("🎯 답변 관련성을 높이기 위해 쿼리 이해도를 개선해야 합니다.")
        if evaluation_summary.avg_context_precision < 0.8:
            recommendations.append("🔍 검색 정확도를 높이기 위해 벡터 검색 알고리즘을 튜닝해야 합니다.")
        if evaluation_summary.avg_response_time > 3.0:
            recommendations.append("⚡ 응답 시간을 단축하기 위해 캐싱 또는 인덱스 최적화가 필요합니다.")
        
        # 청킹 품질 기반 추천
        if chunk_quality_metrics:
            if chunk_quality_metrics.semantic_coherence < 0.8:
                recommendations.append("📚 문서 청킹 시 의미적 일관성을 고려한 청킹 전략이 필요합니다.")
            if chunk_quality_metrics.boundary_quality < 0.8:
                recommendations.append("✂️ 문장 및 단락 경계를 고려한 청킹 개선이 필요합니다.")
        
        # Vector DB 성능 기반 추천
        if vectordb_metrics:
            if vectordb_metrics.search_accuracy < 0.8:
                recommendations.append("🔎 Vector DB 검색 정확도 향상을 위해 임베딩 모델 최적화가 필요합니다.")
            if vectordb_metrics.memory_efficiency < 0.8:
                recommendations.append("💾 메모리 효율성 개선을 위해 인덱스 최적화가 필요합니다.")
            if vectordb_metrics.search_speed < 0.7:
                recommendations.append("🚀 검색 속도 향상을 위해 쿼리 최적화 또는 하드웨어 업그레이드를 고려하세요.")
        
        if recommendations:
            for rec in recommendations:
                logger.info(f"  {rec}")
        else:
            logger.info("  ✅ 모든 지표가 양호합니다! 현재 설정을 유지하세요.")
        
        logger.info("=== 실제 데이터 기반 RAG 평가 완료 ===")
        
        return {
            "success": True,
            "summary": evaluation_summary,
            "evaluation_results": evaluation_results_list,
            "chunk_quality_metrics": chunk_quality_metrics,
            "vectordb_metrics": vectordb_metrics,
            "html_report_path": html_report_path,
            "recommendations": recommendations,
            "vectorstore_stats": vectorstore_stats,
            "enhanced_report": enhanced_report
        }
        
    except Exception as e:
        logger.error(f"실제 데이터 평가 중 오류 발생: {e}")
        import traceback
        traceback.print_exc()
        return {"success": False, "error": str(e)}


async def run_comprehensive_real_evaluation():
    """종합 실제 데이터 평가 - 모든 평가 지표 포함"""
    
    try:
        logger.info("=== 종합 실제 데이터 RAG 평가 시작 ===")
        
        # 1. Vector DB 상태 확인 및 통계
        logger.info("1. Vector DB 상태 확인 중...")
        rag_pipeline._initialize_vectorstore()
        
        vectorstore_stats = {}
        total_docs = 0
        
        if rag_pipeline.vectorstore:
            try:
                collection = rag_pipeline.vectorstore._collection
                total_docs = collection.count()
                
                # 상세 통계 수집
                all_chunks = collection.get(include=['documents', 'metadatas'])
                unique_sources = set()
                
                if all_chunks and all_chunks.get('metadatas'):
                    for metadata in all_chunks['metadatas']:
                        if metadata and metadata.get('source'):
                            unique_sources.add(metadata['source'])
                
                vectorstore_stats = {
                    "total_documents": total_docs,
                    "unique_sources": len(unique_sources),
                    "collection_name": collection.name if hasattr(collection, 'name') else "unknown",
                    "sources": list(unique_sources)[:10]  # 상위 10개만
                }
                
                logger.info(f"Vector DB 상태: {total_docs}개 문서, {len(unique_sources)}개 소스 파일")
                
            except Exception as e:
                logger.warning(f"Vector DB 상태 확인 실패: {e}")
        
        if total_docs == 0:
            logger.error("Vector DB에 문서가 없습니다. 먼저 문서를 로드해주세요.")
            logger.info("문서 로드 방법: python upload_docs.py")
            return {"success": False, "error": "No documents in vector DB"}
        
        # 2. 포괄적인 테스트 쿼리 세트
        logger.info("2. 포괄적인 테스트 쿼리 세트 준비 중...")
        
        comprehensive_test_queries = [
            # 회사 기본 정보
            "회사 이름이 무엇인가요?",
            "회사 설립일은 언제인가요?",
            "회사 본사 위치는 어디인가요?",
            "회사의 주요 사업 분야는 무엇인가요?",
            
            # 기술 및 개발
            "주요 기술 스택은 무엇인가요?",
            "시스템 아키텍처는 어떻게 구성되어 있나요?",
            "사용하는 데이터베이스는 무엇인가요?",
            "코드 리뷰 프로세스는 어떻게 되나요?",
            "Git 워크플로우는 어떻게 되나요?",
            "테스트 커버리지는 몇 퍼센트 이상 유지해야 하나요?",
            "배포 프로세스는 어떻게 진행되나요?",
            "모니터링 도구로는 무엇을 사용하나요?",
            
            # 조직 및 인사
            "DX기획팀 구성원을 알려주세요",
            "AIoT개발팀에는 누가 있나요?",
            "급여 지급일은 언제인가요?",
            "연차는 어떻게 계산되나요?",
            "반차는 몇 시간 단위인가요?",
            "퇴사 통보는 언제까지 해야 하나요?",
            
            # 복지 및 혜택
            "유연근무제는 어떻게 운영되나요?",
            "점심값 지원은 얼마인가요?",
            "생일축하금은 얼마인가요?",
            "건강검진 지원 내용을 알려주세요",
            "회식비 지원은 어떻게 되나요?",
            "연차제도는 어떻게 되나요?",
            
            # 업무 도구 및 프로세스
            "사내 메일 시스템은 무엇을 사용하나요?",
            "업무 메신저는 어떤 것을 사용하나요?",
            "프로젝트 관리 도구는 무엇을 사용하나요?",
            "문서화 도구는 무엇을 사용하나요?",
            "협업 도구는 무엇을 사용하나요?",
            "코드 형상관리는 어떤 도구를 사용하나요?",
            
            # 복합 질문
            "신입 개발자가 알아야 할 주요 사항은 무엇인가요?",
            "회사의 개발 문화와 프로세스를 설명해주세요",
            "온보딩 프로세스는 어떻게 진행되나요?",
            "회사의 복리후생 제도를 종합적으로 설명해주세요",
            "HACCP 컨설팅과 스마트팩토리 사업에 대해 설명해주세요",
            "현재 진행 중인 주요 프로젝트는 무엇인가요?",
            "회사의 비전과 미션은 무엇인가요?",
            "개발자가 지켜야 할 코딩 표준은 무엇인가요?",
            "성과 평가는 어떻게 진행되나요?",
            "교육 및 경력 개발 기회는 어떤 것들이 있나요?"
        ]
        
        # 3. 모든 평가 구성 요소 실행
        logger.info("3. 종합 평가 구성 요소 실행 중...")
        
        # 3-1. RAGAS 평가
        logger.info("3-1. RAGAS 평가 실행 중...")
        evaluator = RAGASEvaluator(
            rag_pipeline=rag_pipeline,
            results_dir="data/evaluation/comprehensive",
            use_ragas=True
        )
        
        # 평가할 쿼리 수 선택 (성능 고려)
        selected_queries = comprehensive_test_queries[:30]  # 상위 30개 쿼리 평가
        
        evaluation_results_list, evaluation_summary = await evaluator.evaluate_dataset(selected_queries)
        
        # 3-2. 청킹 품질 평가
        logger.info("3-2. 청킹 품질 평가 실행 중...")
        chunk_quality_metrics = None
        chunk_quality_report = {}
        
        try:
            chunk_documents = []
            if rag_pipeline.vectorstore and hasattr(rag_pipeline.vectorstore, '_collection'):
                chunks = rag_pipeline.vectorstore._collection.get(include=['documents', 'metadatas'])
                if chunks and chunks.get('documents') and isinstance(chunks['documents'], list):
                    from langchain.schema import Document
                    documents = chunks['documents']
                    metadatas = chunks.get('metadatas', [])
                    
                    # 전체 데이터에서 더 많은 샘플 추출
                    sample_size = min(500, len(documents))
                    import random
                    sample_indices = random.sample(range(len(documents)), sample_size)
                    
                    for idx in sample_indices:
                        if idx < len(documents) and isinstance(documents[idx], str):
                            metadata = metadatas[idx] if metadatas and idx < len(metadatas) else {}
                            chunk_documents.append(Document(page_content=documents[idx], metadata=metadata))
            
            if chunk_documents:
                chunk_evaluator = ChunkQualityEvaluator()
                chunk_quality_metrics = chunk_evaluator.evaluate_chunks(chunks=chunk_documents)
                chunk_quality_report = chunk_evaluator.generate_quality_report(chunk_quality_metrics, chunk_documents)
                logger.info(f"청킹 품질 평가 완료: 전체 품질 점수 {chunk_quality_metrics.overall_quality:.3f}")
                
        except Exception as e:
            logger.error(f"청킹 품질 평가 중 오류 발생: {e}")
        
        # 3-3. Vector DB 성능 평가
        logger.info("3-3. Vector DB 성능 평가 실행 중...")
        vectordb_metrics = None
        vectordb_report = {}
        
        try:
            if rag_pipeline.vectorstore:
                # 다양한 성능 테스트 쿼리
                performance_test_queries = [
                    "회사 이름이 무엇인가요?",
                    "주요 기술 스택은 무엇인가요?",
                    "복리후생 제도를 상세히 설명해주세요",
                    "개발 프로세스와 방법론을 설명해주세요",
                    "시스템 아키텍처의 주요 구성 요소는?",
                    "데이터베이스 설계와 구조를 설명해주세요",
                    "보안 정책과 절차는 어떻게 되나요?",
                    "비상 대응 계획은 어떻게 되나요?",
                    "품질 보증 프로세스는 어떻게 되나요?",
                    "고객 지원 프로세스를 설명해주세요",
                    "성과 측정 지표는 무엇인가요?",
                    "팀 구성과 역할을 설명해주세요",
                    "기술 로드맵은 어떻게 구성되어 있나요?",
                    "파트너십과 협력 관계는 어떻게 되나요?",
                    "회사의 경쟁 우위는 무엇인가요?"
                ]
                
                vectordb_evaluator = VectorDBPerformanceEvaluator(
                    vectorstore=rag_pipeline.vectorstore,
                    test_queries=performance_test_queries
                )
                
                vectordb_metrics = vectordb_evaluator.evaluate_performance()
                vectordb_report = vectordb_evaluator.generate_performance_report(vectordb_metrics)
                logger.info(f"Vector DB 성능 평가 완료: 전체 성능 점수 {vectordb_metrics.overall_performance:.3f}")
                
        except Exception as e:
            logger.error(f"Vector DB 성능 평가 중 오류 발생: {e}")
        
        # 3-4. 하이브리드 검색 성능 테스트
        logger.info("3-4. 하이브리드 검색 성능 테스트 실행 중...")
        hybrid_performance = await test_hybrid_search_performance()
        
        # 4. 종합 결과 리포트 생성
        logger.info("4. 종합 결과 리포트 생성 중...")
        
        dashboard = MetricsDashboard(results_dir="data/evaluation/comprehensive")
        
        # 기본 리포트 생성
        report = dashboard.generate_report(evaluation_results_list, evaluation_summary, save_plots=True)
        
        # 종합 HTML 리포트 생성
        enhanced_report = {
            **report,
            "chunk_quality": {
                "metrics": chunk_quality_metrics.to_dict() if chunk_quality_metrics else {},
                "report": chunk_quality_report if chunk_quality_metrics else {},
                "grade": chunk_quality_report.get("summary", {}).get("quality_grade", "Unknown") if chunk_quality_metrics else "N/A"
            },
            "vectordb_performance": {
                "metrics": vectordb_metrics.to_dict() if vectordb_metrics else {},
                "report": vectordb_report,
                "grade": vectordb_report.get("summary", {}).get("performance_grade", "Unknown") if vectordb_metrics else "N/A"
            },
            "hybrid_search_performance": hybrid_performance,
            "evaluation_timestamp": datetime.now().isoformat(),
            "vectorstore_stats": vectorstore_stats
        }
        
        html_report_path = dashboard.create_comprehensive_html_report(
            results=evaluation_results_list,
            summary=evaluation_summary,
            enhanced_report=enhanced_report
        )
        
        # 5. 결과 요약 출력
        logger.info("=== 종합 실제 데이터 RAG 평가 결과 요약 ===")
        
        # Vector DB 통계
        logger.info(f"🗄️ Vector DB 통계:")
        logger.info(f"  - 총 문서 수: {vectorstore_stats.get('total_documents', 0)}")
        logger.info(f"  - 고유 소스 파일 수: {vectorstore_stats.get('unique_sources', 0)}")
        
        # RAGAS 평가 결과
        logger.info(f"📊 RAGAS 평가 결과:")
        logger.info(f"  - 총 쿼리 수: {evaluation_summary.total_queries}")
        logger.info(f"  - 평균 신뢰도: {evaluation_summary.avg_faithfulness:.3f}")
        logger.info(f"  - 평균 답변 관련성: {evaluation_summary.avg_answer_relevancy:.3f}")
        logger.info(f"  - 평균 컨텍스트 정밀도: {evaluation_summary.avg_context_precision:.3f}")
        logger.info(f"  - 평균 응답 시간: {evaluation_summary.avg_response_time:.3f}초")
        
        if chunk_quality_metrics:
            logger.info(f"📦 청킹 품질 평가:")
            logger.info(f"  - 의미적 일관성: {chunk_quality_metrics.semantic_coherence:.3f}")
            logger.info(f"  - 경계 품질: {chunk_quality_metrics.boundary_quality:.3f}")
            logger.info(f"  - 정보 커버리지: {chunk_quality_metrics.information_coverage:.3f}")
            logger.info(f"  - 전체 품질: {chunk_quality_metrics.overall_quality:.3f}")
            logger.info(f"  - 품질 등급: {chunk_quality_report.get('summary', {}).get('quality_grade', 'N/A')}")
        
        if vectordb_metrics:
            logger.info(f"🗃️ Vector DB 성능:")
            logger.info(f"  - 인덱스 품질: {vectordb_metrics.index_quality:.3f}")
            logger.info(f"  - 검색 정확도: {vectordb_metrics.search_accuracy:.3f}")
            logger.info(f"  - 검색 속도: {vectordb_metrics.search_speed:.3f}")
            logger.info(f"  - 메모리 효율성: {vectordb_metrics.memory_efficiency:.3f}")
            logger.info(f"  - 전체 성능: {vectordb_metrics.overall_performance:.3f}")
            logger.info(f"  - 성능 등급: {vectordb_report.get('summary', {}).get('performance_grade', 'N/A')}")
        
        if hybrid_performance:
            logger.info(f"🔄 하이브리드 검색 성능:")
            logger.info(f"  - 평균 응답 시간: {hybrid_performance.get('avg_query_time', 0):.3f}초")
            logger.info(f"  - 성공률: {hybrid_performance.get('successful_queries', 0)}/{hybrid_performance.get('total_queries', 0)}")
        
        logger.info(f"📄 상세 HTML 리포트: {html_report_path}")
        
        # 6. 종합 개선 추천사항
        recommendations = generate_comprehensive_recommendations(
            evaluation_summary, chunk_quality_metrics, vectordb_metrics, hybrid_performance
        )
        
        if recommendations:
            logger.info("=== 종합 개선 추천사항 ===")
            for i, rec in enumerate(recommendations, 1):
                logger.info(f"{i}. {rec}")
        else:
            logger.info("=== 모든 지표가 우수합니다! ===")
        
        logger.info("=== 종합 실제 데이터 RAG 평가 완료 ===")
        
        return {
            "success": True,
            "summary": evaluation_summary,
            "evaluation_results": evaluation_results_list,
            "chunk_quality_metrics": chunk_quality_metrics,
            "vectordb_metrics": vectordb_metrics,
            "hybrid_performance": hybrid_performance,
            "html_report_path": html_report_path,
            "recommendations": recommendations,
            "vectorstore_stats": vectorstore_stats,
            "enhanced_report": enhanced_report
        }
        
    except Exception as e:
        logger.error(f"종합 실제 데이터 평가 중 오류 발생: {e}")
        import traceback
        traceback.print_exc()
        return {"success": False, "error": str(e)}


async def test_hybrid_search_performance():
    """하이브리드 검색 성능 테스트"""
    try:
        from app.rag.hybrid_search import HybridRAGPipeline
        hybrid_pipeline = HybridRAGPipeline(rag_pipeline)
        
        test_queries = [
            "회사의 주요 기술 스택은 무엇인가요?",
            "직원 복리후생 제도를 상세히 설명해주세요",
            "개발 프로세스와 코드 리뷰 절차를 설명해주세요",
            "온보딩 프로세스는 어떻게 진행되나요?",
            "회사의 비전과 미션은 무엇인가요?"
        ]
        
        performance_metrics = {
            "query_times": [],
            "total_queries": len(test_queries),
            "successful_queries": 0,
            "failed_queries": 0
        }
        
        start_time = time.time()
        
        for query in test_queries:
            query_start = time.time()
            try:
                result = await hybrid_pipeline.answer_with_hybrid_search(query)
                query_time = time.time() - query_start
                
                performance_metrics["query_times"].append(query_time)
                performance_metrics["successful_queries"] += 1
                
            except Exception as e:
                performance_metrics["failed_queries"] += 1
                logger.error(f"Hybrid search failed for query: {query[:50]}... - Error: {e}")
        
        performance_metrics["total_time"] = time.time() - start_time
        performance_metrics["avg_query_time"] = sum(performance_metrics["query_times"]) / len(performance_metrics["query_times"]) if performance_metrics["query_times"] else 0
        
        return performance_metrics
        
    except Exception as e:
        logger.error(f"하이브리드 검색 성능 테스트 중 오류: {e}")
        return None


def generate_comprehensive_recommendations(evaluation_summary, chunk_quality_metrics, vectordb_metrics, hybrid_performance):
    """종합적인 개선 추천사항 생성"""
    recommendations = []
    
    # RAGAS 결과 기반 추천
    if evaluation_summary:
        if evaluation_summary.avg_faithfulness < 0.85:
            recommendations.append("📝 답변의 정확성을 높이기 위해 문서 청킹 전략을 개선하고 컨텍스트 길이를 최적화하세요.")
        if evaluation_summary.avg_answer_relevancy < 0.85:
            recommendations.append("🎯 쿼리 이해도를 높이기 위해 쿼리 재작성(query rewriting) 기능을 개선하세요.")
        if evaluation_summary.avg_context_precision < 0.8:
            recommendations.append("🔍 검색 정확도를 높이기 위해 하이브리드 검색의 가중치를 조정하고 리랭킹 알고리즘을 최적화하세요.")
        if evaluation_summary.avg_response_time > 2.5:
            recommendations.append("⚡ 응답 시간을 단축하기 위해 쿼리 캐싱, 인덱스 최적화, 및 병렬 처리를 강화하세요.")
    
    # 청킹 품질 기반 추천
    if chunk_quality_metrics:
        if chunk_quality_metrics.semantic_coherence < 0.85:
            recommendations.append("📚 의미적 일관성을 높이기 위해 문서의 논리적 구조를 고려한 지능형 청킹 알고리즘을 도입하세요.")
        if chunk_quality_metrics.boundary_quality < 0.8:
            recommendations.append("✂️ 문장 및 단락 경계 인식을 개선하여 더 자연스러운 청크 분할을 구현하세요.")
        if chunk_quality_metrics.size_consistency < 0.7:
            recommendations.append("📏 청크 크기의 일관성을 높이기 위해 동적 청크 크기 조정 알고리즘을 적용하세요.")
    
    # Vector DB 성능 기반 추천
    if vectordb_metrics:
        if vectordb_metrics.search_accuracy < 0.85:
            recommendations.append("🔎 벡터 검색 정확도를 높이기 위해 임베딩 모델을 더 강력한 모델로 업그레이드하거나 파인튜닝하세요.")
        if vectordb_metrics.memory_efficiency < 0.75:
            recommendations.append("💾 메모리 효율성을 개선하기 위해 벡터 차원 축소(PCA, LSH) 또는 인덱스 프루닝을 고려하세요.")
        if vectordb_metrics.search_speed < 0.75:
            recommendations.append("🚀 검색 속도를 향상시키기 위해 HNSW, IVF 등의 고급 인덱싱 기법을 적용하세요.")
        if vectordb_metrics.index_quality < 0.8:
            recommendations.append("🗺️ 인덱스 품질을 개선하기 위해 주기적인 인덱스 재구축과 클러스터링 최적화를 수행하세요.")
    
    # 하이브리드 검색 성능 기반 추천
    if hybrid_performance and hybrid_performance.get('avg_query_time', 0) > 3.0:
        recommendations.append("🔄 하이브리드 검색 성능을 개선하기 위해 병렬 처리를 강화하고 검색 결과 병합 알고리즘을 최적화하세요.")
    
    # 종합적 추천
    if len(recommendations) > 5:
        recommendations.insert(0, "🌟 전반적인 시스템 성능 향상을 위해 단계적 개선 계획을 수립하고 각 개선 사항의 효과를 측정하세요.")
    
    return recommendations


async def quick_single_query_test():
    """단일 쿼리 빠른 테스트"""

    try:
        logger.info("=== 단일 쿼리 테스트 ===")

        # 평가기 초기화
        evaluator = RAGASEvaluator(rag_pipeline=rag_pipeline, use_ragas=True)

        # 단일 쿼리 테스트
        test_query = "회사의 주요 기술 스택은 무엇인가요?"
        result = await evaluator.evaluate_single_query(
            query=test_query,
            ground_truth="Python, FastAPI, LangChain, RAG 시스템, PostgreSQL, Redis, Docker, Kubernetes",
        )

        # 결과 출력
        logger.info(f"질문: {result.query}")
        logger.info(f"답변: {result.answer}")
        logger.info(f"신뢰도: {result.faithfulness:.3f}")
        logger.info(f"답변 관련성: {result.answer_relevancy:.3f}")
        logger.info(f"컨텍스트 정밀도: {result.context_precision:.3f}")
        logger.info(f"응답 시간: {result.response_time:.3f}초")
        logger.info(f"사용된 모델: {result.model_used}")
        logger.info(f"청킹 전략: {result.chunking_strategy}")

        return result

    except Exception as e:
        logger.error(f"단일 쿼리 테스트 중 오류: {e}")
        return None


async def main():
    """메인 함수"""

    # 실행 옵션 확인
    if len(sys.argv) > 1:
        option = sys.argv[1]
        if option == "quick":
            # 빠른 성능 테스트
            await quick_performance_test()
        elif option == "single":
            # 단일 쿼리 테스트
            await quick_single_query_test()
        elif option == "full":
            # 종합 평가
            result = await run_comprehensive_real_evaluation()
            
            if result and result.get("success"):
                print(f"\n✅ 종합 평가 완료! HTML 리포트를 확인하세요: {result['html_report_path']}")
                print("📊 평가 결과:")
                if result.get("summary"):
                    print(f"   RAGAS 평균 신뢰도: {result['summary'].avg_faithfulness:.3f}")
                if result.get("chunk_quality_metrics"):
                    print(f"   청킹 품질 점수: {result['chunk_quality_metrics'].overall_quality:.3f}")
                if result.get("vectordb_metrics"):
                    print(f"   Vector DB 성능: {result['vectordb_metrics'].overall_performance:.3f}")
                
                # 추천사항 출력
                if result.get("recommendations"):
                    print("\n🔧 개선 추천사항:")
                    for rec in result["recommendations"]:
                        print(f"   {rec}")
            else:
                print(f"\n❌ 종합 평가 실패: {result.get('error') if result else 'Unknown error'}")
        else:
            print("사용법: python evaluation_demo.py [quick|single|full]")
            print("  quick: 빠른 성능 테스트")
            print("  single: 단일 쿼리 테스트")
            print("  full: 종합 평가 (청킹 품질 + Vector DB 성능 + RAGAS)")
            print("  (옵션 없음): 기본 실제 데이터 평가")
    else:
        # 기본 실제 데이터 평가
        result = await run_real_data_evaluation()

        if result and result.get("success"):
            print(f"\n✅ 평가 완료! HTML 리포트를 확인하세요: {result['html_report_path']}")
            print("📊 평가 결과:")
            if result.get("summary"):
                print(f"   RAGAS 평균 신뢰도: {result['summary'].avg_faithfulness:.3f}")
            if result.get("chunk_quality_metrics"):
                print(f"   청킹 품질 점수: {result['chunk_quality_metrics'].overall_quality:.3f}")
            if result.get("vectordb_metrics"):
                print(f"   Vector DB 성능: {result['vectordb_metrics'].overall_performance:.3f}")
            
            # 추천사항 출력
            if result.get("recommendations"):
                print("\n🔧 개선 추천사항:")
                for rec in result["recommendations"]:
                    print(f"   {rec}")
        else:
            print(f"\n❌ 평가 실패: {result.get('error') if result else 'Unknown error'}")


if __name__ == "__main__":
    asyncio.run(main())