#!/usr/bin/env python3
"""
RAG 평가 시스템 데모 스크립트
RAGAS 메트릭과 대시보드 기능을 테스트합니다.
"""

import asyncio
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from app.rag.enhanced_rag import rag_pipeline
from app.evaluation.ragas_evaluator import RAGASEvaluator
from app.evaluation.metrics_dashboard import MetricsDashboard
from app.core.logging import logger


async def setup_demo_data():
    """데모용 문서 데이터 준비"""

    # 데모 문서 디렉토리 생성
    demo_docs_dir = project_root / "data" / "demo_documents"
    demo_docs_dir.mkdir(parents=True, exist_ok=True)

    # 샘플 문서들 생성
    sample_docs = {
        "company_overview.md": """
# 회사 개요

## 회사명
MOJI AI Solutions

## 설립연도
2024년

## 주요 사업
- AI 챗봇 개발
- 멀티플랫폼 메신저 통합
- 기업용 AI 어시스턴트 솔루션

## 기술 스택
- Python, FastAPI
- LangChain, RAG 시스템
- PostgreSQL, Redis
- Docker, Kubernetes

## 주요 제품
MOJI Agent - 멀티플랫폼 AI 어시스턴트
""",
        "employee_benefits.md": """
# 직원 복리후생

## 휴가 제도
- 연차: 15일 (입사 첫해)
- 병가: 무제한 (의사 진단서 필요)
- 출산/육아 휴가: 법정 기준 준수

## 보험 혜택
- 4대 보험 완비
- 건강검진 지원 (연 1회)
- 단체상해보험

## 교육 지원
- 외부 교육과정 수강료 지원
- 컨퍼런스 참가비 지원
- 온라인 강의 플랫폼 구독료 지원

## 기타 혜택
- 점심값 지원 (1일 10,000원)
- 간식 및 음료 제공
- 야근 시 저녁식대 지원
""",
        "development_guide.md": """
# 개발 가이드

## 코딩 표준
- Python PEP 8 준수
- Black 포매터 사용
- Type hints 필수

## 테스트
- 테스트 커버리지 80% 이상 유지
- pytest 사용
- 단위 테스트, 통합 테스트 작성

## Git 워크플로우
- feature/브랜치명 으로 브랜치 생성
- Pull Request 필수
- 코드 리뷰 후 merge

## 배포 프로세스
- Docker 컨테이너화
- CI/CD 파이프라인 사용
- 스테이징 환경 테스트 후 프로덕션 배포
""",
        "technical_architecture.md": """
# 기술 아키텍처

## 시스템 구성
- 마이크로서비스 아키텍처
- RESTful API 설계
- 이벤트 드리븐 시스템

## 인프라
- AWS 클라우드 인프라
- Kubernetes 오케스트레이션
- Docker 컨테이너화
- Terraform IaC

## 데이터베이스
- PostgreSQL: 메인 데이터베이스
- Redis: 캐싱 및 세션 관리
- Elasticsearch: 검색 엔진
- ChromaDB: 벡터 데이터베이스

## 모니터링
- Prometheus & Grafana
- ELK Stack
- Sentry 에러 트래킹
""",
        "onboarding_guide.md": """
# 신입 사원 온보딩 가이드

## 첫 주 일정
- 월요일: 오리엔테이션 및 계정 설정
- 화요일: 제품 교육 및 시스템 이해
- 수요일: 개발 환경 설정
- 목요일: 코드베이스 탐색
- 금요일: 첫 번째 작은 태스크 수행

## 필수 도구
- Slack: 팀 커뮤니케이션
- Jira: 프로젝트 관리
- Confluence: 문서화
- GitHub: 버전 관리

## 멘토링 프로그램
- 전담 멘토 배정
- 주간 1:1 미팅
- 월간 성과 리뷰
""",
    }

    # 문서 파일들 생성
    for filename, content in sample_docs.items():
        file_path = demo_docs_dir / filename
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(content)
        logger.info(f"Created demo document: {filename}")

    return demo_docs_dir


async def run_evaluation_demo():
    """평가 시스템 데모 실행"""

    try:
        logger.info("=== RAG 평가 시스템 데모 시작 ===")

        # 1. 데모 데이터 준비
        logger.info("1. 데모 문서 준비 중...")
        demo_docs_dir = await setup_demo_data()

        # 2. RAG 파이프라인에 문서 로드
        logger.info("2. RAG 파이프라인에 문서 로드 중...")
        doc_files = [str(f) for f in demo_docs_dir.glob("*.md")]
        load_result = await rag_pipeline.load_documents(doc_files)

        if not load_result.get("success"):
            logger.error(f"문서 로드 실패: {load_result}")
            return

        logger.info(f"문서 로드 완료: {load_result['total_chunks']}개 청크 생성")

        # 3. RAGAS 평가기 초기화
        logger.info("3. RAGAS 평가기 초기화 중...")
        evaluator = RAGASEvaluator(
            rag_pipeline=rag_pipeline,
            results_dir="data/evaluation",
            use_ragas=True,  # RAGAS 사용 시도, 실패 시 fallback 메트릭 사용
        )

        # 4. 테스트 쿼리 생성
        logger.info("4. 테스트 쿼리 실행 중...")
        test_queries = [
            "회사 이름이 무엇인가요?",
            "주요 기술 스택은 무엇인가요?",
            "직원 복리후생 중 휴가 제도는 어떻게 되나요?",
            "개발 시 코딩 표준은 무엇을 따라야 하나요?",
            "테스트 커버리지는 몇 퍼센트 이상 유지해야 하나요?",
            "점심값 지원은 얼마나 되나요?",
            "Git 워크플로우는 어떻게 되나요?",
            "배포는 어떤 방식으로 하나요?",
            "사용하는 데이터베이스는 무엇인가요?",
            "신입 사원 첫 주 일정은 어떻게 되나요?",
            "모니터링 도구로는 무엇을 사용하나요?",
            "시스템 아키텍처는 어떤 방식인가요?",
        ]

        # Ground truth 답변 (선택적)
        ground_truths = [
            "MOJI AI Solutions",
            "Python, FastAPI, LangChain, RAG 시스템, PostgreSQL, Redis, Docker, Kubernetes",
            "연차 15일, 병가 무제한, 출산/육아 휴가",
            "Python PEP 8 준수, Black 포매터 사용, Type hints 필수",
            "80% 이상",
            "1일 10,000원",
            "feature/브랜치명으로 브랜치 생성, Pull Request 필수, 코드 리뷰 후 merge",
            "Docker 컨테이너화, CI/CD 파이프라인, 스테이징 환경 테스트 후 프로덕션 배포",
            "PostgreSQL (메인), Redis (캐싱), Elasticsearch (검색), ChromaDB (벡터)",
            "월: 오리엔테이션, 화: 제품 교육, 수: 개발 환경 설정, 목: 코드베이스 탐색, 금: 첫 태스크",
            "Prometheus & Grafana, ELK Stack, Sentry",
            "마이크로서비스 아키텍처, RESTful API, 이벤트 드리븐 시스템",
        ]

        # 5. 평가 실행
        logger.info(f"5. {len(test_queries)}개 쿼리에 대한 평가 실행 중...")
        results, summary = await evaluator.evaluate_dataset(
            queries=test_queries, ground_truths=ground_truths, save_results=True
        )

        # 6. 대시보드 생성
        logger.info("6. 메트릭 대시보드 생성 중...")
        dashboard = MetricsDashboard(results_dir="data/evaluation")

        # 리포트 생성
        report = dashboard.generate_report(results, summary, save_plots=True)

        # HTML 리포트 생성
        html_report_path = dashboard.create_html_report(
            results, summary, report.get("plot_paths")
        )

        # 7. 결과 출력
        logger.info("=== 평가 결과 요약 ===")
        logger.info(f"총 쿼리 수: {summary.total_queries}")
        logger.info(f"평균 신뢰도: {summary.avg_faithfulness:.3f}")
        logger.info(f"평균 답변 관련성: {summary.avg_answer_relevancy:.3f}")
        logger.info(f"평균 컨텍스트 정밀도: {summary.avg_context_precision:.3f}")
        logger.info(f"평균 응답 시간: {summary.avg_response_time:.3f}초")
        logger.info(f"전체 평가 시간: {summary.total_evaluation_time:.3f}초")

        # 8. 개선 추천사항
        recommendations = report.get("recommendations", [])
        if recommendations:
            logger.info("=== 개선 추천사항 ===")
            for i, rec in enumerate(recommendations, 1):
                logger.info(f"{i}. {rec}")

        # 9. 파일 경로 안내
        logger.info("=== 생성된 파일들 ===")
        logger.info(f"HTML 리포트: {html_report_path}")
        if report.get("plot_paths"):
            for plot_name, plot_path in report["plot_paths"].items():
                logger.info(f"{plot_name.title()} 플롯: {plot_path}")

        logger.info("=== RAG 평가 시스템 데모 완료 ===")

        return {
            "success": True,
            "summary": summary,
            "report": report,
            "html_report": html_report_path,
            "recommendations": recommendations,
        }

    except Exception as e:
        logger.error(f"평가 데모 실행 중 오류 발생: {e}")
        return {"success": False, "error": str(e)}


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
    if len(sys.argv) > 1 and sys.argv[1] == "quick":
        # 빠른 단일 쿼리 테스트
        await quick_single_query_test()
    else:
        # 전체 평가 데모
        result = await run_evaluation_demo()

        if result and result.get("success"):
            print(f"\n✅ 평가 완료! HTML 리포트를 확인하세요: {result['html_report']}")
        else:
            print(
                f"\n❌ 평가 실패: {result.get('error') if result else 'Unknown error'}"
            )


if __name__ == "__main__":
    asyncio.run(main())
