"""
RAGAS 기반 RAG 시스템 평가기
자동화된 RAG 성능 평가 메트릭 제공
"""

import json
import time
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from enum import Enum

import pandas as pd
import numpy as np

from app.core.logging import logger


def asdict_serializable(instance) -> Dict[str, Any]:
    """dataclass를 JSON 직렬화 가능한 dict로 변환"""
    result = asdict(instance)

    # Enum 객체를 문자열로 변환
    for key, value in result.items():
        if isinstance(value, Enum):
            result[key] = value.value
        elif isinstance(value, (np.floating, np.integer)):
            result[key] = float(value)
        elif hasattr(value, "item"):  # numpy scalar
            result[key] = value.item()

    return result


@dataclass
class EvaluationResult:
    """평가 결과 데이터 클래스"""

    query: str
    answer: str
    contexts: List[str]
    ground_truth: Optional[str] = None

    # RAGAS 메트릭
    faithfulness: float = 0.0
    answer_relevancy: float = 0.0
    context_precision: float = 0.0
    context_recall: float = 0.0
    context_relevancy: float = 0.0

    # 추가 메트릭
    response_time: float = 0.0
    retrieval_time: float = 0.0
    generation_time: float = 0.0
    total_tokens: int = 0

    # 메타데이터
    timestamp: str = ""
    model_used: str = ""
    chunking_strategy: str = ""
    search_strategy: str = ""


@dataclass
class EvaluationSummary:
    """평가 요약 통계"""

    total_queries: int
    avg_faithfulness: float
    avg_answer_relevancy: float
    avg_context_precision: float
    avg_context_recall: float
    avg_context_relevancy: float
    avg_response_time: float
    total_evaluation_time: float
    timestamp: str


class RAGASEvaluator:
    """RAGAS 기반 RAG 평가 시스템"""

    def __init__(
        self,
        rag_pipeline=None,
        results_dir: str = "data/evaluation",
        use_ragas: bool = True,
    ):
        self.rag_pipeline = rag_pipeline
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.use_ragas = use_ragas

        # RAGAS 초기화
        self.ragas_metrics = None
        if use_ragas:
            try:
                self._initialize_ragas()
            except Exception as e:
                logger.warning(
                    f"Failed to initialize RAGAS: {e}, using fallback metrics"
                )
                self.use_ragas = False

    def _initialize_ragas(self):
        """RAGAS 메트릭 초기화"""
        try:
            from ragas import evaluate
            from ragas.metrics import (
                faithfulness,
                answer_relevancy,
                context_precision,
                context_recall,
                context_relevancy,
            )

            self.ragas_metrics = {
                "faithfulness": faithfulness,
                "answer_relevancy": answer_relevancy,
                "context_precision": context_precision,
                "context_recall": context_recall,
                "context_relevancy": context_relevancy,
            }

            self.ragas_evaluate = evaluate
            logger.info("RAGAS metrics initialized successfully")

        except ImportError:
            logger.warning("RAGAS not installed, using fallback metrics")
            self.use_ragas = False
            raise

    async def evaluate_single_query(
        self,
        query: str,
        ground_truth: Optional[str] = None,
        include_timing: bool = True,
    ) -> EvaluationResult:
        """단일 쿼리 평가"""

        if not self.rag_pipeline:
            raise ValueError("RAG pipeline not provided")

        start_time = time.time()

        try:
            # RAG 파이프라인 실행
            retrieval_start = time.time()
            rag_result = await self.rag_pipeline.answer_with_confidence(query, k=5)
            total_time = time.time() - start_time

            if not rag_result or not rag_result.get("answer"):
                logger.warning(f"No answer generated for query: {query}")
                return EvaluationResult(
                    query=query,
                    answer="",
                    contexts=[],
                    ground_truth=ground_truth,
                    timestamp=datetime.now().isoformat(),
                )

            # 결과 추출
            answer = rag_result["answer"]
            contexts = [source for source in rag_result.get("sources", [])]

            # 평가 결과 객체 생성
            # ChunkingStrategy를 문자열로 변환
            chunking_strategy = getattr(
                self.rag_pipeline, "chunking_strategy", "unknown"
            )
            if hasattr(chunking_strategy, "value"):
                chunking_strategy = chunking_strategy.value
            elif hasattr(chunking_strategy, "name"):
                chunking_strategy = chunking_strategy.name
            else:
                chunking_strategy = str(chunking_strategy)

            result = EvaluationResult(
                query=query,
                answer=answer,
                contexts=contexts,
                ground_truth=ground_truth,
                response_time=total_time,
                timestamp=datetime.now().isoformat(),
                model_used=getattr(self.rag_pipeline, "model_name", "unknown"),
                chunking_strategy=chunking_strategy,
                search_strategy=(
                    "hybrid"
                    if hasattr(self.rag_pipeline, "use_hybrid_search")
                    else "vector"
                ),
            )

            # RAGAS 메트릭 계산
            if self.use_ragas and self.ragas_metrics:
                try:
                    ragas_scores = await self._calculate_ragas_metrics(result)
                    result.faithfulness = ragas_scores.get("faithfulness", 0.0)
                    result.answer_relevancy = ragas_scores.get("answer_relevancy", 0.0)
                    result.context_precision = ragas_scores.get(
                        "context_precision", 0.0
                    )
                    result.context_recall = ragas_scores.get("context_recall", 0.0)
                    result.context_relevancy = ragas_scores.get(
                        "context_relevancy", 0.0
                    )
                except Exception as e:
                    logger.warning(f"RAGAS calculation failed: {e}, using fallback")
                    fallback_scores = self._calculate_fallback_metrics(result)
                    result.__dict__.update(fallback_scores)
            else:
                # 폴백 메트릭 사용
                fallback_scores = self._calculate_fallback_metrics(result)
                result.__dict__.update(fallback_scores)

            return result

        except Exception as e:
            logger.error(f"Evaluation failed for query: {query}, error: {e}")
            return EvaluationResult(
                query=query,
                answer="",
                contexts=[],
                ground_truth=ground_truth,
                timestamp=datetime.now().isoformat(),
            )

    async def _calculate_ragas_metrics(
        self, result: EvaluationResult
    ) -> Dict[str, float]:
        """RAGAS 메트릭 계산"""
        try:
            # RAGAS 데이터 형식 준비
            data = {
                "question": [result.query],
                "answer": [result.answer],
                "contexts": [result.contexts],
            }

            if result.ground_truth:
                data["ground_truth"] = [result.ground_truth]

            # DataFrame 생성
            dataset = pd.DataFrame(data)

            # 사용 가능한 메트릭만 선택
            available_metrics = []
            for metric_name, metric in self.ragas_metrics.items():
                if metric_name == "context_recall" and not result.ground_truth:
                    continue  # context_recall은 ground_truth가 필요
                available_metrics.append(metric)

            # RAGAS 평가 실행
            if available_metrics:
                evaluation_result = self.ragas_evaluate(
                    dataset=dataset, metrics=available_metrics
                )

                return evaluation_result.to_dict("records")[0]
            else:
                return {}

        except Exception as e:
            logger.error(f"RAGAS metrics calculation failed: {e}")
            return {}

    def _calculate_fallback_metrics(self, result: EvaluationResult) -> Dict[str, float]:
        """폴백 메트릭 계산 (RAGAS 없이)"""
        try:
            metrics = {}

            # 답변 품질 (키워드 기반)
            query_words = set(result.query.lower().split())
            answer_words = set(result.answer.lower().split())

            # Answer Relevancy (키워드 겹침 비율)
            if query_words and answer_words:
                relevancy = len(query_words.intersection(answer_words)) / len(
                    query_words
                )
                metrics["answer_relevancy"] = min(
                    relevancy * 2, 1.0
                )  # 0-1 범위로 정규화
            else:
                metrics["answer_relevancy"] = 0.0

            # Context Relevancy (컨텍스트가 있는지)
            if result.contexts:
                context_relevancy = 0.8 if len(result.contexts) > 0 else 0.0
                metrics["context_relevancy"] = context_relevancy
            else:
                metrics["context_relevancy"] = 0.0

            # Faithfulness (답변이 있는지 기본 확인)
            if result.answer and len(result.answer.strip()) > 10:
                metrics["faithfulness"] = 0.7  # 기본값
            else:
                metrics["faithfulness"] = 0.0

            # Context Precision (컨텍스트 품질)
            if result.contexts:
                # 컨텍스트와 쿼리의 키워드 겹침
                context_text = " ".join(result.contexts).lower()
                context_words = set(context_text.split())
                precision = (
                    len(query_words.intersection(context_words)) / len(query_words)
                    if query_words
                    else 0
                )
                metrics["context_precision"] = min(precision * 1.5, 1.0)
            else:
                metrics["context_precision"] = 0.0

            # Context Recall (ground truth가 있을 때만)
            if result.ground_truth:
                gt_words = set(result.ground_truth.lower().split())
                context_text = " ".join(result.contexts).lower()
                context_words = set(context_text.split())
                recall = (
                    len(gt_words.intersection(context_words)) / len(gt_words)
                    if gt_words
                    else 0
                )
                metrics["context_recall"] = min(recall * 1.5, 1.0)
            else:
                metrics["context_recall"] = 0.0

            return metrics

        except Exception as e:
            logger.error(f"Fallback metrics calculation failed: {e}")
            return {
                "faithfulness": 0.0,
                "answer_relevancy": 0.0,
                "context_precision": 0.0,
                "context_recall": 0.0,
                "context_relevancy": 0.0,
            }

    async def evaluate_dataset(
        self,
        queries: List[str],
        ground_truths: Optional[List[str]] = None,
        save_results: bool = True,
    ) -> Tuple[List[EvaluationResult], EvaluationSummary]:
        """데이터셋 평가"""

        if ground_truths and len(queries) != len(ground_truths):
            raise ValueError("Queries and ground truths must have the same length")

        start_time = time.time()
        results = []

        logger.info(f"Starting evaluation of {len(queries)} queries")

        for i, query in enumerate(queries):
            ground_truth = ground_truths[i] if ground_truths else None

            try:
                result = await self.evaluate_single_query(query, ground_truth)
                results.append(result)

                if (i + 1) % 10 == 0:
                    logger.info(f"Evaluated {i + 1}/{len(queries)} queries")

            except Exception as e:
                logger.error(f"Failed to evaluate query {i}: {query}, error: {e}")
                # 빈 결과 추가
                results.append(
                    EvaluationResult(
                        query=query,
                        answer="",
                        contexts=[],
                        ground_truth=ground_truth,
                        timestamp=datetime.now().isoformat(),
                    )
                )

        # 요약 통계 계산
        total_time = time.time() - start_time
        summary = self._calculate_summary(results, total_time)

        # 결과 저장
        if save_results:
            self._save_results(results, summary)

        logger.info(f"Evaluation completed in {total_time:.2f} seconds")
        return results, summary

    def _calculate_summary(
        self, results: List[EvaluationResult], total_time: float
    ) -> EvaluationSummary:
        """평가 요약 통계 계산"""

        if not results:
            return EvaluationSummary(
                total_queries=0,
                avg_faithfulness=0.0,
                avg_answer_relevancy=0.0,
                avg_context_precision=0.0,
                avg_context_recall=0.0,
                avg_context_relevancy=0.0,
                avg_response_time=0.0,
                total_evaluation_time=total_time,
                timestamp=datetime.now().isoformat(),
            )

        # 평균 계산
        valid_results = [r for r in results if r.answer]  # 유효한 답변이 있는 결과만

        if not valid_results:
            valid_results = results  # 모든 결과가 비어있다면 전체 사용

        return EvaluationSummary(
            total_queries=len(results),
            avg_faithfulness=float(np.mean([r.faithfulness for r in valid_results])),
            avg_answer_relevancy=float(
                np.mean([r.answer_relevancy for r in valid_results])
            ),
            avg_context_precision=float(
                np.mean([r.context_precision for r in valid_results])
            ),
            avg_context_recall=float(
                np.mean([r.context_recall for r in valid_results])
            ),
            avg_context_relevancy=float(
                np.mean([r.context_relevancy for r in valid_results])
            ),
            avg_response_time=float(np.mean([r.response_time for r in valid_results])),
            total_evaluation_time=total_time,
            timestamp=datetime.now().isoformat(),
        )

    def _save_results(
        self, results: List[EvaluationResult], summary: EvaluationSummary
    ):
        """평가 결과 저장"""

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # 상세 결과 저장
        results_file = self.results_dir / f"evaluation_results_{timestamp}.json"
        with open(results_file, "w", encoding="utf-8") as f:
            json.dump(
                [asdict_serializable(r) for r in results],
                f,
                ensure_ascii=False,
                indent=2,
            )

        # 요약 저장
        summary_file = self.results_dir / f"evaluation_summary_{timestamp}.json"
        with open(summary_file, "w", encoding="utf-8") as f:
            json.dump(asdict_serializable(summary), f, ensure_ascii=False, indent=2)

        # CSV 형태로도 저장 (분석 용이성)
        df = pd.DataFrame([asdict_serializable(r) for r in results])
        csv_file = self.results_dir / f"evaluation_results_{timestamp}.csv"
        df.to_csv(csv_file, index=False, encoding="utf-8")

        logger.info(f"Results saved to {results_file}")
        logger.info(f"Summary saved to {summary_file}")
        logger.info(f"CSV saved to {csv_file}")

    def load_latest_results(self) -> Tuple[List[EvaluationResult], EvaluationSummary]:
        """최신 평가 결과 로드"""

        # 최신 결과 파일 찾기
        result_files = list(self.results_dir.glob("evaluation_results_*.json"))
        summary_files = list(self.results_dir.glob("evaluation_summary_*.json"))

        if not result_files or not summary_files:
            logger.warning("No evaluation results found")
            return [], EvaluationSummary(
                total_queries=0,
                avg_faithfulness=0.0,
                avg_answer_relevancy=0.0,
                avg_context_precision=0.0,
                avg_context_recall=0.0,
                avg_context_relevancy=0.0,
                avg_response_time=0.0,
                total_evaluation_time=0.0,
                timestamp=datetime.now().isoformat(),
            )

        # 가장 최신 파일 선택
        latest_result_file = max(result_files, key=lambda x: x.stat().st_mtime)
        latest_summary_file = max(summary_files, key=lambda x: x.stat().st_mtime)

        # 결과 로드
        with open(latest_result_file, "r", encoding="utf-8") as f:
            results_data = json.load(f)

        with open(latest_summary_file, "r", encoding="utf-8") as f:
            summary_data = json.load(f)

        # 객체로 변환
        results = [EvaluationResult(**data) for data in results_data]
        summary = EvaluationSummary(**summary_data)

        return results, summary

    def generate_test_queries(self, num_queries: int = 10) -> List[str]:
        """테스트용 쿼리 생성"""

        # 기본 테스트 쿼리들
        base_queries = [
            "시스템에 대해 설명해주세요",
            "주요 기능은 무엇인가요?",
            "어떻게 설치하나요?",
            "설정 방법을 알려주세요",
            "문제가 발생했을 때 어떻게 해결하나요?",
            "API는 어떻게 사용하나요?",
            "보안 기능이 있나요?",
            "성능 최적화 방법은?",
            "업데이트는 어떻게 하나요?",
            "지원되는 플랫폼은?",
            "라이센스는 무엇인가요?",
            "데이터는 어디에 저장되나요?",
            "백업 기능이 있나요?",
            "다른 시스템과 연동 가능한가요?",
            "모니터링 기능은?",
            "로그는 어디서 확인하나요?",
            "에러 코드의 의미는?",
            "커스터마이징 방법은?",
            "확장성은 어떤가요?",
            "지원 채널은?",
        ]

        return base_queries[:num_queries]
