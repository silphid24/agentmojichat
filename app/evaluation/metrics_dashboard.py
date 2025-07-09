"""
RAG 메트릭 대시보드
평가 결과 시각화 및 분석 도구
"""

import json
import pandas as pd
from typing import List, Dict, Any, Optional
from pathlib import Path
from datetime import datetime
import numpy as np

from app.core.logging import logger
from .ragas_evaluator import EvaluationResult, EvaluationSummary

# Optional visualization dependencies
try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    VISUALIZATION_AVAILABLE = True
except ImportError:
    VISUALIZATION_AVAILABLE = False
    logger.warning("matplotlib/seaborn not available, visualization features disabled")


class MetricsDashboard:
    """RAG 메트릭 대시보드"""

    def __init__(self, results_dir: str = "data/evaluation"):
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)

        # 시각화 설정 (if available)
        if VISUALIZATION_AVAILABLE:
            try:
                plt.style.use("seaborn-v0_8")
                sns.set_palette("husl")
            except:
                # Fallback to default style if seaborn-v0_8 not available
                plt.style.use("default")
                sns.set_palette("husl")

    def generate_report(
        self,
        results: List[EvaluationResult],
        summary: EvaluationSummary,
        save_plots: bool = True,
    ) -> Dict[str, Any]:
        """종합 리포트 생성"""

        if not results:
            logger.warning("No results to generate report")
            return {}

        report = {
            "summary": summary.__dict__,
            "detailed_metrics": self._analyze_detailed_metrics(results),
            "performance_analysis": self._analyze_performance(results),
            "quality_distribution": self._analyze_quality_distribution(results),
            "recommendations": self._generate_recommendations(results, summary),
        }

        if save_plots and VISUALIZATION_AVAILABLE:
            plot_paths = self._generate_plots(results, summary)
            report["plot_paths"] = plot_paths
        elif save_plots and not VISUALIZATION_AVAILABLE:
            logger.warning("Visualization not available, skipping plot generation")

        # 리포트 저장
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_file = self.results_dir / f"rag_report_{timestamp}.json"
        with open(report_file, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2, default=str)

        logger.info(f"Report generated: {report_file}")
        return report

    def _analyze_detailed_metrics(
        self, results: List[EvaluationResult]
    ) -> Dict[str, Any]:
        """상세 메트릭 분석"""

        df = pd.DataFrame(
            [
                {
                    "faithfulness": r.faithfulness,
                    "answer_relevancy": r.answer_relevancy,
                    "context_precision": r.context_precision,
                    "context_recall": r.context_recall,
                    "context_relevancy": r.context_relevancy,
                    "response_time": r.response_time,
                    "query_length": len(r.query),
                    "answer_length": len(r.answer),
                    "num_contexts": len(r.contexts),
                    "chunk_type": r.chunking_strategy,
                }
                for r in results
            ]
        )

        analysis = {
            "metric_statistics": {
                metric: {
                    "mean": float(df[metric].mean()),
                    "std": float(df[metric].std()),
                    "min": float(df[metric].min()),
                    "max": float(df[metric].max()),
                    "median": float(df[metric].median()),
                }
                for metric in [
                    "faithfulness",
                    "answer_relevancy",
                    "context_precision",
                    "context_recall",
                    "context_relevancy",
                ]
            },
            "correlation_matrix": df[
                [
                    "faithfulness",
                    "answer_relevancy",
                    "context_precision",
                    "context_recall",
                    "context_relevancy",
                    "response_time",
                ]
            ]
            .corr()
            .to_dict(),
            "performance_by_chunk_type": df.groupby("chunk_type")
            .agg(
                {
                    "faithfulness": "mean",
                    "answer_relevancy": "mean",
                    "response_time": "mean",
                }
            )
            .to_dict(),
        }

        return analysis

    def _analyze_performance(self, results: List[EvaluationResult]) -> Dict[str, Any]:
        """성능 분석"""

        response_times = [r.response_time for r in results if r.response_time > 0]
        answer_lengths = [len(r.answer) for r in results if r.answer]

        analysis = {
            "response_time_stats": {
                "mean": np.mean(response_times) if response_times else 0,
                "p95": np.percentile(response_times, 95) if response_times else 0,
                "p99": np.percentile(response_times, 99) if response_times else 0,
                "slow_queries": len([t for t in response_times if t > 5.0]),  # 5초 이상
            },
            "answer_quality": {
                "avg_answer_length": np.mean(answer_lengths) if answer_lengths else 0,
                "empty_answers": len([r for r in results if not r.answer.strip()]),
                "short_answers": len(
                    [r for r in results if r.answer and len(r.answer) < 50]
                ),
            },
            "context_usage": {
                "avg_contexts_per_query": np.mean([len(r.contexts) for r in results]),
                "queries_without_context": len([r for r in results if not r.contexts]),
            },
        }

        return analysis

    def _analyze_quality_distribution(
        self, results: List[EvaluationResult]
    ) -> Dict[str, Any]:
        """품질 분포 분석"""

        # 품질 등급 분류
        def classify_quality(result: EvaluationResult) -> str:
            avg_score = (
                result.faithfulness + result.answer_relevancy + result.context_precision
            ) / 3
            if avg_score >= 0.8:
                return "Excellent"
            elif avg_score >= 0.6:
                return "Good"
            elif avg_score >= 0.4:
                return "Fair"
            else:
                return "Poor"

        quality_grades = [classify_quality(r) for r in results]
        grade_counts = pd.Series(quality_grades).value_counts()

        analysis = {
            "quality_distribution": grade_counts.to_dict(),
            "quality_percentages": (grade_counts / len(results) * 100).to_dict(),
            "high_quality_queries": len(
                [g for g in quality_grades if g in ["Excellent", "Good"]]
            ),
            "needs_improvement": len(
                [g for g in quality_grades if g in ["Fair", "Poor"]]
            ),
        }

        return analysis

    def _generate_recommendations(
        self, results: List[EvaluationResult], summary: EvaluationSummary
    ) -> List[str]:
        """개선 추천사항 생성"""

        recommendations = []

        # 신뢰도 개선
        if summary.avg_faithfulness < 0.7:
            recommendations.append(
                "신뢰도가 낮습니다. 더 정확한 문서 청킹과 컨텍스트 필터링을 고려하세요."
            )

        # 관련성 개선
        if summary.avg_answer_relevancy < 0.6:
            recommendations.append(
                "답변 관련성이 낮습니다. 쿼리 재작성이나 더 나은 검색 알고리즘을 사용하세요."
            )

        # 성능 개선
        if summary.avg_response_time > 3.0:
            recommendations.append(
                "응답 시간이 느립니다. 캐싱이나 더 빠른 임베딩 모델을 고려하세요."
            )

        # 컨텍스트 품질
        if summary.avg_context_precision < 0.5:
            recommendations.append(
                "컨텍스트 정밀도가 낮습니다. 검색 임계값을 조정하거나 리랭킹을 활용하세요."
            )

        # 빈 답변
        empty_answers = len([r for r in results if not r.answer.strip()])
        if empty_answers > len(results) * 0.1:  # 10% 이상
            recommendations.append(
                f"빈 답변이 {empty_answers}개 있습니다. 문서 커버리지를 확인하고 fallback 답변을 준비하세요."
            )

        # 일반적인 권장사항
        if not recommendations:
            recommendations.append(
                "전반적인 성능이 양호합니다. 지속적인 모니터링을 유지하세요."
            )

        return recommendations

    def _generate_plots(
        self, results: List[EvaluationResult], summary: EvaluationSummary
    ) -> Dict[str, str]:
        """시각화 생성"""

        if not VISUALIZATION_AVAILABLE:
            logger.warning("Visualization not available, returning empty plot paths")
            return {}

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        plot_paths = {}

        try:
            # 1. 메트릭 오버뷰
            fig, axes = plt.subplots(2, 3, figsize=(15, 10))
            fig.suptitle("RAG Metrics Overview", fontsize=16)

            metrics = [
                "faithfulness",
                "answer_relevancy",
                "context_precision",
                "context_recall",
                "context_relevancy",
            ]
            values = [getattr(summary, f"avg_{metric}") for metric in metrics]

            # 방사형 차트
            ax = axes[0, 0]
            angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False)
            values_plot = values + [values[0]]  # 닫힌 형태로 만들기
            angles_plot = np.concatenate([angles, [angles[0]]])

            ax.plot(angles_plot, values_plot, "o-", linewidth=2)
            ax.fill(angles_plot, values_plot, alpha=0.25)
            ax.set_xticks(angles)
            ax.set_xticklabels(metrics, rotation=45)
            ax.set_ylim(0, 1)
            ax.set_title("Metrics Radar Chart")
            ax.grid(True)

            # 메트릭 바 차트
            ax = axes[0, 1]
            ax.bar(metrics, values)
            ax.set_title("Average Metrics")
            ax.set_ylabel("Score")
            ax.tick_params(axis="x", rotation=45)

            # 응답 시간 분포
            ax = axes[0, 2]
            response_times = [r.response_time for r in results if r.response_time > 0]
            if response_times:
                ax.hist(response_times, bins=20, alpha=0.7)
                ax.set_title("Response Time Distribution")
                ax.set_xlabel("Response Time (s)")
                ax.set_ylabel("Frequency")

            # 품질 등급 분포
            ax = axes[1, 0]
            quality_grades = []
            for r in results:
                avg_score = (
                    r.faithfulness + r.answer_relevancy + r.context_precision
                ) / 3
                if avg_score >= 0.8:
                    quality_grades.append("Excellent")
                elif avg_score >= 0.6:
                    quality_grades.append("Good")
                elif avg_score >= 0.4:
                    quality_grades.append("Fair")
                else:
                    quality_grades.append("Poor")

            grade_counts = pd.Series(quality_grades).value_counts()
            ax.pie(grade_counts.values, labels=grade_counts.index, autopct="%1.1f%%")
            ax.set_title("Quality Grade Distribution")

            # 메트릭 상관관계
            ax = axes[1, 1]
            df = pd.DataFrame(
                [
                    {
                        "faithfulness": r.faithfulness,
                        "answer_relevancy": r.answer_relevancy,
                        "context_precision": r.context_precision,
                        "response_time": r.response_time,
                    }
                    for r in results
                ]
            )

            correlation_matrix = df.corr()
            sns.heatmap(
                correlation_matrix, annot=True, cmap="coolwarm", center=0, ax=ax
            )
            ax.set_title("Metrics Correlation")

            # 시간별 성능 트렌드 (만약 충분한 데이터가 있다면)
            ax = axes[1, 2]
            if len(results) > 5:
                smoothed_scores = []
                window_size = max(1, len(results) // 10)
                for i in range(len(results)):
                    start_idx = max(0, i - window_size)
                    end_idx = min(len(results), i + window_size + 1)
                    window_results = results[start_idx:end_idx]
                    avg_score = np.mean(
                        [
                            (r.faithfulness + r.answer_relevancy + r.context_precision)
                            / 3
                            for r in window_results
                        ]
                    )
                    smoothed_scores.append(avg_score)

                ax.plot(range(len(results)), smoothed_scores)
                ax.set_title("Quality Trend")
                ax.set_xlabel("Query Index")
                ax.set_ylabel("Average Quality Score")
            else:
                ax.text(
                    0.5,
                    0.5,
                    "Insufficient data\nfor trend analysis",
                    ha="center",
                    va="center",
                    transform=ax.transAxes,
                )
                ax.set_title("Quality Trend")

            plt.tight_layout()

            # 플롯 저장
            overview_path = self.results_dir / f"metrics_overview_{timestamp}.png"
            plt.savefig(overview_path, dpi=300, bbox_inches="tight")
            plt.close()

            plot_paths["overview"] = str(overview_path)

            # 2. 상세 분석 플롯
            fig, axes = plt.subplots(2, 2, figsize=(12, 10))
            fig.suptitle("Detailed Analysis", fontsize=16)

            # 답변 길이 vs 품질
            ax = axes[0, 0]
            answer_lengths = [len(r.answer) for r in results if r.answer]
            quality_scores = [
                (r.faithfulness + r.answer_relevancy) / 2 for r in results if r.answer
            ]

            if answer_lengths and quality_scores:
                ax.scatter(answer_lengths, quality_scores, alpha=0.6)
                ax.set_xlabel("Answer Length")
                ax.set_ylabel("Quality Score")
                ax.set_title("Answer Length vs Quality")

            # 컨텍스트 수 vs 정밀도
            ax = axes[0, 1]
            context_counts = [len(r.contexts) for r in results]
            precision_scores = [r.context_precision for r in results]

            if context_counts and precision_scores:
                ax.scatter(context_counts, precision_scores, alpha=0.6)
                ax.set_xlabel("Number of Contexts")
                ax.set_ylabel("Context Precision")
                ax.set_title("Context Count vs Precision")

            # 청킹 전략별 성능
            ax = axes[1, 0]
            chunk_strategies = [
                r.chunking_strategy for r in results if r.chunking_strategy
            ]
            if chunk_strategies:
                strategy_performance = {}
                for r in results:
                    if r.chunking_strategy:
                        strategy = r.chunking_strategy
                        if strategy not in strategy_performance:
                            strategy_performance[strategy] = []
                        avg_score = (
                            r.faithfulness + r.answer_relevancy + r.context_precision
                        ) / 3
                        strategy_performance[strategy].append(avg_score)

                strategies = list(strategy_performance.keys())
                avg_scores = [np.mean(strategy_performance[s]) for s in strategies]

                ax.bar(strategies, avg_scores)
                ax.set_title("Performance by Chunking Strategy")
                ax.set_ylabel("Average Score")
                ax.tick_params(axis="x", rotation=45)

            # 에러 분석
            ax = axes[1, 1]
            error_types = {
                "Empty Answer": len([r for r in results if not r.answer.strip()]),
                "No Context": len([r for r in results if not r.contexts]),
                "Low Quality": len(
                    [
                        r
                        for r in results
                        if (r.faithfulness + r.answer_relevancy + r.context_precision)
                        / 3
                        < 0.4
                    ]
                ),
                "Slow Response": len([r for r in results if r.response_time > 5.0]),
            }

            ax.bar(error_types.keys(), error_types.values())
            ax.set_title("Error Analysis")
            ax.set_ylabel("Count")
            ax.tick_params(axis="x", rotation=45)

            plt.tight_layout()

            detail_path = self.results_dir / f"detailed_analysis_{timestamp}.png"
            plt.savefig(detail_path, dpi=300, bbox_inches="tight")
            plt.close()

            plot_paths["detailed"] = str(detail_path)

        except Exception as e:
            logger.error(f"Failed to generate plots: {e}")

        return plot_paths

    def create_html_report(
        self,
        results: List[EvaluationResult],
        summary: EvaluationSummary,
        plot_paths: Optional[Dict[str, str]] = None,
    ) -> str:
        """HTML 리포트 생성"""

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>RAG Evaluation Report - {timestamp}</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 40px; }}
                .header {{ background-color: #f4f4f4; padding: 20px; border-radius: 5px; }}
                .section {{ margin: 20px 0; }}
                .metric {{ background-color: #e9e9e9; padding: 10px; margin: 10px 0; border-radius: 3px; }}
                .good {{ color: green; }}
                .warning {{ color: orange; }}
                .bad {{ color: red; }}
                table {{ border-collapse: collapse; width: 100%; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #f2f2f2; }}
                .plot {{ text-align: center; margin: 20px 0; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>RAG System Evaluation Report</h1>
                <p>Generated on: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>
                <p>Total Queries Evaluated: {summary.total_queries}</p>
            </div>
            
            <div class="section">
                <h2>Executive Summary</h2>
                <div class="metric">
                    <strong>Overall Performance:</strong> 
                    {"Excellent" if summary.avg_faithfulness > 0.8 else "Good" if summary.avg_faithfulness > 0.6 else "Needs Improvement"}
                </div>
                <div class="metric">
                    <strong>Average Response Time:</strong> {summary.avg_response_time:.2f} seconds
                </div>
            </div>
            
            <div class="section">
                <h2>Key Metrics</h2>
                <table>
                    <tr><th>Metric</th><th>Score</th><th>Status</th></tr>
                    <tr>
                        <td>Faithfulness</td>
                        <td>{summary.avg_faithfulness:.3f}</td>
                        <td class="{'good' if summary.avg_faithfulness > 0.7 else 'warning' if summary.avg_faithfulness > 0.5 else 'bad'}">
                            {'Good' if summary.avg_faithfulness > 0.7 else 'Warning' if summary.avg_faithfulness > 0.5 else 'Poor'}
                        </td>
                    </tr>
                    <tr>
                        <td>Answer Relevancy</td>
                        <td>{summary.avg_answer_relevancy:.3f}</td>
                        <td class="{'good' if summary.avg_answer_relevancy > 0.7 else 'warning' if summary.avg_answer_relevancy > 0.5 else 'bad'}">
                            {'Good' if summary.avg_answer_relevancy > 0.7 else 'Warning' if summary.avg_answer_relevancy > 0.5 else 'Poor'}
                        </td>
                    </tr>
                    <tr>
                        <td>Context Precision</td>
                        <td>{summary.avg_context_precision:.3f}</td>
                        <td class="{'good' if summary.avg_context_precision > 0.7 else 'warning' if summary.avg_context_precision > 0.5 else 'bad'}">
                            {'Good' if summary.avg_context_precision > 0.7 else 'Warning' if summary.avg_context_precision > 0.5 else 'Poor'}
                        </td>
                    </tr>
                </table>
            </div>
        """

        # 플롯 추가
        if plot_paths:
            html_content += """
            <div class="section">
                <h2>Visualizations</h2>
            """
            for plot_name, plot_path in plot_paths.items():
                html_content += f"""
                <div class="plot">
                    <h3>{plot_name.title()} Analysis</h3>
                    <img src="{plot_path}" alt="{plot_name} plot" style="max-width: 100%;">
                </div>
                """
            html_content += "</div>"

        html_content += """
            </body>
            </html>
        """

        # HTML 파일 저장
        html_path = self.results_dir / f"evaluation_report_{timestamp}.html"
        with open(html_path, "w", encoding="utf-8") as f:
            f.write(html_content)

        logger.info(f"HTML report generated: {html_path}")
        return str(html_path)

    def create_comprehensive_html_report(
        self,
        results: List[EvaluationResult],
        summary: EvaluationSummary,
        enhanced_report: Dict[str, Any],
    ) -> str:
        """종합 HTML 리포트 생성 (청킹 품질 및 Vector DB 성능 포함)"""

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 기본 HTML 구조
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Comprehensive RAG Evaluation Report - {timestamp}</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 40px; background-color: #f8f9fa; }}
                .container {{ max-width: 1200px; margin: 0 auto; background-color: white; padding: 30px; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }}
                .header {{ background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 30px; border-radius: 10px; margin-bottom: 30px; }}
                .section {{ margin: 30px 0; padding: 20px; background-color: #f8f9fa; border-radius: 8px; }}
                .metric-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 20px; margin: 20px 0; }}
                .metric-card {{ background-color: white; padding: 20px; border-radius: 8px; box-shadow: 0 2px 5px rgba(0,0,0,0.1); }}
                .metric-value {{ font-size: 2em; font-weight: bold; margin: 10px 0; }}
                .good {{ color: #28a745; }}
                .warning {{ color: #ffc107; }}
                .bad {{ color: #dc3545; }}
                table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
                th, td {{ border: 1px solid #ddd; padding: 12px; text-align: left; }}
                th {{ background-color: #e9ecef; font-weight: bold; }}
                tr:nth-child(even) {{ background-color: #f8f9fa; }}
                .plot {{ text-align: center; margin: 20px 0; }}
                .recommendations {{ background-color: #e7f3ff; border-left: 4px solid #007bff; padding: 20px; margin: 20px 0; }}
                .quality-badge {{ display: inline-block; padding: 5px 15px; border-radius: 20px; color: white; font-weight: bold; margin: 5px; }}
                .excellent {{ background-color: #28a745; }}
                .very-good {{ background-color: #17a2b8; }}
                .good-badge {{ background-color: #20c997; }}
                .fair {{ background-color: #ffc107; color: #212529; }}
                .poor {{ background-color: #fd7e14; }}
                .very-poor {{ background-color: #dc3545; }}
            </style>
        </head>
        <body>
            <div class="container">
                <div class="header">
                    <h1>🚀 Comprehensive RAG Evaluation Report</h1>
                    <p>Generated on: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>
                    <p>Total Queries Evaluated: {summary.total_queries}</p>
                </div>
        """

        # Executive Summary
        overall_score = (summary.avg_faithfulness + summary.avg_answer_relevancy + summary.avg_context_precision) / 3
        html_content += f"""
                <div class="section">
                    <h2>📊 Executive Summary</h2>
                    <div class="metric-grid">
                        <div class="metric-card">
                            <h3>Overall RAG Performance</h3>
                            <div class="metric-value {'good' if overall_score > 0.7 else 'warning' if overall_score > 0.5 else 'bad'}">
                                {overall_score:.2f}
                            </div>
                            <p>{"Excellent" if overall_score > 0.8 else "Good" if overall_score > 0.7 else "Fair" if overall_score > 0.5 else "Needs Improvement"}</p>
                        </div>
                        <div class="metric-card">
                            <h3>Average Response Time</h3>
                            <div class="metric-value {'good' if summary.avg_response_time < 2 else 'warning' if summary.avg_response_time < 5 else 'bad'}">
                                {summary.avg_response_time:.2f}s
                            </div>
                            <p>{"Fast" if summary.avg_response_time < 2 else "Acceptable" if summary.avg_response_time < 5 else "Slow"}</p>
                        </div>
        """

        # 청킹 품질 카드 추가
        if enhanced_report.get("chunk_quality"):
            chunk_quality = enhanced_report["chunk_quality"]
            chunk_score = chunk_quality["metrics"].get("overall_quality", 0)
            chunk_grade = chunk_quality.get("grade", "Unknown")
            
            html_content += f"""
                        <div class="metric-card">
                            <h3>🧩 Chunk Quality</h3>
                            <div class="metric-value {'good' if chunk_score > 0.7 else 'warning' if chunk_score > 0.5 else 'bad'}">
                                {chunk_score:.2f}
                            </div>
                            <p><span class="quality-badge {chunk_grade.lower().replace(' ', '-')}">{chunk_grade}</span></p>
                        </div>
            """

        # Vector DB 성능 카드 추가
        if enhanced_report.get("vectordb_performance"):
            vectordb_perf = enhanced_report["vectordb_performance"]
            vectordb_score = vectordb_perf["metrics"].get("overall_performance", 0)
            vectordb_grade = vectordb_perf.get("grade", "Unknown")
            
            html_content += f"""
                        <div class="metric-card">
                            <h3>🗃️ Vector DB Performance</h3>
                            <div class="metric-value {'good' if vectordb_score > 0.7 else 'warning' if vectordb_score > 0.5 else 'bad'}">
                                {vectordb_score:.2f}
                            </div>
                            <p><span class="quality-badge {vectordb_grade.lower().replace(' ', '-')}">{vectordb_grade}</span></p>
                        </div>
            """

        html_content += """
                    </div>
                </div>
        """

        # RAGAS 메트릭 상세
        html_content += f"""
                <div class="section">
                    <h2>📈 RAGAS Evaluation Metrics</h2>
                    <table>
                        <tr><th>Metric</th><th>Score</th><th>Status</th><th>Description</th></tr>
                        <tr>
                            <td><strong>Faithfulness</strong></td>
                            <td>{summary.avg_faithfulness:.3f}</td>
                            <td class="{'good' if summary.avg_faithfulness > 0.7 else 'warning' if summary.avg_faithfulness > 0.5 else 'bad'}">
                                {'Good' if summary.avg_faithfulness > 0.7 else 'Warning' if summary.avg_faithfulness > 0.5 else 'Poor'}
                            </td>
                            <td>How factually accurate the generated answer is</td>
                        </tr>
                        <tr>
                            <td><strong>Answer Relevancy</strong></td>
                            <td>{summary.avg_answer_relevancy:.3f}</td>
                            <td class="{'good' if summary.avg_answer_relevancy > 0.7 else 'warning' if summary.avg_answer_relevancy > 0.5 else 'bad'}">
                                {'Good' if summary.avg_answer_relevancy > 0.7 else 'Warning' if summary.avg_answer_relevancy > 0.5 else 'Poor'}
                            </td>
                            <td>How relevant the answer is to the given question</td>
                        </tr>
                        <tr>
                            <td><strong>Context Precision</strong></td>
                            <td>{summary.avg_context_precision:.3f}</td>
                            <td class="{'good' if summary.avg_context_precision > 0.7 else 'warning' if summary.avg_context_precision > 0.5 else 'bad'}">
                                {'Good' if summary.avg_context_precision > 0.7 else 'Warning' if summary.avg_context_precision > 0.5 else 'Poor'}
                            </td>
                            <td>How precise and relevant the retrieved context is</td>
                        </tr>
                        <tr>
                            <td><strong>Context Recall</strong></td>
                            <td>{summary.avg_context_recall:.3f}</td>
                            <td class="{'good' if summary.avg_context_recall > 0.7 else 'warning' if summary.avg_context_recall > 0.5 else 'bad'}">
                                {'Good' if summary.avg_context_recall > 0.7 else 'Warning' if summary.avg_context_recall > 0.5 else 'Poor'}
                            </td>
                            <td>How much relevant context was successfully retrieved</td>
                        </tr>
                    </table>
                </div>
        """

        # 청킹 품질 상세 분석
        if enhanced_report.get("chunk_quality"):
            chunk_metrics = enhanced_report["chunk_quality"]["metrics"]
            html_content += f"""
                <div class="section">
                    <h2>🧩 Chunk Quality Analysis</h2>
                    <table>
                        <tr><th>Metric</th><th>Score</th><th>Status</th><th>Description</th></tr>
                        <tr>
                            <td><strong>Semantic Coherence</strong></td>
                            <td>{chunk_metrics.get('semantic_coherence', 0):.3f}</td>
                            <td class="{'good' if chunk_metrics.get('semantic_coherence', 0) > 0.7 else 'warning' if chunk_metrics.get('semantic_coherence', 0) > 0.5 else 'bad'}">
                                {'Good' if chunk_metrics.get('semantic_coherence', 0) > 0.7 else 'Warning' if chunk_metrics.get('semantic_coherence', 0) > 0.5 else 'Poor'}
                            </td>
                            <td>How semantically coherent content within chunks is</td>
                        </tr>
                        <tr>
                            <td><strong>Boundary Quality</strong></td>
                            <td>{chunk_metrics.get('boundary_quality', 0):.3f}</td>
                            <td class="{'good' if chunk_metrics.get('boundary_quality', 0) > 0.7 else 'warning' if chunk_metrics.get('boundary_quality', 0) > 0.5 else 'bad'}">
                                {'Good' if chunk_metrics.get('boundary_quality', 0) > 0.7 else 'Warning' if chunk_metrics.get('boundary_quality', 0) > 0.5 else 'Poor'}
                            </td>
                            <td>How well chunks respect natural text boundaries</td>
                        </tr>
                        <tr>
                            <td><strong>Information Coverage</strong></td>
                            <td>{chunk_metrics.get('information_coverage', 0):.3f}</td>
                            <td class="{'good' if chunk_metrics.get('information_coverage', 0) > 0.7 else 'warning' if chunk_metrics.get('information_coverage', 0) > 0.5 else 'bad'}">
                                {'Good' if chunk_metrics.get('information_coverage', 0) > 0.7 else 'Warning' if chunk_metrics.get('information_coverage', 0) > 0.5 else 'Poor'}
                            </td>
                            <td>How well chunks cover the original information</td>
                        </tr>
                        <tr>
                            <td><strong>Size Consistency</strong></td>
                            <td>{chunk_metrics.get('size_consistency', 0):.3f}</td>
                            <td class="{'good' if chunk_metrics.get('size_consistency', 0) > 0.7 else 'warning' if chunk_metrics.get('size_consistency', 0) > 0.5 else 'bad'}">
                                {'Good' if chunk_metrics.get('size_consistency', 0) > 0.7 else 'Warning' if chunk_metrics.get('size_consistency', 0) > 0.5 else 'Poor'}
                            </td>
                            <td>How consistent chunk sizes are across the dataset</td>
                        </tr>
                    </table>
                </div>
            """

        # Vector DB 성능 상세 분석
        if enhanced_report.get("vectordb_performance"):
            vectordb_metrics = enhanced_report["vectordb_performance"]["metrics"]
            html_content += f"""
                <div class="section">
                    <h2>🗃️ Vector Database Performance Analysis</h2>
                    <table>
                        <tr><th>Metric</th><th>Score</th><th>Status</th><th>Description</th></tr>
                        <tr>
                            <td><strong>Index Quality</strong></td>
                            <td>{vectordb_metrics.get('index_quality', 0):.3f}</td>
                            <td class="{'good' if vectordb_metrics.get('index_quality', 0) > 0.7 else 'warning' if vectordb_metrics.get('index_quality', 0) > 0.5 else 'bad'}">
                                {'Good' if vectordb_metrics.get('index_quality', 0) > 0.7 else 'Warning' if vectordb_metrics.get('index_quality', 0) > 0.5 else 'Poor'}
                            </td>
                            <td>Quality of vector distribution and clustering</td>
                        </tr>
                        <tr>
                            <td><strong>Search Accuracy</strong></td>
                            <td>{vectordb_metrics.get('search_accuracy', 0):.3f}</td>
                            <td class="{'good' if vectordb_metrics.get('search_accuracy', 0) > 0.7 else 'warning' if vectordb_metrics.get('search_accuracy', 0) > 0.5 else 'bad'}">
                                {'Good' if vectordb_metrics.get('search_accuracy', 0) > 0.7 else 'Warning' if vectordb_metrics.get('search_accuracy', 0) > 0.5 else 'Poor'}
                            </td>
                            <td>Accuracy of similarity search results</td>
                        </tr>
                        <tr>
                            <td><strong>Search Speed</strong></td>
                            <td>{vectordb_metrics.get('search_speed', 0):.3f}</td>
                            <td class="{'good' if vectordb_metrics.get('search_speed', 0) > 0.7 else 'warning' if vectordb_metrics.get('search_speed', 0) > 0.5 else 'bad'}">
                                {'Good' if vectordb_metrics.get('search_speed', 0) > 0.7 else 'Warning' if vectordb_metrics.get('search_speed', 0) > 0.5 else 'Poor'}
                            </td>
                            <td>Speed of query processing and retrieval</td>
                        </tr>
                        <tr>
                            <td><strong>Memory Efficiency</strong></td>
                            <td>{vectordb_metrics.get('memory_efficiency', 0):.3f}</td>
                            <td class="{'good' if vectordb_metrics.get('memory_efficiency', 0) > 0.7 else 'warning' if vectordb_metrics.get('memory_efficiency', 0) > 0.5 else 'bad'}">
                                {'Good' if vectordb_metrics.get('memory_efficiency', 0) > 0.7 else 'Warning' if vectordb_metrics.get('memory_efficiency', 0) > 0.5 else 'Poor'}
                            </td>
                            <td>Efficiency of memory usage for vector storage</td>
                        </tr>
                    </table>
                </div>
            """

        # 추천사항
        recommendations = enhanced_report.get("recommendations", [])
        if recommendations:
            html_content += """
                <div class="section">
                    <h2>💡 Comprehensive Recommendations</h2>
                    <div class="recommendations">
                        <h3>Key Areas for Improvement:</h3>
                        <ul>
            """
            for rec in recommendations:
                html_content += f"<li>{rec}</li>"
            
            html_content += """
                        </ul>
                    </div>
                </div>
            """

        # 기술적 상세정보
        html_content += f"""
                <div class="section">
                    <h2>🔧 Technical Details</h2>
                    <div class="metric-grid">
                        <div class="metric-card">
                            <h3>Evaluation Configuration</h3>
                            <p><strong>Total Queries:</strong> {summary.total_queries}</p>
                            <p><strong>Evaluation Time:</strong> {summary.total_evaluation_time:.2f}s</p>
                            <p><strong>Average Response Time:</strong> {summary.avg_response_time:.2f}s</p>
                        </div>
        """

        if enhanced_report.get("chunk_quality", {}).get("report", {}).get("chunk_statistics"):
            chunk_stats = enhanced_report["chunk_quality"]["report"]["chunk_statistics"]
            html_content += f"""
                        <div class="metric-card">
                            <h3>Chunk Statistics</h3>
                            <p><strong>Total Chunks:</strong> {enhanced_report["chunk_quality"]["report"]["summary"].get("total_chunks", "N/A")}</p>
                            <p><strong>Avg Chunk Size:</strong> {chunk_stats.get("avg_chunk_size", 0):.0f} chars</p>
                            <p><strong>Size Range:</strong> {chunk_stats.get("min_size", 0):.0f} - {chunk_stats.get("max_size", 0):.0f}</p>
                        </div>
            """

        if enhanced_report.get("vectordb_performance", {}).get("report", {}).get("technical_details"):
            tech_details = enhanced_report["vectordb_performance"]["report"]["technical_details"]
            html_content += f"""
                        <div class="metric-card">
                            <h3>Vector DB Stats</h3>
                            <p><strong>Total Documents:</strong> {tech_details.get("index_stats", {}).get("document_count", "N/A")}</p>
                            <p><strong>Query Throughput:</strong> {vectordb_metrics.get("query_throughput", 0):.1f} queries/sec</p>
                            <p><strong>Embedding Dimension:</strong> {tech_details.get("index_stats", {}).get("embedding_dimension", "N/A")}</p>
                        </div>
            """

        html_content += """
                    </div>
                </div>
            </div>
        </body>
        </html>
        """

        # HTML 파일 저장
        report_path = self.results_dir / f"comprehensive_evaluation_report_{timestamp}.html"
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(html_content)

        logger.info(f"Comprehensive HTML report saved to: {report_path}")
        return str(report_path)
