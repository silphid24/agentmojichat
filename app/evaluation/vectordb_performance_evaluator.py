"""
Vector DB 성능 평가기 (Vector Database Performance Evaluator)
인덱스 품질, 검색 정확도, 메모리 효율성을 종합적으로 평가
"""

import time
import numpy as np
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass
from pathlib import Path
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics.pairwise import cosine_similarity
from collections import defaultdict

# Optional dependencies
try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False

try:
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False

from langchain.schema import Document
from langchain_community.vectorstores import Chroma
from app.core.logging import logger


@dataclass
class VectorDBPerformanceMetrics:
    """Vector DB 성능 메트릭"""
    
    # 인덱스 품질
    index_quality: float = 0.0
    vector_distribution_score: float = 0.0
    clustering_quality: float = 0.0
    dimensionality_efficiency: float = 0.0
    
    # 검색 성능
    search_accuracy: float = 0.0
    search_speed: float = 0.0
    precision_at_k: Dict[int, float] = None
    recall_at_k: Dict[int, float] = None
    
    # 메모리 효율성
    memory_efficiency: float = 0.0
    storage_efficiency: float = 0.0
    index_size_ratio: float = 0.0
    
    # 확장성
    scalability_score: float = 0.0
    query_throughput: float = 0.0
    
    # 일관성
    consistency_score: float = 0.0
    retrieval_stability: float = 0.0
    
    # 종합 점수
    overall_performance: float = 0.0

    def __post_init__(self):
        if self.precision_at_k is None:
            self.precision_at_k = {}
        if self.recall_at_k is None:
            self.recall_at_k = {}

    def to_dict(self) -> Dict[str, Any]:
        """딕셔너리로 변환"""
        return {
            'index_quality': self.index_quality,
            'vector_distribution_score': self.vector_distribution_score,
            'clustering_quality': self.clustering_quality,
            'dimensionality_efficiency': self.dimensionality_efficiency,
            'search_accuracy': self.search_accuracy,
            'search_speed': self.search_speed,
            'precision_at_k': self.precision_at_k,
            'recall_at_k': self.recall_at_k,
            'memory_efficiency': self.memory_efficiency,
            'storage_efficiency': self.storage_efficiency,
            'index_size_ratio': self.index_size_ratio,
            'scalability_score': self.scalability_score,
            'query_throughput': self.query_throughput,
            'consistency_score': self.consistency_score,
            'retrieval_stability': self.retrieval_stability,
            'overall_performance': self.overall_performance
        }


class VectorDBPerformanceEvaluator:
    """Vector DB 성능 종합 평가기"""
    
    def __init__(
        self,
        vectorstore: Chroma,
        test_queries: List[str] = None,
        k_values: List[int] = None
    ):
        self.vectorstore = vectorstore
        self.test_queries = test_queries or self._generate_default_queries()
        self.k_values = k_values or [1, 3, 5, 10]
        
        # 성능 측정 데이터
        self.query_times = []
        self.memory_usage = []
        self.search_results_cache = {}
        
        logger.info("Vector DB Performance Evaluator initialized")
    
    def evaluate_performance(self, detailed_analysis: bool = True) -> VectorDBPerformanceMetrics:
        """Vector DB 성능 종합 평가"""
        
        try:
            logger.info("Starting Vector DB performance evaluation...")
            
            metrics = VectorDBPerformanceMetrics()
            
            # 1. 인덱스 품질 평가
            index_metrics = self._evaluate_index_quality()
            metrics.index_quality = index_metrics['overall']
            metrics.vector_distribution_score = index_metrics['distribution']
            metrics.clustering_quality = index_metrics['clustering']
            metrics.dimensionality_efficiency = index_metrics['dimensionality']
            
            # 2. 검색 성능 평가
            search_metrics = self._evaluate_search_performance()
            metrics.search_accuracy = search_metrics['accuracy']
            metrics.search_speed = search_metrics['speed']
            metrics.precision_at_k = search_metrics['precision_at_k']
            metrics.recall_at_k = search_metrics['recall_at_k']
            
            # 3. 메모리 효율성 평가
            memory_metrics = self._evaluate_memory_efficiency()
            metrics.memory_efficiency = memory_metrics['efficiency']
            metrics.storage_efficiency = memory_metrics['storage']
            metrics.index_size_ratio = memory_metrics['size_ratio']
            
            # 4. 확장성 평가
            scalability_metrics = self._evaluate_scalability()
            metrics.scalability_score = scalability_metrics['score']
            metrics.query_throughput = scalability_metrics['throughput']
            
            # 5. 일관성 평가
            consistency_metrics = self._evaluate_consistency()
            metrics.consistency_score = consistency_metrics['overall']
            metrics.retrieval_stability = consistency_metrics['stability']
            
            # 6. 종합 점수 계산
            metrics.overall_performance = self._calculate_overall_performance(metrics)
            
            logger.info(f"Vector DB performance evaluation completed. Overall score: {metrics.overall_performance:.3f}")
            return metrics
            
        except Exception as e:
            logger.error(f"Error evaluating Vector DB performance: {e}")
            return VectorDBPerformanceMetrics()
    
    def _evaluate_index_quality(self) -> Dict[str, float]:
        """인덱스 품질 평가"""
        try:
            logger.info("Evaluating index quality...")
            
            # 벡터 데이터 수집
            vectors = self._collect_vectors()
            
            if not vectors or len(vectors) < 10:
                logger.warning("Insufficient vectors for quality evaluation")
                return {
                    'overall': 0.5,
                    'distribution': 0.5,
                    'clustering': 0.5,
                    'dimensionality': 0.5
                }
            
            # 1. 벡터 분포 품질
            distribution_score = self._analyze_vector_distribution(vectors)
            
            # 2. 클러스터링 품질
            clustering_score = self._analyze_clustering_quality(vectors)
            
            # 3. 차원 효율성
            dimensionality_score = self._analyze_dimensionality_efficiency(vectors)
            
            # 종합 인덱스 품질
            overall_quality = (
                distribution_score * 0.4 +
                clustering_score * 0.4 +
                dimensionality_score * 0.2
            )
            
            return {
                'overall': overall_quality,
                'distribution': distribution_score,
                'clustering': clustering_score,
                'dimensionality': dimensionality_score
            }
            
        except Exception as e:
            logger.warning(f"Error evaluating index quality: {e}")
            return {'overall': 0.0, 'distribution': 0.0, 'clustering': 0.0, 'dimensionality': 0.0}
    
    def _collect_vectors(self) -> np.ndarray:
        """벡터 데이터 수집"""
        try:
            # Chroma에서 모든 벡터 수집
            collection = self.vectorstore._collection
            
            # 모든 문서 ID 가져오기
            all_data = collection.get(include=['embeddings'])
            
            if all_data and all_data['embeddings']:
                vectors = np.array(all_data['embeddings'])
                logger.info(f"Collected {len(vectors)} vectors with dimension {vectors.shape[1] if len(vectors) > 0 else 0}")
                return vectors
            else:
                logger.warning("No vectors found in the collection")
                return np.array([])
                
        except Exception as e:
            logger.warning(f"Error collecting vectors: {e}")
            return np.array([])
    
    def _analyze_vector_distribution(self, vectors: np.ndarray) -> float:
        """벡터 분포 분석"""
        try:
            if len(vectors) == 0:
                return 0.0
            
            # 1. 벡터 정규화 확인
            norms = np.linalg.norm(vectors, axis=1)
            if len(norms) == 0 or np.mean(norms) == 0:
                norm_consistency = 0.0
            else:
                norm_consistency = max(0.0, 1.0 - np.std(norms) / (np.mean(norms) + 1e-8))
            
            # 2. 차원별 분산 분석
            dim_variances = np.var(vectors, axis=0)
            if len(dim_variances) == 0 or np.mean(dim_variances) == 0:
                variance_balance = 0.0
            else:
                variance_balance = max(0.0, 1.0 - (np.std(dim_variances) / (np.mean(dim_variances) + 1e-8)))
            
            # 3. 벡터 간 거리 분포
            sample_size = min(100, len(vectors))
            if sample_size < 2:
                distance_uniformity = 0.5
            else:
                sample_indices = np.random.choice(len(vectors), sample_size, replace=False)
                sample_vectors = vectors[sample_indices]
                
                distances = []
                for i in range(len(sample_vectors)):
                    for j in range(i + 1, len(sample_vectors)):
                        dist = np.linalg.norm(sample_vectors[i] - sample_vectors[j])
                        distances.append(dist)
                
                # 거리 분포의 균일성 (너무 가깝거나 너무 멀지 않아야 함)
                if distances and len(distances) > 1:
                    mean_dist = np.mean(distances)
                    if mean_dist > 0:
                        distance_uniformity = max(0.0, 1.0 - (np.std(distances) / (mean_dist + 1e-8)))
                    else:
                        distance_uniformity = 0.5
                else:
                    distance_uniformity = 0.5
            
            # 종합 분포 점수
            distribution_score = (
                norm_consistency * 0.3 +
                variance_balance * 0.4 +
                distance_uniformity * 0.3
            )
            
            return min(1.0, max(0.0, distribution_score))
            
        except Exception as e:
            logger.warning(f"Error analyzing vector distribution: {e}")
            return 0.0
    
    def _analyze_clustering_quality(self, vectors: np.ndarray) -> float:
        """클러스터링 품질 분석"""
        try:
            if len(vectors) < 10:
                return 0.5
            
            # K-means 클러스터링으로 자연스러운 클러스터 수 추정
            from sklearn.cluster import KMeans
            from sklearn.metrics import silhouette_score
            
            # 적절한 클러스터 수 범위
            max_clusters = min(10, len(vectors) // 3)
            cluster_range = range(2, max_clusters + 1)
            
            silhouette_scores = []
            inertias = []
            
            for n_clusters in cluster_range:
                try:
                    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
                    cluster_labels = kmeans.fit_predict(vectors)
                    
                    # 실루엣 점수 계산
                    silhouette_avg = silhouette_score(vectors, cluster_labels)
                    silhouette_scores.append(silhouette_avg)
                    
                    # 관성(inertia) 기록
                    inertias.append(kmeans.inertia_)
                    
                except Exception as e:
                    logger.warning(f"Error in clustering with {n_clusters} clusters: {e}")
                    continue
            
            if silhouette_scores:
                # 최고 실루엣 점수를 클러스터링 품질로 사용
                best_silhouette = max(silhouette_scores)
                # 실루엣 점수는 -1 ~ 1 범위이므로 0 ~ 1로 정규화
                clustering_quality = (best_silhouette + 1) / 2
            else:
                clustering_quality = 0.0
            
            return min(1.0, max(0.0, clustering_quality))
            
        except Exception as e:
            logger.warning(f"Error analyzing clustering quality: {e}")
            return 0.0
    
    def _analyze_dimensionality_efficiency(self, vectors: np.ndarray) -> float:
        """차원 효율성 분석"""
        try:
            if len(vectors) == 0:
                return 0.0
            
            # PCA를 통한 차원 효율성 분석
            from sklearn.decomposition import PCA
            
            # 원본 차원 수
            original_dim = vectors.shape[1]
            
            # PCA로 설명 가능한 분산 비율 계산
            pca = PCA()
            pca.fit(vectors)
            
            explained_variance_ratio = pca.explained_variance_ratio_
            cumulative_variance = np.cumsum(explained_variance_ratio)
            
            # 90% 분산을 설명하는데 필요한 차원 수
            dims_for_90_variance = np.argmax(cumulative_variance >= 0.9) + 1
            
            # 95% 분산을 설명하는데 필요한 차원 수
            dims_for_95_variance = np.argmax(cumulative_variance >= 0.95) + 1
            
            # 효율성 점수 계산
            efficiency_90 = dims_for_90_variance / original_dim
            efficiency_95 = dims_for_95_variance / original_dim
            
            # 낮은 비율일수록 효율적 (적은 차원으로 많은 정보 표현)
            dimensionality_efficiency = 1.0 - (efficiency_90 * 0.6 + efficiency_95 * 0.4)
            
            return min(1.0, max(0.0, dimensionality_efficiency))
            
        except Exception as e:
            logger.warning(f"Error analyzing dimensionality efficiency: {e}")
            return 0.0
    
    def _evaluate_search_performance(self) -> Dict[str, Any]:
        """검색 성능 평가"""
        try:
            logger.info("Evaluating search performance...")
            
            # 검색 속도 측정
            speed_scores = []
            precision_scores = {k: [] for k in self.k_values}
            recall_scores = {k: [] for k in self.k_values}
            
            for query in self.test_queries:
                # 검색 속도 측정
                start_time = time.time()
                
                try:
                    # 다양한 k 값으로 검색 수행
                    max_k = max(self.k_values)
                    results = self.vectorstore.similarity_search_with_score(query, k=max_k)
                    
                    search_time = time.time() - start_time
                    speed_scores.append(1.0 / (search_time + 0.1))  # 속도 점수 (빠를수록 높음)
                    
                    # 각 k 값에 대한 정밀도/재현율 계산
                    for k in self.k_values:
                        k_results = results[:k]
                        
                        # 기본적인 관련성 평가 (점수 임계값 기반)
                        relevant_results = [r for r in k_results if r[1] > 0.7]  # 높은 유사도만 관련성 있음으로 간주
                        
                        precision = len(relevant_results) / len(k_results) if k_results else 0.0
                        recall = min(1.0, len(relevant_results) / 3)  # 가정: 평균 3개의 관련 문서 존재
                        
                        precision_scores[k].append(precision)
                        recall_scores[k].append(recall)
                        
                except Exception as e:
                    logger.warning(f"Error in search for query '{query}': {e}")
                    continue
            
            # 평균 성능 계산
            avg_speed = np.mean(speed_scores) if speed_scores else 0.0
            avg_precision_at_k = {k: np.mean(scores) for k, scores in precision_scores.items()}
            avg_recall_at_k = {k: np.mean(scores) for k, scores in recall_scores.items()}
            
            # 종합 정확도 (precision@5와 recall@5의 조화평균)
            p5 = avg_precision_at_k.get(5, 0.0)
            r5 = avg_recall_at_k.get(5, 0.0)
            accuracy = 2 * p5 * r5 / (p5 + r5) if (p5 + r5) > 0 else 0.0
            
            return {
                'accuracy': accuracy,
                'speed': min(1.0, avg_speed),  # 정규화
                'precision_at_k': avg_precision_at_k,
                'recall_at_k': avg_recall_at_k
            }
            
        except Exception as e:
            logger.warning(f"Error evaluating search performance: {e}")
            return {
                'accuracy': 0.0,
                'speed': 0.0,
                'precision_at_k': {k: 0.0 for k in self.k_values},
                'recall_at_k': {k: 0.0 for k in self.k_values}
            }
    
    def _evaluate_memory_efficiency(self) -> Dict[str, float]:
        """메모리 효율성 평가"""
        try:
            logger.info("Evaluating memory efficiency...")
            
            # 벡터 스토어 크기 추정
            collection = self.vectorstore._collection
            doc_count = collection.count()
            
            if doc_count == 0:
                return {
                    'efficiency': 0.0,
                    'storage': 0.0,
                    'size_ratio': 0.0
                }
            
            # 메모리 사용량 (MB)
            if PSUTIL_AVAILABLE:
                process = psutil.Process()
                memory_info = process.memory_info()
                memory_usage_mb = memory_info.rss / 1024 / 1024
            else:
                # Fallback - estimate based on document count
                memory_usage_mb = doc_count * 0.1  # Rough estimate: 0.1MB per document
            
            # 문서당 메모리 사용량
            memory_per_doc = memory_usage_mb / doc_count
            
            # 효율성 점수 (작을수록 효율적)
            # 가정: 문서당 1MB 이하면 효율적
            memory_efficiency = max(0.0, 1.0 - memory_per_doc / 1.0)
            
            # 저장 공간 효율성
            try:
                # 벡터 스토어 디렉토리 크기 계산
                vector_dir = Path(self.vectorstore._persist_directory)
                if vector_dir.exists():
                    total_size = sum(f.stat().st_size for f in vector_dir.rglob('*') if f.is_file())
                    storage_size_mb = total_size / 1024 / 1024
                    storage_per_doc = storage_size_mb / doc_count
                    
                    # 저장 효율성 (문서당 100KB 이하면 효율적)
                    storage_efficiency = max(0.0, 1.0 - storage_per_doc / 0.1)
                    
                    # 크기 비율 (메모리 대비 디스크 크기)
                    size_ratio = storage_size_mb / memory_usage_mb if memory_usage_mb > 0 else 0.0
                else:
                    storage_efficiency = 0.5
                    size_ratio = 0.0
                    
            except Exception as e:
                logger.warning(f"Error calculating storage efficiency: {e}")
                storage_efficiency = 0.5
                size_ratio = 0.0
            
            return {
                'efficiency': memory_efficiency,
                'storage': storage_efficiency,
                'size_ratio': min(1.0, size_ratio)  # 정규화
            }
            
        except Exception as e:
            logger.warning(f"Error evaluating memory efficiency: {e}")
            return {'efficiency': 0.0, 'storage': 0.0, 'size_ratio': 0.0}
    
    def _evaluate_scalability(self) -> Dict[str, float]:
        """확장성 평가"""
        try:
            logger.info("Evaluating scalability...")
            
            # 현재 문서 수
            doc_count = self.vectorstore._collection.count()
            
            # 쿼리 처리량 측정
            start_time = time.time()
            query_count = 0
            
            # 10초 동안 최대한 많은 쿼리 처리
            test_duration = 5  # 5초로 단축
            end_time = start_time + test_duration
            
            sample_queries = self.test_queries[:5]  # 샘플 쿼리 제한
            
            while time.time() < end_time and query_count < 50:  # 최대 50개 쿼리
                for query in sample_queries:
                    if time.time() >= end_time:
                        break
                    try:
                        self.vectorstore.similarity_search(query, k=3)
                        query_count += 1
                    except Exception as e:
                        logger.warning(f"Query failed during throughput test: {e}")
                        break
            
            actual_duration = time.time() - start_time
            throughput = query_count / actual_duration if actual_duration > 0 else 0.0
            
            # 확장성 점수 계산
            # 문서 수와 처리량을 고려한 점수
            doc_scale_factor = min(1.0, doc_count / 1000)  # 1000개 문서까지는 선형
            throughput_score = min(1.0, throughput / 10)  # 초당 10쿼리를 기준
            
            scalability_score = (doc_scale_factor * 0.4 + throughput_score * 0.6)
            
            return {
                'score': scalability_score,
                'throughput': throughput
            }
            
        except Exception as e:
            logger.warning(f"Error evaluating scalability: {e}")
            return {'score': 0.0, 'throughput': 0.0}
    
    def _evaluate_consistency(self) -> Dict[str, float]:
        """일관성 평가"""
        try:
            logger.info("Evaluating consistency...")
            
            # 동일 쿼리 반복 실행으로 일관성 측정
            consistency_scores = []
            
            for query in self.test_queries[:3]:  # 처음 3개 쿼리만 테스트
                results_list = []
                
                # 동일 쿼리를 5번 실행
                for _ in range(5):
                    try:
                        results = self.vectorstore.similarity_search_with_score(query, k=5)
                        # 결과의 ID 순서만 저장 (점수는 약간 변할 수 있음)
                        result_ids = [doc.metadata.get('chunk_id', hash(doc.page_content)) for doc, score in results]
                        results_list.append(result_ids)
                    except Exception as e:
                        logger.warning(f"Consistency test failed for query '{query}': {e}")
                        continue
                
                if len(results_list) >= 2:
                    # 결과 간 유사도 계산
                    similarities = []
                    for i in range(len(results_list)):
                        for j in range(i + 1, len(results_list)):
                            # 순서 기반 유사도 계산
                            similarity = self._calculate_list_similarity(results_list[i], results_list[j])
                            similarities.append(similarity)
                    
                    if similarities:
                        consistency_scores.append(np.mean(similarities))
            
            consistency_score = np.mean(consistency_scores) if consistency_scores else 0.0
            
            # 검색 안정성 평가 (결과 점수의 분산)
            stability_scores = []
            for query in self.test_queries[:3]:
                score_variations = []
                try:
                    for _ in range(3):  # 3번 실행
                        results = self.vectorstore.similarity_search_with_score(query, k=3)
                        scores = [score for doc, score in results]
                        if scores:
                            score_variations.extend(scores)
                    
                    if len(score_variations) > 1:
                        score_std = np.std(score_variations)
                        score_mean = np.mean(score_variations)
                        stability = 1.0 - (score_std / (score_mean + 1e-8))
                        stability_scores.append(max(0.0, stability))
                        
                except Exception as e:
                    logger.warning(f"Stability test failed for query '{query}': {e}")
                    continue
            
            stability_score = np.mean(stability_scores) if stability_scores else 0.0
            
            return {
                'overall': (consistency_score * 0.6 + stability_score * 0.4),
                'stability': stability_score
            }
            
        except Exception as e:
            logger.warning(f"Error evaluating consistency: {e}")
            return {'overall': 0.0, 'stability': 0.0}
    
    def _calculate_list_similarity(self, list1: List, list2: List) -> float:
        """두 리스트의 순서 기반 유사도 계산"""
        if not list1 or not list2:
            return 0.0
        
        # 교집합 계산
        set1, set2 = set(list1), set(list2)
        intersection = len(set1.intersection(set2))
        union = len(set1.union(set2))
        
        # Jaccard 유사도와 순서 유사도 결합
        jaccard = intersection / union if union > 0 else 0.0
        
        # 순서 유사도 (상위 결과의 순서가 얼마나 비슷한지)
        order_similarity = 0.0
        min_len = min(len(list1), len(list2))
        if min_len > 0:
            order_matches = sum(1 for i in range(min_len) if list1[i] == list2[i])
            order_similarity = order_matches / min_len
        
        return (jaccard * 0.7 + order_similarity * 0.3)
    
    def _calculate_overall_performance(self, metrics: VectorDBPerformanceMetrics) -> float:
        """종합 성능 점수 계산"""
        try:
            # 가중치 정의
            weights = {
                'index_quality': 0.25,      # 인덱스 품질
                'search_accuracy': 0.25,    # 검색 정확도
                'search_speed': 0.15,       # 검색 속도
                'memory_efficiency': 0.15,  # 메모리 효율성
                'scalability': 0.10,        # 확장성
                'consistency': 0.10         # 일관성
            }
            
            # 가중 평균 계산
            total_score = (
                metrics.index_quality * weights['index_quality'] +
                metrics.search_accuracy * weights['search_accuracy'] +
                metrics.search_speed * weights['search_speed'] +
                metrics.memory_efficiency * weights['memory_efficiency'] +
                metrics.scalability_score * weights['scalability'] +
                metrics.consistency_score * weights['consistency']
            )
            
            return min(1.0, max(0.0, total_score))
            
        except Exception as e:
            logger.warning(f"Error calculating overall performance: {e}")
            return 0.0
    
    def _generate_default_queries(self) -> List[str]:
        """기본 테스트 쿼리 생성"""
        return [
            "시스템 설치 방법",
            "API 사용법",
            "에러 해결 방법",
            "설정 변경",
            "성능 최적화",
            "데이터베이스 연결",
            "보안 설정",
            "로그 분석"
        ]
    
    def generate_performance_report(
        self, 
        metrics: VectorDBPerformanceMetrics
    ) -> Dict[str, Any]:
        """Vector DB 성능 리포트 생성"""
        try:
            doc_count = self.vectorstore._collection.count()
            
            report = {
                'summary': {
                    'overall_performance': metrics.overall_performance,
                    'total_documents': doc_count,
                    'performance_grade': self._get_performance_grade(metrics.overall_performance),
                    'best_metrics': self._identify_best_metrics(metrics),
                    'worst_metrics': self._identify_worst_metrics(metrics)
                },
                'detailed_metrics': metrics.to_dict(),
                'recommendations': self._generate_recommendations(metrics),
                'technical_details': {
                    'index_stats': self._get_index_stats(),
                    'query_performance': {
                        'avg_query_time': np.mean(self.query_times) if self.query_times else 0.0,
                        'throughput': metrics.query_throughput
                    }
                }
            }
            
            return report
            
        except Exception as e:
            logger.error(f"Error generating performance report: {e}")
            return {}
    
    def _get_performance_grade(self, score: float) -> str:
        """성능 점수를 등급으로 변환"""
        if score >= 0.9:
            return "Excellent"
        elif score >= 0.8:
            return "Very Good"
        elif score >= 0.7:
            return "Good"
        elif score >= 0.6:
            return "Fair"
        elif score >= 0.5:
            return "Poor"
        else:
            return "Very Poor"
    
    def _identify_best_metrics(self, metrics: VectorDBPerformanceMetrics) -> List[str]:
        """가장 좋은 메트릭들 식별"""
        metric_scores = {
            'Index Quality': metrics.index_quality,
            'Search Accuracy': metrics.search_accuracy,
            'Search Speed': metrics.search_speed,
            'Memory Efficiency': metrics.memory_efficiency,
            'Scalability': metrics.scalability_score,
            'Consistency': metrics.consistency_score
        }
        
        # 상위 3개 메트릭 반환
        sorted_metrics = sorted(metric_scores.items(), key=lambda x: x[1], reverse=True)
        return [name for name, score in sorted_metrics[:3] if score > 0.7]
    
    def _identify_worst_metrics(self, metrics: VectorDBPerformanceMetrics) -> List[str]:
        """가장 나쁜 메트릭들 식별"""
        metric_scores = {
            'Index Quality': metrics.index_quality,
            'Search Accuracy': metrics.search_accuracy,
            'Search Speed': metrics.search_speed,
            'Memory Efficiency': metrics.memory_efficiency,
            'Scalability': metrics.scalability_score,
            'Consistency': metrics.consistency_score
        }
        
        # 하위 메트릭들 반환 (0.6 미만)
        return [name for name, score in metric_scores.items() if score < 0.6]
    
    def _get_index_stats(self) -> Dict[str, Any]:
        """인덱스 통계 정보"""
        try:
            collection = self.vectorstore._collection
            
            return {
                'document_count': collection.count(),
                'collection_name': collection.name,
                'embedding_dimension': self._get_embedding_dimension()
            }
            
        except Exception as e:
            logger.warning(f"Error getting index stats: {e}")
            return {}
    
    def _get_embedding_dimension(self) -> int:
        """임베딩 차원 수 반환"""
        try:
            collection = self.vectorstore._collection
            sample_data = collection.get(limit=1, include=['embeddings'])
            
            if sample_data and sample_data['embeddings']:
                return len(sample_data['embeddings'][0])
            else:
                return 0
                
        except Exception as e:
            logger.warning(f"Error getting embedding dimension: {e}")
            return 0
    
    def _generate_recommendations(self, metrics: VectorDBPerformanceMetrics) -> List[str]:
        """개선 권장사항 생성"""
        recommendations = []
        
        if metrics.index_quality < 0.7:
            recommendations.append("인덱스 품질이 낮습니다. 벡터 정규화나 차원 축소를 고려하세요.")
        
        if metrics.search_accuracy < 0.7:
            recommendations.append("검색 정확도가 낮습니다. 임베딩 모델 업그레이드나 하이브리드 검색을 고려하세요.")
        
        if metrics.search_speed < 0.6:
            recommendations.append("검색 속도가 느립니다. 인덱스 최적화나 하드웨어 업그레이드를 고려하세요.")
        
        if metrics.memory_efficiency < 0.6:
            recommendations.append("메모리 효율성이 낮습니다. 벡터 압축이나 배치 처리를 고려하세요.")
        
        if metrics.scalability_score < 0.6:
            recommendations.append("확장성이 부족합니다. 분산 인덱싱이나 샤딩을 고려하세요.")
        
        if metrics.consistency_score < 0.7:
            recommendations.append("일관성이 부족합니다. 인덱스 업데이트 정책을 검토하세요.")
        
        if not recommendations:
            recommendations.append("Vector DB 성능이 우수합니다. 현재 설정을 유지하세요.")
        
        return recommendations 