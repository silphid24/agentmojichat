"""
청킹 품질 평가기 (Chunk Quality Evaluator)
청크의 의미적 일관성, 경계 품질, 정보 커버리지를 종합적으로 평가
"""

import re
import math
import numpy as np
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass
from pathlib import Path
import nltk
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from collections import Counter

from langchain.schema import Document
from app.core.logging import logger


@dataclass
class ChunkQualityMetrics:
    """청크 품질 메트릭"""
    
    # 의미적 일관성
    semantic_coherence: float = 0.0
    intra_chunk_similarity: float = 0.0
    
    # 경계 품질
    boundary_quality: float = 0.0
    sentence_boundary_score: float = 0.0
    paragraph_boundary_score: float = 0.0
    
    # 정보 커버리지
    information_coverage: float = 0.0
    overlap_efficiency: float = 0.0
    redundancy_score: float = 0.0
    
    # 구조 품질
    structure_preservation: float = 0.0
    heading_alignment: float = 0.0
    
    # 크기 일관성
    size_consistency: float = 0.0
    optimal_size_ratio: float = 0.0
    
    # 종합 점수
    overall_quality: float = 0.0

    def to_dict(self) -> Dict[str, float]:
        """딕셔너리로 변환"""
        return {
            'semantic_coherence': self.semantic_coherence,
            'intra_chunk_similarity': self.intra_chunk_similarity,
            'boundary_quality': self.boundary_quality,
            'sentence_boundary_score': self.sentence_boundary_score,
            'paragraph_boundary_score': self.paragraph_boundary_score,
            'information_coverage': self.information_coverage,
            'overlap_efficiency': self.overlap_efficiency,
            'redundancy_score': self.redundancy_score,
            'structure_preservation': self.structure_preservation,
            'heading_alignment': self.heading_alignment,
            'size_consistency': self.size_consistency,
            'optimal_size_ratio': self.optimal_size_ratio,
            'overall_quality': self.overall_quality
        }


class ChunkQualityEvaluator:
    """청킹 품질 종합 평가기"""
    
    def __init__(
        self,
        embedding_model: str = "all-MiniLM-L6-v2",
        optimal_chunk_size: int = 1000,
        size_tolerance: float = 0.3
    ):
        self.embedding_model_name = embedding_model
        self.optimal_chunk_size = optimal_chunk_size
        self.size_tolerance = size_tolerance
        
        # 모델 초기화
        self.sentence_model = None
        self._initialize_models()
        
        # 구조 패턴 정의
        self.structure_patterns = {
            'heading': re.compile(r'^#{1,6}\s+(.+)$', re.MULTILINE),
            'list_item': re.compile(r'^\s*[-*+]\s+(.+)$', re.MULTILINE),
            'numbered_list': re.compile(r'^\s*\d+\.\s+(.+)$', re.MULTILINE),
            'code_block': re.compile(r'```[\s\S]*?```', re.MULTILINE),
            'quote': re.compile(r'^>\s+(.+)$', re.MULTILINE)
        }
    
    def _initialize_models(self):
        """모델 초기화"""
        try:
            # NLTK 데이터 확인
            try:
                nltk.data.find('tokenizers/punkt')
            except LookupError:
                nltk.download('punkt', quiet=True)
            
            # 문장 임베딩 모델 로드
            self.sentence_model = SentenceTransformer(self.embedding_model_name)
            logger.info(f"Chunk quality evaluator initialized with model: {self.embedding_model_name}")
            
        except Exception as e:
            logger.error(f"Failed to initialize chunk quality evaluator: {e}")
            raise
    
    def evaluate_chunks(
        self,
        chunks: List[Document],
        original_text: str = None
    ) -> ChunkQualityMetrics:
        """청크 품질 종합 평가"""
        
        try:
            if not chunks:
                logger.warning("No chunks provided for evaluation")
                return ChunkQualityMetrics()
            
            metrics = ChunkQualityMetrics()
            
            # 1. 의미적 일관성 평가
            metrics.semantic_coherence = self._evaluate_semantic_coherence(chunks)
            metrics.intra_chunk_similarity = self._evaluate_intra_chunk_similarity(chunks)
            
            # 2. 경계 품질 평가
            if original_text:
                boundary_metrics = self._evaluate_boundary_quality(chunks, original_text)
                metrics.boundary_quality = boundary_metrics['overall']
                metrics.sentence_boundary_score = boundary_metrics['sentence']
                metrics.paragraph_boundary_score = boundary_metrics['paragraph']
            
            # 3. 정보 커버리지 평가
            coverage_metrics = self._evaluate_information_coverage(chunks, original_text)
            metrics.information_coverage = coverage_metrics['coverage']
            metrics.overlap_efficiency = coverage_metrics['efficiency']
            metrics.redundancy_score = coverage_metrics['redundancy']
            
            # 4. 구조 품질 평가
            if original_text:
                structure_metrics = self._evaluate_structure_preservation(chunks, original_text)
                metrics.structure_preservation = structure_metrics['overall']
                metrics.heading_alignment = structure_metrics['heading']
            
            # 5. 크기 일관성 평가
            size_metrics = self._evaluate_size_consistency(chunks)
            metrics.size_consistency = size_metrics['consistency']
            metrics.optimal_size_ratio = size_metrics['optimal_ratio']
            
            # 6. 종합 점수 계산
            metrics.overall_quality = self._calculate_overall_quality(metrics)
            
            logger.info(f"Chunk quality evaluation completed. Overall quality: {metrics.overall_quality:.3f}")
            return metrics
            
        except Exception as e:
            logger.error(f"Error evaluating chunk quality: {e}")
            return ChunkQualityMetrics()
    
    def _evaluate_semantic_coherence(self, chunks: List[Document]) -> float:
        """의미적 일관성 평가"""
        try:
            coherence_scores = []
            
            for chunk in chunks:
                # 청크 내 문장들로 분할
                sentences = nltk.sent_tokenize(chunk.page_content)
                
                if len(sentences) < 2:
                    coherence_scores.append(1.0)  # 단일 문장은 완벽한 일관성
                    continue
                
                # 문장 임베딩 생성
                embeddings = self.sentence_model.encode(sentences)
                
                # 연속 문장 간 유사도 계산
                similarities = []
                for i in range(len(embeddings) - 1):
                    sim = cosine_similarity([embeddings[i]], [embeddings[i + 1]])[0][0]
                    similarities.append(sim)
                
                # 청크 내 평균 일관성
                chunk_coherence = np.mean(similarities) if similarities else 0.0
                coherence_scores.append(chunk_coherence)
            
            return float(np.mean(coherence_scores)) if coherence_scores else 0.0
            
        except Exception as e:
            logger.warning(f"Error calculating semantic coherence: {e}")
            return 0.0
    
    def _evaluate_intra_chunk_similarity(self, chunks: List[Document]) -> float:
        """청크 내 문장 유사도 분포 평가"""
        try:
            similarity_scores = []
            
            for chunk in chunks:
                sentences = nltk.sent_tokenize(chunk.page_content)
                
                if len(sentences) < 2:
                    continue
                
                embeddings = self.sentence_model.encode(sentences)
                similarity_matrix = cosine_similarity(embeddings)
                
                # 대각선 제외한 평균 유사도
                mask = ~np.eye(similarity_matrix.shape[0], dtype=bool)
                avg_similarity = similarity_matrix[mask].mean()
                similarity_scores.append(avg_similarity)
            
            return float(np.mean(similarity_scores)) if similarity_scores else 0.0
            
        except Exception as e:
            logger.warning(f"Error calculating intra-chunk similarity: {e}")
            return 0.0
    
    def _evaluate_boundary_quality(
        self, 
        chunks: List[Document], 
        original_text: str
    ) -> Dict[str, float]:
        """경계 품질 평가"""
        try:
            # 원본 텍스트의 자연스러운 경계점 식별
            sentences = nltk.sent_tokenize(original_text)
            paragraphs = original_text.split('\n\n')
            
            sentence_boundary_score = 0.0
            paragraph_boundary_score = 0.0
            
            # 청크 경계가 문장/문단 경계와 얼마나 일치하는지 확인
            total_boundaries = len(chunks) - 1
            
            if total_boundaries > 0:
                sentence_alignments = 0
                paragraph_alignments = 0
                
                current_pos = 0
                for i, chunk in enumerate(chunks[:-1]):  # 마지막 청크 제외
                    chunk_end_pos = current_pos + len(chunk.page_content)
                    
                    # 이 위치가 문장 경계인지 확인
                    for sentence in sentences:
                        sentence_end = original_text.find(sentence) + len(sentence)
                        if abs(sentence_end - chunk_end_pos) < 10:  # 10자 이내 허용
                            sentence_alignments += 1
                            break
                    
                    # 이 위치가 문단 경계인지 확인
                    for paragraph in paragraphs:
                        paragraph_end = original_text.find(paragraph) + len(paragraph)
                        if abs(paragraph_end - chunk_end_pos) < 10:  # 10자 이내 허용
                            paragraph_alignments += 1
                            break
                    
                    current_pos = chunk_end_pos
                
                sentence_boundary_score = sentence_alignments / total_boundaries
                paragraph_boundary_score = paragraph_alignments / total_boundaries
            
            overall_boundary = (sentence_boundary_score * 0.6 + paragraph_boundary_score * 0.4)
            
            return {
                'overall': overall_boundary,
                'sentence': sentence_boundary_score,
                'paragraph': paragraph_boundary_score
            }
            
        except Exception as e:
            logger.warning(f"Error evaluating boundary quality: {e}")
            return {'overall': 0.0, 'sentence': 0.0, 'paragraph': 0.0}
    
    def _evaluate_information_coverage(
        self, 
        chunks: List[Document], 
        original_text: str = None
    ) -> Dict[str, float]:
        """정보 커버리지 평가"""
        try:
            # 1. 중복도 계산
            redundancy_score = self._calculate_redundancy(chunks)
            
            # 2. 중복 효율성 계산
            overlap_efficiency = self._calculate_overlap_efficiency(chunks)
            
            # 3. 전체 커버리지 계산
            coverage = 1.0  # 기본값
            if original_text:
                total_original_length = len(original_text)
                total_chunk_length = sum(len(chunk.page_content) for chunk in chunks)
                coverage = min(1.0, total_chunk_length / total_original_length)
            
            return {
                'coverage': coverage,
                'efficiency': overlap_efficiency,
                'redundancy': 1.0 - redundancy_score  # 중복이 낮을수록 좋음
            }
            
        except Exception as e:
            logger.warning(f"Error evaluating information coverage: {e}")
            return {'coverage': 0.0, 'efficiency': 0.0, 'redundancy': 0.0}
    
    def _calculate_redundancy(self, chunks: List[Document]) -> float:
        """청크 간 중복도 계산"""
        try:
            if len(chunks) < 2:
                return 0.0
            
            redundancy_scores = []
            
            for i in range(len(chunks)):
                for j in range(i + 1, len(chunks)):
                    # 두 청크의 단어 집합
                    words_i = set(chunks[i].page_content.lower().split())
                    words_j = set(chunks[j].page_content.lower().split())
                    
                    # Jaccard 유사도 계산
                    intersection = len(words_i.intersection(words_j))
                    union = len(words_i.union(words_j))
                    
                    if union > 0:
                        jaccard = intersection / union
                        redundancy_scores.append(jaccard)
            
            return float(np.mean(redundancy_scores)) if redundancy_scores else 0.0
            
        except Exception as e:
            logger.warning(f"Error calculating redundancy: {e}")
            return 0.0
    
    def _calculate_overlap_efficiency(self, chunks: List[Document]) -> float:
        """중복 효율성 계산"""
        try:
            if len(chunks) < 2:
                return 1.0
            
            # 의도적인 중복(overlap)과 불필요한 중복 구분
            # 연속된 청크 간의 중복은 의도적, 비연속 청크 간 중복은 비효율
            
            adjacent_overlaps = []
            non_adjacent_overlaps = []
            
            for i in range(len(chunks)):
                for j in range(i + 1, len(chunks)):
                    words_i = set(chunks[i].page_content.lower().split())
                    words_j = set(chunks[j].page_content.lower().split())
                    
                    if len(words_i.union(words_j)) > 0:
                        overlap = len(words_i.intersection(words_j)) / len(words_i.union(words_j))
                        
                        if j == i + 1:  # 인접한 청크
                            adjacent_overlaps.append(overlap)
                        else:  # 비인접 청크
                            non_adjacent_overlaps.append(overlap)
            
            # 인접 중복은 적절히 있어야 하고, 비인접 중복은 적어야 함
            adjacent_score = np.mean(adjacent_overlaps) if adjacent_overlaps else 0.0
            non_adjacent_score = np.mean(non_adjacent_overlaps) if non_adjacent_overlaps else 0.0
            
            # 적절한 인접 중복(0.1-0.3)과 낮은 비인접 중복을 선호
            optimal_adjacent = min(1.0, adjacent_score / 0.2) if adjacent_score <= 0.2 else max(0.0, 1.0 - (adjacent_score - 0.2) / 0.3)
            penalty_non_adjacent = max(0.0, 1.0 - non_adjacent_score * 2)
            
            return (optimal_adjacent + penalty_non_adjacent) / 2
            
        except Exception as e:
            logger.warning(f"Error calculating overlap efficiency: {e}")
            return 0.0
    
    def _evaluate_structure_preservation(
        self, 
        chunks: List[Document], 
        original_text: str
    ) -> Dict[str, float]:
        """구조 보존 평가"""
        try:
            # 원본 텍스트의 구조 요소 추출
            original_structures = self._extract_structures(original_text)
            
            # 청크별 구조 요소 추출
            chunk_structures = []
            for chunk in chunks:
                chunk_structures.append(self._extract_structures(chunk.page_content))
            
            # 구조 보존 점수 계산
            heading_preservation = self._calculate_heading_preservation(
                original_structures, chunk_structures
            )
            
            # 전체 구조 보존 점수
            overall_preservation = heading_preservation  # 향후 다른 구조 요소 추가 가능
            
            return {
                'overall': overall_preservation,
                'heading': heading_preservation
            }
            
        except Exception as e:
            logger.warning(f"Error evaluating structure preservation: {e}")
            return {'overall': 0.0, 'heading': 0.0}
    
    def _extract_structures(self, text: str) -> Dict[str, List[str]]:
        """텍스트에서 구조 요소 추출"""
        structures = {}
        
        for structure_type, pattern in self.structure_patterns.items():
            matches = pattern.findall(text)
            structures[structure_type] = matches
        
        return structures
    
    def _calculate_heading_preservation(
        self, 
        original_structures: Dict[str, List[str]], 
        chunk_structures: List[Dict[str, List[str]]]
    ) -> float:
        """제목 구조 보존 점수 계산"""
        original_headings = original_structures.get('heading', [])
        
        if not original_headings:
            return 1.0  # 원본에 제목이 없으면 완벽 점수
        
        # 청크에서 발견된 제목들
        found_headings = []
        for chunk_struct in chunk_structures:
            found_headings.extend(chunk_struct.get('heading', []))
        
        # 제목 보존 비율
        preserved_count = 0
        for original_heading in original_headings:
            if any(original_heading.strip() in found_heading for found_heading in found_headings):
                preserved_count += 1
        
        return preserved_count / len(original_headings) if original_headings else 1.0
    
    def _evaluate_size_consistency(self, chunks: List[Document]) -> Dict[str, float]:
        """크기 일관성 평가"""
        try:
            chunk_sizes = [len(chunk.page_content) for chunk in chunks]
            
            if not chunk_sizes:
                return {'consistency': 0.0, 'optimal_ratio': 0.0}
            
            # 1. 크기 일관성 (표준편차 기반)
            mean_size = np.mean(chunk_sizes)
            std_size = np.std(chunk_sizes)
            
            # 변동계수 계산 (낮을수록 일관성이 높음)
            cv = std_size / mean_size if mean_size > 0 else 0
            consistency = max(0.0, 1.0 - cv)  # 변동계수가 0에 가까우면 1점
            
            # 2. 최적 크기 비율
            optimal_count = sum(
                1 for size in chunk_sizes 
                if abs(size - self.optimal_chunk_size) <= self.optimal_chunk_size * self.size_tolerance
            )
            optimal_ratio = optimal_count / len(chunk_sizes)
            
            return {
                'consistency': consistency,
                'optimal_ratio': optimal_ratio
            }
            
        except Exception as e:
            logger.warning(f"Error evaluating size consistency: {e}")
            return {'consistency': 0.0, 'optimal_ratio': 0.0}
    
    def _calculate_overall_quality(self, metrics: ChunkQualityMetrics) -> float:
        """종합 품질 점수 계산"""
        try:
            # 가중치 정의 (중요도에 따라 조정 가능)
            weights = {
                'semantic_coherence': 0.25,      # 의미적 일관성 (가장 중요)
                'boundary_quality': 0.20,       # 경계 품질
                'information_coverage': 0.20,   # 정보 커버리지
                'overlap_efficiency': 0.15,     # 중복 효율성
                'structure_preservation': 0.10, # 구조 보존
                'size_consistency': 0.10        # 크기 일관성
            }
            
            # 가중 평균 계산
            total_score = (
                metrics.semantic_coherence * weights['semantic_coherence'] +
                metrics.boundary_quality * weights['boundary_quality'] +
                metrics.information_coverage * weights['information_coverage'] +
                metrics.overlap_efficiency * weights['overlap_efficiency'] +
                metrics.structure_preservation * weights['structure_preservation'] +
                metrics.size_consistency * weights['size_consistency']
            )
            
            return min(1.0, max(0.0, total_score))  # 0-1 범위로 클램핑
            
        except Exception as e:
            logger.warning(f"Error calculating overall quality: {e}")
            return 0.0
    
    def generate_quality_report(
        self, 
        metrics: ChunkQualityMetrics,
        chunks: List[Document]
    ) -> Dict[str, Any]:
        """청킹 품질 리포트 생성"""
        try:
            chunk_count = len(chunks)
            chunk_sizes = [len(chunk.page_content) for chunk in chunks]
            
            report = {
                'summary': {
                    'overall_quality': metrics.overall_quality,
                    'total_chunks': chunk_count,
                    'avg_chunk_size': np.mean(chunk_sizes) if chunk_sizes else 0,
                    'quality_grade': self._get_quality_grade(metrics.overall_quality)
                },
                'detailed_metrics': metrics.to_dict(),
                'recommendations': self._generate_recommendations(metrics),
                'chunk_statistics': {
                    'min_size': min(chunk_sizes) if chunk_sizes else 0,
                    'max_size': max(chunk_sizes) if chunk_sizes else 0,
                    'size_std': np.std(chunk_sizes) if chunk_sizes else 0,
                    'total_length': sum(chunk_sizes)
                }
            }
            
            return report
            
        except Exception as e:
            logger.error(f"Error generating quality report: {e}")
            return {}
    
    def _get_quality_grade(self, score: float) -> str:
        """품질 점수를 등급으로 변환"""
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
    
    def _generate_recommendations(self, metrics: ChunkQualityMetrics) -> List[str]:
        """개선 권장사항 생성"""
        recommendations = []
        
        if metrics.semantic_coherence < 0.7:
            recommendations.append("의미적 일관성이 낮습니다. 더 엄격한 유사도 임계값을 사용하거나 의미 기반 청킹을 고려하세요.")
        
        if metrics.boundary_quality < 0.6:
            recommendations.append("청크 경계가 자연스럽지 않습니다. 문장이나 문단 경계에서 분할하도록 조정하세요.")
        
        if metrics.overlap_efficiency < 0.6:
            recommendations.append("중복 효율성이 낮습니다. 인접 청크 간 적절한 오버랩(10-30%)을 유지하세요.")
        
        if metrics.size_consistency < 0.7:
            recommendations.append("청크 크기가 일관성이 부족합니다. 최대/최소 크기 제한을 조정하세요.")
        
        if metrics.structure_preservation < 0.8:
            recommendations.append("문서 구조 보존이 부족합니다. 제목이나 섹션 구조를 고려한 청킹 전략을 사용하세요.")
        
        if not recommendations:
            recommendations.append("청킹 품질이 우수합니다. 현재 설정을 유지하세요.")
        
        return recommendations 