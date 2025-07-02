"""
적응형 기능 관리 시스템
쿼리 복잡도와 상황에 따라 RAG 기능을 선택적으로 활성화
"""

import re
import time
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

from app.core.logging import logger


class QueryComplexity(Enum):
    """쿼리 복잡도 수준"""
    SIMPLE = "simple"
    MEDIUM = "medium"
    COMPLEX = "complex"


class ProcessingMode(Enum):
    """처리 모드"""
    FAST = "fast"           # 빠른 응답 우선
    BALANCED = "balanced"   # 속도와 품질 균형
    QUALITY = "quality"     # 품질 우선


@dataclass
class QueryAnalysis:
    """쿼리 분석 결과"""
    complexity: QueryComplexity
    keyword_count: int
    has_specific_terms: bool
    has_compound_concepts: bool
    estimated_processing_time: float
    recommended_mode: ProcessingMode
    features_to_enable: Dict[str, bool]


class AdaptiveFeatureManager:
    """적응형 기능 관리자"""
    
    def __init__(self):
        # 복잡도 판단 기준
        self.complexity_thresholds = {
            "simple": {"max_keywords": 3, "max_length": 50},
            "medium": {"max_keywords": 7, "max_length": 100},
            "complex": {"max_keywords": float('inf'), "max_length": float('inf')}
        }
        
        # 특정 용어 패턴
        self.specific_terms = [
            # 기술 용어
            r'\b(API|SDK|데이터베이스|아키텍처|프레임워크|알고리즘)\b',
            # 회사/제품 고유명사
            r'\b(MOJI|SMHACCP|에이전트|시스템)\b',
            # 기능 관련
            r'\b(기능|특징|장점|단점|비교|분석)\b'
        ]
        
        # 복합 개념 패턴
        self.compound_patterns = [
            r'(.+)\s+(와|과|및|그리고)\s+(.+)',  # A와 B
            r'(.+)\s+(vs|대비|비교)\s+(.+)',     # A vs B
            r'(.+)\s+(관련|연관|관계)\s+(.+)',   # A 관련 B
            r'어떻게|어떤|무엇|왜|언제|어디',      # 의문문
        ]
        
        # 성능 히스토리
        self.performance_history = []
        self.max_history = 100
    
    def analyze_query(self, query: str, context: Optional[Dict[str, Any]] = None) -> QueryAnalysis:
        """쿼리 분석 및 처리 방식 추천"""
        try:
            # 기본 통계
            query_length = len(query)
            keywords = self._extract_keywords(query)
            keyword_count = len(keywords)
            
            # 특정 용어 검사
            has_specific_terms = self._has_specific_terms(query)
            
            # 복합 개념 검사
            has_compound_concepts = self._has_compound_concepts(query)
            
            # 복잡도 판단
            complexity = self._determine_complexity(
                query_length, keyword_count, has_specific_terms, has_compound_concepts
            )
            
            # 처리 시간 추정
            estimated_time = self._estimate_processing_time(complexity, keyword_count)
            
            # 처리 모드 추천
            recommended_mode = self._recommend_processing_mode(
                complexity, estimated_time, context
            )
            
            # 기능 활성화 결정
            features_to_enable = self._decide_features(
                complexity, recommended_mode, has_specific_terms, has_compound_concepts
            )
            
            analysis = QueryAnalysis(
                complexity=complexity,
                keyword_count=keyword_count,
                has_specific_terms=has_specific_terms,
                has_compound_concepts=has_compound_concepts,
                estimated_processing_time=estimated_time,
                recommended_mode=recommended_mode,
                features_to_enable=features_to_enable
            )
            
            logger.info(f"Query analysis: {complexity.value} complexity, {keyword_count} keywords, mode: {recommended_mode.value}")
            return analysis
            
        except Exception as e:
            logger.error(f"Query analysis error: {e}")
            # 기본값 반환
            return QueryAnalysis(
                complexity=QueryComplexity.MEDIUM,
                keyword_count=0,
                has_specific_terms=False,
                has_compound_concepts=False,
                estimated_processing_time=2.0,
                recommended_mode=ProcessingMode.BALANCED,
                features_to_enable={
                    "query_rewriting": True,
                    "parallel_search": True,
                    "reranking": True,
                    "hybrid_search": True
                }
            )
    
    def _extract_keywords(self, query: str) -> List[str]:
        """키워드 추출"""
        # 한글, 영문, 숫자만 추출하고 불용어 제거
        words = re.findall(r'[가-힣a-zA-Z0-9]+', query)
        
        # 불용어 제거
        stopwords = {
            '은', '는', '이', '가', '을', '를', '에', '에서', '로', '으로',
            '의', '와', '과', '그리고', '또한', '그런데', '하지만',
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at'
        }
        
        keywords = [word for word in words if word.lower() not in stopwords and len(word) > 1]
        return keywords
    
    def _has_specific_terms(self, query: str) -> bool:
        """특정 용어 포함 여부 검사"""
        for pattern in self.specific_terms:
            if re.search(pattern, query, re.IGNORECASE):
                return True
        return False
    
    def _has_compound_concepts(self, query: str) -> bool:
        """복합 개념 포함 여부 검사"""
        for pattern in self.compound_patterns:
            if re.search(pattern, query):
                return True
        return False
    
    def _determine_complexity(
        self, 
        query_length: int, 
        keyword_count: int, 
        has_specific_terms: bool, 
        has_compound_concepts: bool
    ) -> QueryComplexity:
        """복잡도 결정"""
        # 점수 기반 복잡도 계산
        complexity_score = 0
        
        # 길이 기반 점수
        if query_length > 100:
            complexity_score += 3
        elif query_length > 50:
            complexity_score += 2
        elif query_length > 20:
            complexity_score += 1
        
        # 키워드 수 기반 점수
        if keyword_count > 7:
            complexity_score += 3
        elif keyword_count > 4:
            complexity_score += 2
        elif keyword_count > 2:
            complexity_score += 1
        
        # 특정 용어 가산점
        if has_specific_terms:
            complexity_score += 1
        
        # 복합 개념 가산점
        if has_compound_concepts:
            complexity_score += 2
        
        # 복잡도 결정
        if complexity_score <= 2:
            return QueryComplexity.SIMPLE
        elif complexity_score <= 5:
            return QueryComplexity.MEDIUM
        else:
            return QueryComplexity.COMPLEX
    
    def _estimate_processing_time(self, complexity: QueryComplexity, keyword_count: int) -> float:
        """처리 시간 추정"""
        base_times = {
            QueryComplexity.SIMPLE: 0.5,
            QueryComplexity.MEDIUM: 1.5,
            QueryComplexity.COMPLEX: 3.0
        }
        
        base_time = base_times[complexity]
        
        # 키워드 수에 따른 추가 시간
        additional_time = keyword_count * 0.1
        
        # 과거 성능 데이터 반영
        if self.performance_history:
            avg_performance = sum(self.performance_history) / len(self.performance_history)
            performance_factor = avg_performance / 2.0  # 기준 시간 2초
            base_time *= performance_factor
        
        return base_time + additional_time
    
    def _recommend_processing_mode(
        self, 
        complexity: QueryComplexity, 
        estimated_time: float, 
        context: Optional[Dict[str, Any]]
    ) -> ProcessingMode:
        """처리 모드 추천"""
        # 컨텍스트에서 우선순위 확인
        if context:
            priority = context.get('priority', 'balanced')
            if priority == 'speed':
                return ProcessingMode.FAST
            elif priority == 'quality':
                return ProcessingMode.QUALITY
        
        # 복잡도와 시간 기반 추천
        if complexity == QueryComplexity.SIMPLE:
            return ProcessingMode.FAST
        elif complexity == QueryComplexity.COMPLEX or estimated_time > 4.0:
            return ProcessingMode.QUALITY
        else:
            return ProcessingMode.BALANCED
    
    def _decide_features(
        self,
        complexity: QueryComplexity,
        mode: ProcessingMode,
        has_specific_terms: bool,
        has_compound_concepts: bool
    ) -> Dict[str, bool]:
        """기능 활성화 결정"""
        features = {
            "query_rewriting": False,
            "parallel_search": False,
            "reranking": False,
            "hybrid_search": True,  # 기본적으로 활성화
            "cache_enabled": True   # 기본적으로 활성화
        }
        
        # 복잡도별 기본 설정
        if complexity == QueryComplexity.SIMPLE:
            features.update({
                "query_rewriting": False,
                "parallel_search": False,
                "reranking": False
            })
        elif complexity == QueryComplexity.MEDIUM:
            features.update({
                "query_rewriting": has_compound_concepts,
                "parallel_search": has_compound_concepts,
                "reranking": has_specific_terms
            })
        else:  # COMPLEX
            features.update({
                "query_rewriting": True,
                "parallel_search": True,
                "reranking": True
            })
        
        # 모드별 조정
        if mode == ProcessingMode.FAST:
            features.update({
                "query_rewriting": False,
                "reranking": False
            })
        elif mode == ProcessingMode.QUALITY:
            features.update({
                "query_rewriting": True,
                "parallel_search": True,
                "reranking": True
            })
        
        return features
    
    def record_performance(self, actual_time: float):
        """성능 기록"""
        self.performance_history.append(actual_time)
        if len(self.performance_history) > self.max_history:
            self.performance_history.pop(0)
    
    def get_performance_stats(self) -> Dict[str, float]:
        """성능 통계 반환"""
        if not self.performance_history:
            return {"avg": 0.0, "min": 0.0, "max": 0.0, "count": 0}
        
        return {
            "avg": sum(self.performance_history) / len(self.performance_history),
            "min": min(self.performance_history),
            "max": max(self.performance_history),
            "count": len(self.performance_history)
        }


class ResponseTimeTracker:
    """응답 시간 추적기"""
    
    def __init__(self):
        self.current_requests = {}
    
    def start_tracking(self, request_id: str) -> str:
        """요청 추적 시작"""
        self.current_requests[request_id] = time.time()
        return request_id
    
    def end_tracking(self, request_id: str) -> float:
        """요청 추적 종료 및 소요 시간 반환"""
        if request_id in self.current_requests:
            elapsed = time.time() - self.current_requests[request_id]
            del self.current_requests[request_id]
            return elapsed
        return 0.0


# 전역 인스턴스
adaptive_feature_manager = AdaptiveFeatureManager()
response_time_tracker = ResponseTimeTracker()


# 편의 함수들
def analyze_query_complexity(query: str, context: Optional[Dict[str, Any]] = None) -> QueryAnalysis:
    """쿼리 복잡도 분석"""
    return adaptive_feature_manager.analyze_query(query, context)


def should_enable_feature(query: str, feature_name: str, context: Optional[Dict[str, Any]] = None) -> bool:
    """특정 기능 활성화 여부 판단"""
    analysis = analyze_query_complexity(query, context)
    return analysis.features_to_enable.get(feature_name, False)


def get_optimal_search_config(query: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """최적 검색 설정 반환"""
    analysis = analyze_query_complexity(query, context)
    
    return {
        "use_query_rewriting": analysis.features_to_enable.get("query_rewriting", False),
        "use_parallel_search": analysis.features_to_enable.get("parallel_search", False),
        "use_reranking": analysis.features_to_enable.get("reranking", False),
        "use_hybrid_search": analysis.features_to_enable.get("hybrid_search", True),
        "processing_mode": analysis.recommended_mode.value,
        "estimated_time": analysis.estimated_processing_time,
        "complexity": analysis.complexity.value
    }