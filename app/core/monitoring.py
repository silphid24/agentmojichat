"""
성능 모니터링 시스템
실시간 성능 지표 수집, 분석, 알림
"""

import time
import asyncio
import threading
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import defaultdict, deque
from enum import Enum
import json

from app.core.logging import logger


class MetricType(Enum):
    """지표 유형"""
    COUNTER = "counter"          # 누적 카운터
    GAUGE = "gauge"             # 현재 값
    HISTOGRAM = "histogram"     # 분포
    TIMER = "timer"             # 시간 측정


class AlertLevel(Enum):
    """알림 수준"""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class MetricValue:
    """지표 값"""
    timestamp: float
    value: float
    labels: Dict[str, str] = field(default_factory=dict)


@dataclass
class Alert:
    """알림"""
    level: AlertLevel
    metric_name: str
    message: str
    value: float
    threshold: float
    timestamp: float
    resolved: bool = False


@dataclass
class PerformanceThreshold:
    """성능 임계값"""
    warning_threshold: float
    error_threshold: float
    critical_threshold: float
    check_window_seconds: int = 60  # 확인 윈도우


class MetricCollector:
    """지표 수집기"""
    
    def __init__(self, max_history: int = 1000):
        self.metrics = defaultdict(lambda: deque(maxlen=max_history))
        self.metric_types = {}
        self.lock = threading.Lock()
    
    def record_counter(self, name: str, value: float = 1.0, labels: Dict[str, str] = None):
        """카운터 기록"""
        self._record_metric(name, MetricType.COUNTER, value, labels or {})
    
    def record_gauge(self, name: str, value: float, labels: Dict[str, str] = None):
        """게이지 기록"""
        self._record_metric(name, MetricType.GAUGE, value, labels or {})
    
    def record_histogram(self, name: str, value: float, labels: Dict[str, str] = None):
        """히스토그램 기록"""
        self._record_metric(name, MetricType.HISTOGRAM, value, labels or {})
    
    def record_timer(self, name: str, duration: float, labels: Dict[str, str] = None):
        """타이머 기록"""
        self._record_metric(name, MetricType.TIMER, duration, labels or {})
    
    def _record_metric(self, name: str, metric_type: MetricType, value: float, labels: Dict[str, str]):
        """지표 기록 (내부 메서드)"""
        with self.lock:
            metric_value = MetricValue(
                timestamp=time.time(),
                value=value,
                labels=labels
            )
            
            self.metrics[name].append(metric_value)
            self.metric_types[name] = metric_type
    
    def get_metric_summary(self, name: str, window_seconds: int = 300) -> Dict[str, Any]:
        """지표 요약 통계"""
        with self.lock:
            if name not in self.metrics:
                return {"error": f"Metric {name} not found"}
            
            current_time = time.time()
            window_start = current_time - window_seconds
            
            # 윈도우 내 데이터 필터링
            recent_values = [
                metric.value for metric in self.metrics[name]
                if metric.timestamp >= window_start
            ]
            
            if not recent_values:
                return {"count": 0, "window_seconds": window_seconds}
            
            return {
                "count": len(recent_values),
                "min": min(recent_values),
                "max": max(recent_values),
                "avg": sum(recent_values) / len(recent_values),
                "latest": recent_values[-1],
                "window_seconds": window_seconds,
                "metric_type": self.metric_types[name].value
            }
    
    def get_all_metrics_summary(self, window_seconds: int = 300) -> Dict[str, Dict[str, Any]]:
        """모든 지표 요약"""
        summary = {}
        for metric_name in self.metrics.keys():
            summary[metric_name] = self.get_metric_summary(metric_name, window_seconds)
        return summary


class PerformanceMonitor:
    """성능 모니터"""
    
    def __init__(self):
        self.collector = MetricCollector()
        self.thresholds = {}
        self.alerts = deque(maxlen=1000)
        self.alert_callbacks = []
        self.monitoring_enabled = True
        
        # 기본 임계값 설정
        self._setup_default_thresholds()
    
    def _setup_default_thresholds(self):
        """기본 임계값 설정"""
        self.thresholds.update({
            "response_time": PerformanceThreshold(
                warning_threshold=2.0,    # 2초
                error_threshold=5.0,      # 5초  
                critical_threshold=10.0   # 10초
            ),
            "error_rate": PerformanceThreshold(
                warning_threshold=0.05,   # 5%
                error_threshold=0.10,     # 10%
                critical_threshold=0.20   # 20%
            ),
            "cache_hit_rate": PerformanceThreshold(
                warning_threshold=0.80,   # 80% (낮으면 경고)
                error_threshold=0.60,     # 60%
                critical_threshold=0.40   # 40%
            ),
            "memory_usage": PerformanceThreshold(
                warning_threshold=0.70,   # 70%
                error_threshold=0.85,     # 85%
                critical_threshold=0.95   # 95%
            )
        })
    
    def set_threshold(self, metric_name: str, threshold: PerformanceThreshold):
        """임계값 설정"""
        self.thresholds[metric_name] = threshold
    
    def add_alert_callback(self, callback: Callable[[Alert], None]):
        """알림 콜백 추가"""
        self.alert_callbacks.append(callback)
    
    def record_request_start(self, request_id: str, operation: str = "query") -> str:
        """요청 시작 기록"""
        if not self.monitoring_enabled:
            return request_id
        
        self.collector.record_counter("requests_total", labels={"operation": operation})
        self.collector.record_gauge(f"request_start_{request_id}", time.time())
        
        return request_id
    
    def record_request_end(
        self, 
        request_id: str, 
        success: bool = True, 
        operation: str = "query",
        additional_metrics: Dict[str, float] = None
    ):
        """요청 종료 기록"""
        if not self.monitoring_enabled:
            return
        
        try:
            # 응답 시간 계산
            start_metrics = [
                m for m in self.collector.metrics[f"request_start_{request_id}"] 
                if m.timestamp
            ]
            
            if start_metrics:
                start_time = start_metrics[-1].value
                response_time = time.time() - start_time
                
                self.collector.record_timer(
                    "response_time", 
                    response_time, 
                    labels={"operation": operation, "success": str(success)}
                )
                
                # 임계값 확인
                self._check_thresholds("response_time", response_time)
            
            # 성공/실패 카운터
            if success:
                self.collector.record_counter("requests_success", labels={"operation": operation})
            else:
                self.collector.record_counter("requests_error", labels={"operation": operation})
            
            # 추가 지표
            if additional_metrics:
                for metric_name, value in additional_metrics.items():
                    self.collector.record_gauge(metric_name, value, labels={"operation": operation})
            
            # 에러율 계산 및 확인
            self._calculate_and_check_error_rate(operation)
            
        except Exception as e:
            logger.error(f"Failed to record request end: {e}")
    
    def record_cache_hit(self, cache_type: str):
        """캐시 히트 기록"""
        self.collector.record_counter("cache_hits", labels={"type": cache_type})
    
    def record_cache_miss(self, cache_type: str):
        """캐시 미스 기록"""
        self.collector.record_counter("cache_misses", labels={"type": cache_type})
        self._calculate_and_check_cache_hit_rate(cache_type)
    
    def record_model_performance(self, model_name: str, operation: str, duration: float, success: bool = True):
        """모델 성능 기록"""
        labels = {"model": model_name, "operation": operation, "success": str(success)}
        
        self.collector.record_timer("model_duration", duration, labels=labels)
        self.collector.record_counter("model_operations", labels=labels)
    
    def record_system_metrics(self):
        """시스템 지표 기록"""
        try:
            import psutil
            
            # CPU 사용률
            cpu_percent = psutil.cpu_percent()
            self.collector.record_gauge("cpu_usage", cpu_percent / 100.0)
            
            # 메모리 사용률
            memory = psutil.virtual_memory()
            memory_percent = memory.percent / 100.0
            self.collector.record_gauge("memory_usage", memory_percent)
            
            # 메모리 임계값 확인
            self._check_thresholds("memory_usage", memory_percent)
            
        except ImportError:
            # psutil이 없으면 건너뛰기
            pass
        except Exception as e:
            logger.error(f"Failed to record system metrics: {e}")
    
    def _calculate_and_check_error_rate(self, operation: str):
        """에러율 계산 및 확인"""
        try:
            success_count = len([
                m for m in self.collector.metrics["requests_success"]
                if m.labels.get("operation") == operation and 
                   m.timestamp > time.time() - 300  # 5분 윈도우
            ])
            
            error_count = len([
                m for m in self.collector.metrics["requests_error"]
                if m.labels.get("operation") == operation and 
                   m.timestamp > time.time() - 300
            ])
            
            total_requests = success_count + error_count
            if total_requests > 0:
                error_rate = error_count / total_requests
                self.collector.record_gauge(f"error_rate_{operation}", error_rate)
                self._check_thresholds("error_rate", error_rate)
                
        except Exception as e:
            logger.error(f"Failed to calculate error rate: {e}")
    
    def _calculate_and_check_cache_hit_rate(self, cache_type: str):
        """캐시 히트율 계산 및 확인"""
        try:
            hits = len([
                m for m in self.collector.metrics["cache_hits"]
                if m.labels.get("type") == cache_type and 
                   m.timestamp > time.time() - 300  # 5분 윈도우
            ])
            
            misses = len([
                m for m in self.collector.metrics["cache_misses"]
                if m.labels.get("type") == cache_type and 
                   m.timestamp > time.time() - 300
            ])
            
            total_requests = hits + misses
            if total_requests > 0:
                hit_rate = hits / total_requests
                self.collector.record_gauge(f"cache_hit_rate_{cache_type}", hit_rate)
                
                # 캐시 히트율은 낮을수록 나쁨 (반대 로직)
                if hit_rate < self.thresholds.get("cache_hit_rate", PerformanceThreshold(0.8, 0.6, 0.4)).critical_threshold:
                    self._create_alert("cache_hit_rate", AlertLevel.CRITICAL, hit_rate, "Cache hit rate critically low")
                elif hit_rate < self.thresholds.get("cache_hit_rate", PerformanceThreshold(0.8, 0.6, 0.4)).error_threshold:
                    self._create_alert("cache_hit_rate", AlertLevel.ERROR, hit_rate, "Cache hit rate low")
                elif hit_rate < self.thresholds.get("cache_hit_rate", PerformanceThreshold(0.8, 0.6, 0.4)).warning_threshold:
                    self._create_alert("cache_hit_rate", AlertLevel.WARNING, hit_rate, "Cache hit rate below optimal")
                
        except Exception as e:
            logger.error(f"Failed to calculate cache hit rate: {e}")
    
    def _check_thresholds(self, metric_name: str, value: float):
        """임계값 확인"""
        if metric_name not in self.thresholds:
            return
        
        threshold = self.thresholds[metric_name]
        
        if value >= threshold.critical_threshold:
            self._create_alert(metric_name, AlertLevel.CRITICAL, value, f"{metric_name} critically high")
        elif value >= threshold.error_threshold:
            self._create_alert(metric_name, AlertLevel.ERROR, value, f"{metric_name} high")
        elif value >= threshold.warning_threshold:
            self._create_alert(metric_name, AlertLevel.WARNING, value, f"{metric_name} elevated")
    
    def _create_alert(self, metric_name: str, level: AlertLevel, value: float, message: str):
        """알림 생성"""
        alert = Alert(
            level=level,
            metric_name=metric_name,
            message=message,
            value=value,
            threshold=getattr(self.thresholds.get(metric_name), f"{level.value}_threshold", 0),
            timestamp=time.time()
        )
        
        self.alerts.append(alert)
        
        # 로그 기록
        log_level = {
            AlertLevel.INFO: logger.info,
            AlertLevel.WARNING: logger.warning,
            AlertLevel.ERROR: logger.error,
            AlertLevel.CRITICAL: logger.critical
        }[level]
        
        log_level(f"ALERT [{level.value.upper()}] {metric_name}: {message} (value: {value:.3f})")
        
        # 콜백 실행
        for callback in self.alert_callbacks:
            try:
                callback(alert)
            except Exception as e:
                logger.error(f"Alert callback failed: {e}")
    
    def get_dashboard_data(self) -> Dict[str, Any]:
        """대시보드 데이터 반환"""
        try:
            # 최근 5분 데이터
            metrics_summary = self.collector.get_all_metrics_summary(window_seconds=300)
            
            # 최근 알림 (최근 1시간)
            recent_alerts = [
                {
                    "level": alert.level.value,
                    "metric": alert.metric_name,
                    "message": alert.message,
                    "value": alert.value,
                    "timestamp": alert.timestamp,
                    "resolved": alert.resolved
                }
                for alert in self.alerts 
                if alert.timestamp > time.time() - 3600  # 1시간
            ]
            
            # 시스템 상태 요약
            system_status = self._get_system_status()
            
            return {
                "timestamp": time.time(),
                "system_status": system_status,
                "metrics": metrics_summary,
                "recent_alerts": recent_alerts[-20:],  # 최근 20개
                "alert_count": {
                    "total": len(recent_alerts),
                    "critical": len([a for a in recent_alerts if a["level"] == "critical"]),
                    "error": len([a for a in recent_alerts if a["level"] == "error"]),
                    "warning": len([a for a in recent_alerts if a["level"] == "warning"])
                }
            }
            
        except Exception as e:
            logger.error(f"Failed to get dashboard data: {e}")
            return {"error": str(e)}
    
    def _get_system_status(self) -> str:
        """시스템 상태 반환"""
        try:
            # 최근 5분간 심각한 알림 확인
            recent_critical = [
                alert for alert in self.alerts
                if alert.timestamp > time.time() - 300 and 
                   alert.level == AlertLevel.CRITICAL and
                   not alert.resolved
            ]
            
            if recent_critical:
                return "critical"
            
            # 최근 5분간 에러 알림 확인
            recent_errors = [
                alert for alert in self.alerts
                if alert.timestamp > time.time() - 300 and 
                   alert.level == AlertLevel.ERROR and
                   not alert.resolved
            ]
            
            if recent_errors:
                return "error"
            
            # 최근 5분간 경고 알림 확인
            recent_warnings = [
                alert for alert in self.alerts
                if alert.timestamp > time.time() - 300 and 
                   alert.level == AlertLevel.WARNING and
                   not alert.resolved
            ]
            
            if recent_warnings:
                return "warning"
            
            return "healthy"
            
        except Exception:
            return "unknown"
    
    def enable_monitoring(self):
        """모니터링 활성화"""
        self.monitoring_enabled = True
        logger.info("Performance monitoring enabled")
    
    def disable_monitoring(self):
        """모니터링 비활성화"""
        self.monitoring_enabled = False
        logger.info("Performance monitoring disabled")


class TimerContext:
    """시간 측정 컨텍스트 매니저"""
    
    def __init__(self, monitor: PerformanceMonitor, metric_name: str, labels: Dict[str, str] = None):
        self.monitor = monitor
        self.metric_name = metric_name
        self.labels = labels or {}
        self.start_time = None
    
    def __enter__(self):
        self.start_time = time.time()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.start_time:
            duration = time.time() - self.start_time
            self.monitor.collector.record_timer(self.metric_name, duration, self.labels)


# 전역 모니터 인스턴스
performance_monitor = PerformanceMonitor()


# 편의 함수들
def timer(metric_name: str, labels: Dict[str, str] = None) -> TimerContext:
    """타이머 컨텍스트 매니저"""
    return TimerContext(performance_monitor, metric_name, labels)


def monitor_function(metric_name: str = None, labels: Dict[str, str] = None):
    """함수 모니터링 데코레이터"""
    def decorator(func):
        def wrapper(*args, **kwargs):
            name = metric_name or f"{func.__name__}_duration"
            with timer(name, labels):
                try:
                    result = func(*args, **kwargs)
                    performance_monitor.collector.record_counter(
                        f"{func.__name__}_success", 
                        labels=labels
                    )
                    return result
                except Exception as e:
                    performance_monitor.collector.record_counter(
                        f"{func.__name__}_error", 
                        labels=labels
                    )
                    raise
        return wrapper
    return decorator


# 시스템 메트릭 백그라운드 수집
class SystemMetricsCollector:
    """시스템 메트릭 백그라운드 수집기"""
    
    def __init__(self, interval: int = 30):
        self.interval = interval
        self.running = False
        self.task = None
    
    async def start(self):
        """메트릭 수집 시작"""
        if self.running:
            return
        
        self.running = True
        self.task = asyncio.create_task(self._collection_loop())
        logger.info(f"System metrics collection started (interval: {self.interval}s)")
    
    async def stop(self):
        """메트릭 수집 중지"""
        self.running = False
        if self.task:
            self.task.cancel()
            try:
                await self.task
            except asyncio.CancelledError:
                pass
        logger.info("System metrics collection stopped")
    
    async def _collection_loop(self):
        """메트릭 수집 루프"""
        while self.running:
            try:
                performance_monitor.record_system_metrics()
                await asyncio.sleep(self.interval)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"System metrics collection failed: {e}")
                await asyncio.sleep(self.interval)


# 전역 시스템 메트릭 수집기
system_metrics_collector = SystemMetricsCollector()


# 초기화 함수
def setup_monitoring():
    """모니터링 설정"""
    # 기본 알림 콜백 설정
    def log_alert(alert: Alert):
        level_map = {
            AlertLevel.INFO: logger.info,
            AlertLevel.WARNING: logger.warning,
            AlertLevel.ERROR: logger.error,
            AlertLevel.CRITICAL: logger.critical
        }
        level_map[alert.level](f"Performance Alert: {alert.message}")
    
    performance_monitor.add_alert_callback(log_alert)
    logger.info("Performance monitoring setup completed")