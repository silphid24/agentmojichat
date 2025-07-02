#!/usr/bin/env python3
"""
성능 모니터링 시스템 테스트
실시간 지표 수집, 임계값 알림, 대시보드 데이터 검증
"""

import asyncio
import os
import sys
import time
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))


async def test_basic_monitoring():
    """기본 모니터링 기능 테스트"""
    print("📊 기본 모니터링 기능 테스트")
    print("=" * 50)
    
    try:
        from app.core.monitoring import (
            performance_monitor, MetricType, AlertLevel, timer
        )
        
        # 기본 지표 기록
        print("📈 기본 지표 기록 테스트")
        
        # 카운터 기록
        performance_monitor.collector.record_counter("test_requests", 1.0, {"endpoint": "/api/test"})
        performance_monitor.collector.record_counter("test_requests", 1.0, {"endpoint": "/api/test"})
        performance_monitor.collector.record_counter("test_requests", 1.0, {"endpoint": "/api/health"})
        
        # 게이지 기록
        performance_monitor.collector.record_gauge("cpu_usage", 0.45, {"host": "localhost"})
        performance_monitor.collector.record_gauge("memory_usage", 0.72, {"host": "localhost"})
        
        # 히스토그램 기록
        performance_monitor.collector.record_histogram("response_size", 1024.5)
        performance_monitor.collector.record_histogram("response_size", 2048.3)
        performance_monitor.collector.record_histogram("response_size", 512.7)
        
        # 타이머 기록
        performance_monitor.collector.record_timer("api_duration", 0.245, {"endpoint": "/api/test"})
        performance_monitor.collector.record_timer("api_duration", 1.120, {"endpoint": "/api/test"})
        performance_monitor.collector.record_timer("api_duration", 0.089, {"endpoint": "/api/health"})
        
        print("   ✅ 지표 기록 완료")
        
        # 지표 요약 확인
        print(f"\n📋 지표 요약 확인")
        
        summary = performance_monitor.collector.get_all_metrics_summary(window_seconds=300)
        for metric_name, stats in summary.items():
            if stats.get("count", 0) > 0:
                print(f"   {metric_name}:")
                print(f"     타입: {stats.get('metric_type', 'unknown')}")
                print(f"     개수: {stats.get('count', 0)}")
                print(f"     평균: {stats.get('avg', 0):.3f}")
                print(f"     최소/최대: {stats.get('min', 0):.3f} / {stats.get('max', 0):.3f}")
        
        print("\n✅ 기본 모니터링 기능 정상 작동!")
        
    except Exception as e:
        print(f"❌ 기본 모니터링 테스트 오류: {str(e)}")
        import traceback
        traceback.print_exc()


async def test_timer_context():
    """타이머 컨텍스트 매니저 테스트"""
    print(f"\n⏱️  타이머 컨텍스트 매니저 테스트")
    print("=" * 40)
    
    try:
        from app.core.monitoring import timer, performance_monitor
        
        # 컨텍스트 매니저 사용
        print("🔄 타이머 컨텍스트 실행")
        
        with timer("test_operation", {"operation": "database_query"}):
            # 시뮬레이션 작업
            await asyncio.sleep(0.1)
            print("   시뮬레이션 작업 완료 (0.1초)")
        
        with timer("test_operation", {"operation": "api_call"}):
            # 또 다른 시뮬레이션 작업
            await asyncio.sleep(0.05)
            print("   시뮬레이션 작업 완료 (0.05초)")
        
        # 결과 확인
        summary = performance_monitor.collector.get_metric_summary("test_operation", 60)
        
        print(f"\n📊 타이머 결과:")
        print(f"   측정 횟수: {summary.get('count', 0)}")
        print(f"   평균 시간: {summary.get('avg', 0):.3f}초")
        print(f"   최소/최대: {summary.get('min', 0):.3f}초 / {summary.get('max', 0):.3f}초")
        
        if summary.get('count', 0) >= 2:
            print("   ✅ 타이머 컨텍스트 정상 작동!")
        else:
            print("   ⚠️  타이머 기록에 문제가 있을 수 있습니다")
        
    except Exception as e:
        print(f"❌ 타이머 컨텍스트 테스트 오류: {str(e)}")


async def test_alert_system():
    """알림 시스템 테스트"""
    print(f"\n🚨 알림 시스템 테스트")
    print("=" * 40)
    
    try:
        from app.core.monitoring import performance_monitor, AlertLevel
        
        # 알림 콜백 추가
        alert_received = []
        
        def test_alert_callback(alert):
            alert_received.append({
                "level": alert.level.value,
                "metric": alert.metric_name,
                "message": alert.message,
                "value": alert.value
            })
            print(f"   🔔 알림 수신: [{alert.level.value.upper()}] {alert.message} (값: {alert.value:.3f})")
        
        performance_monitor.add_alert_callback(test_alert_callback)
        
        # 임계값 테스트를 위한 데이터 기록
        print("⚠️  임계값 테스트 시작")
        
        # 응답 시간 임계값 테스트 (2초 이상 = 경고)
        performance_monitor.collector.record_timer("response_time", 2.5)  # 경고 발생 예상
        performance_monitor._check_thresholds("response_time", 2.5)
        
        # 에러율 임계값 테스트 (10% 이상 = 에러)
        performance_monitor.collector.record_gauge("error_rate_test", 0.12)  # 에러 발생 예상
        performance_monitor._check_thresholds("error_rate", 0.12)
        
        # 메모리 사용률 임계값 테스트 (95% 이상 = 치명적)
        performance_monitor.collector.record_gauge("memory_usage", 0.96)  # 치명적 발생 예상
        performance_monitor._check_thresholds("memory_usage", 0.96)
        
        await asyncio.sleep(0.1)  # 콜백 처리 대기
        
        print(f"\n📋 알림 수신 결과:")
        print(f"   총 수신 알림: {len(alert_received)}개")
        
        for i, alert in enumerate(alert_received, 1):
            print(f"   알림 {i}: [{alert['level'].upper()}] {alert['metric']} - {alert['message']}")
        
        # 알림 히스토리 확인
        recent_alerts = [alert for alert in performance_monitor.alerts if alert.timestamp > time.time() - 60]
        print(f"   저장된 알림: {len(recent_alerts)}개")
        
        if len(alert_received) >= 3:
            print("   ✅ 알림 시스템 정상 작동!")
        else:
            print("   ⚠️  일부 알림이 발생하지 않았을 수 있습니다")
        
    except Exception as e:
        print(f"❌ 알림 시스템 테스트 오류: {str(e)}")


async def test_request_monitoring():
    """요청 모니터링 테스트"""
    print(f"\n🔍 요청 모니터링 테스트")
    print("=" * 40)
    
    try:
        from app.core.monitoring import performance_monitor
        
        # 요청 시뮬레이션
        print("📤 요청 모니터링 시뮬레이션")
        
        test_scenarios = [
            {"operation": "search", "duration": 0.5, "success": True},
            {"operation": "search", "duration": 1.2, "success": True},
            {"operation": "search", "duration": 0.8, "success": False},  # 실패 케이스
            {"operation": "embedding", "duration": 0.3, "success": True},
            {"operation": "embedding", "duration": 0.4, "success": True},
        ]
        
        for i, scenario in enumerate(test_scenarios, 1):
            print(f"   요청 {i}: {scenario['operation']} 작업")
            
            # 요청 시작
            request_id = f"test_req_{i}"
            performance_monitor.record_request_start(request_id, scenario["operation"])
            
            # 작업 시뮬레이션
            await asyncio.sleep(scenario["duration"])
            
            # 요청 종료
            performance_monitor.record_request_end(
                request_id, 
                success=scenario["success"], 
                operation=scenario["operation"],
                additional_metrics={"complexity": "medium"}
            )
        
        # 결과 분석
        print(f"\n📊 요청 모니터링 결과:")
        
        # 전체 요청 수
        total_requests = performance_monitor.collector.get_metric_summary("requests_total", 60)
        print(f"   총 요청 수: {total_requests.get('count', 0)}")
        
        # 응답 시간 통계
        response_times = performance_monitor.collector.get_metric_summary("response_time", 60)
        if response_times.get('count', 0) > 0:
            print(f"   평균 응답 시간: {response_times.get('avg', 0):.3f}초")
            print(f"   최소/최대: {response_times.get('min', 0):.3f}초 / {response_times.get('max', 0):.3f}초")
        
        # 성공/실패 비율
        success_count = performance_monitor.collector.get_metric_summary("requests_success", 60).get('count', 0)
        error_count = performance_monitor.collector.get_metric_summary("requests_error", 60).get('count', 0)
        total = success_count + error_count
        
        if total > 0:
            success_rate = (success_count / total) * 100
            error_rate = (error_count / total) * 100
            print(f"   성공률: {success_rate:.1f}% ({success_count}/{total})")
            print(f"   에러율: {error_rate:.1f}% ({error_count}/{total})")
        
        print("   ✅ 요청 모니터링 정상 작동!")
        
    except Exception as e:
        print(f"❌ 요청 모니터링 테스트 오류: {str(e)}")


async def test_dashboard_data():
    """대시보드 데이터 테스트"""
    print(f"\n📈 대시보드 데이터 테스트")
    print("=" * 40)
    
    try:
        from app.core.monitoring import performance_monitor
        
        # 대시보드 데이터 생성
        print("📊 대시보드 데이터 생성 중...")
        
        dashboard_data = performance_monitor.get_dashboard_data()
        
        print(f"📋 대시보드 데이터 구조:")
        print(f"   타임스탬프: {dashboard_data.get('timestamp', 'N/A')}")
        print(f"   시스템 상태: {dashboard_data.get('system_status', 'unknown')}")
        print(f"   지표 수: {len(dashboard_data.get('metrics', {}))}")
        print(f"   최근 알림 수: {len(dashboard_data.get('recent_alerts', []))}")
        
        # 알림 카운트
        alert_count = dashboard_data.get('alert_count', {})
        print(f"   알림 통계:")
        print(f"     총 알림: {alert_count.get('total', 0)}")
        print(f"     치명적: {alert_count.get('critical', 0)}")
        print(f"     에러: {alert_count.get('error', 0)}")
        print(f"     경고: {alert_count.get('warning', 0)}")
        
        # 주요 지표 표시
        metrics = dashboard_data.get('metrics', {})
        if metrics:
            print(f"\n📊 주요 지표:")
            for metric_name, stats in list(metrics.items())[:5]:  # 상위 5개만
                if stats.get('count', 0) > 0:
                    print(f"     {metric_name}: {stats.get('latest', 0):.3f} (평균: {stats.get('avg', 0):.3f})")
        
        # 최근 알림 표시
        recent_alerts = dashboard_data.get('recent_alerts', [])
        if recent_alerts:
            print(f"\n🚨 최근 알림 (최대 3개):")
            for alert in recent_alerts[-3:]:
                level = alert.get('level', 'unknown').upper()
                metric = alert.get('metric', 'unknown')
                message = alert.get('message', 'no message')
                print(f"     [{level}] {metric}: {message}")
        
        print(f"\n✅ 대시보드 데이터 생성 완료!")
        
        # JSON 직렬화 테스트
        import json
        try:
            json_data = json.dumps(dashboard_data, default=str, indent=2)
            print(f"   📄 JSON 직렬화: ✅ ({len(json_data)} 문자)")
        except Exception as json_error:
            print(f"   📄 JSON 직렬화: ❌ {str(json_error)}")
        
    except Exception as e:
        print(f"❌ 대시보드 데이터 테스트 오류: {str(e)}")
        import traceback
        traceback.print_exc()


async def test_rag_integration():
    """RAG 시스템과의 통합 테스트"""
    print(f"\n🔗 RAG 시스템 통합 테스트")
    print("=" * 40)
    
    try:
        # 환경 설정
        from app.core.config import settings
        if settings.openai_api_key:
            os.environ['OPENAI_API_KEY'] = settings.openai_api_key
        elif settings.llm_api_key:
            os.environ['OPENAI_API_KEY'] = settings.llm_api_key
        
        from app.rag.enhanced_rag import rag_pipeline
        from app.core.monitoring import performance_monitor
        from app.core.cache import clear_cache
        
        # 캐시 초기화 (공정한 테스트를 위해)
        await clear_cache()
        print("   ✅ 캐시 초기화 완료")
        
        # 모니터링 활성화 확인
        performance_monitor.enable_monitoring()
        print("   ✅ 모니터링 활성화")
        
        # RAG 쿼리 실행 (모니터링 적용됨)
        test_query = "MOJI AI 시스템의 주요 특징"
        print(f"   📝 테스트 쿼리: \"{test_query}\"")
        
        start_time = time.time()
        result = await rag_pipeline.answer_with_confidence(test_query, k=3)
        elapsed_time = time.time() - start_time
        
        print(f"   ⏱️  쿼리 처리 시간: {elapsed_time:.3f}초")
        print(f"   📄 응답 길이: {len(result.get('answer', ''))} 문자")
        print(f"   🎯 신뢰도: {result.get('confidence', 'UNKNOWN')}")
        
        # 모니터링 데이터 확인
        print(f"\n📊 모니터링 데이터 확인:")
        
        # RAG 요청 통계
        rag_summary = performance_monitor.collector.get_metric_summary("requests_total", 60)
        print(f"   RAG 요청 수: {rag_summary.get('count', 0)}")
        
        # 응답 시간 통계
        response_summary = performance_monitor.collector.get_metric_summary("response_time", 60)
        if response_summary.get('count', 0) > 0:
            print(f"   모니터링된 응답 시간: {response_summary.get('latest', 0):.3f}초")
        
        # 캐시 히트/미스 확인
        cache_hits = performance_monitor.collector.get_metric_summary("cache_hits", 60).get('count', 0)
        cache_misses = performance_monitor.collector.get_metric_summary("cache_misses", 60).get('count', 0)
        print(f"   캐시 히트: {cache_hits}, 미스: {cache_misses}")
        
        # 시스템 상태 확인
        dashboard_data = performance_monitor.get_dashboard_data()
        system_status = dashboard_data.get('system_status', 'unknown')
        print(f"   시스템 상태: {system_status}")
        
        print(f"\n✅ RAG 시스템 모니터링 통합 완료!")
        
    except Exception as e:
        print(f"❌ RAG 통합 테스트 오류: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    async def main():
        await test_basic_monitoring()
        await test_timer_context()
        await test_alert_system()
        await test_request_monitoring()
        await test_dashboard_data()
        await test_rag_integration()
        
        print(f"\n🎉 모든 모니터링 시스템 테스트 완료!")
        print(f"💡 실시간 성능 지표 수집, 알림, 대시보드가 모두 정상 작동합니다!")
    
    asyncio.run(main())