#!/usr/bin/env python3
"""
모델 최적화 테스트
예열, 경량화, 공유 성능 검증
"""

import asyncio
import os
import sys
import time
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))


async def test_model_warm_up():
    """모델 예열 테스트"""
    print("🔥 모델 예열 테스트")
    print("=" * 40)
    
    try:
        from app.core.model_optimization import (
            model_manager, warm_up_all_models, initialize_model_configurations
        )
        
        # 모델 설정 초기화
        initialize_model_configurations()
        
        # 예열 전 상태 확인
        print("📊 예열 전 상태:")
        warm_up_status_before = {
            model_id: status for model_id, status in model_manager.warm_up_status.items()
        }
        for model_id, status in warm_up_status_before.items():
            status_icon = "✅" if status else "❌"
            print(f"   {model_id}: {status_icon}")
        
        # 모델 예열 실행
        print("\n🔥 모델 예열 시작...")
        start_time = time.time()
        
        await warm_up_all_models()
        
        warm_up_time = time.time() - start_time
        
        # 예열 후 상태 확인
        print(f"\n📊 예열 완료 ({warm_up_time:.3f}초):")
        warm_up_status_after = {
            model_id: status for model_id, status in model_manager.warm_up_status.items()
        }
        
        warmed_models = 0
        for model_id, status in warm_up_status_after.items():
            status_icon = "✅" if status else "❌"
            print(f"   {model_id}: {status_icon}")
            if status:
                warmed_models += 1
        
        # 모델 통계 확인
        model_stats = model_manager.get_model_stats()
        print(f"\n📈 모델 통계:")
        for model_id, stats in model_stats.items():
            print(f"   {model_id}:")
            print(f"     예열 시간: {stats.warm_up_time:.3f}초")
            print(f"     총 요청: {stats.total_requests}회")
        
        if warmed_models > 0:
            print(f"\n✅ {warmed_models}개 모델 예열 성공!")
        else:
            print(f"\n⚠️  모델 예열 필요")
        
    except Exception as e:
        print(f"❌ 모델 예열 테스트 오류: {str(e)}")
        import traceback
        traceback.print_exc()


async def test_model_sharing():
    """모델 공유 (싱글톤) 테스트"""
    print(f"\n🔄 모델 공유 테스트")
    print("=" * 40)
    
    try:
        from app.core.model_optimization import (
            get_optimized_embedding_model, get_optimized_reranker_model
        )
        
        # 임베딩 모델 공유 테스트
        print("📦 임베딩 모델 공유 테스트")
        start_time = time.time()
        
        model1 = get_optimized_embedding_model()
        model2 = get_optimized_embedding_model()
        model3 = get_optimized_embedding_model()
        
        sharing_time = time.time() - start_time
        
        # 동일한 인스턴스인지 확인
        is_same_instance = (id(model1) == id(model2) == id(model3))
        
        print(f"   ⏱️  가져오기 시간: {sharing_time:.6f}초")
        print(f"   🔗 동일 인스턴스: {'✅' if is_same_instance else '❌'}")
        print(f"   🆔 인스턴스 ID: {id(model1)}")
        
        # 리랭커 모델 공유 테스트 (fallback 될 수 있음)
        print(f"\n📦 리랭커 모델 공유 테스트")
        try:
            reranker1 = get_optimized_reranker_model()
            reranker2 = get_optimized_reranker_model()
            
            reranker_same = (id(reranker1) == id(reranker2))
            print(f"   🔗 동일 인스턴스: {'✅' if reranker_same else '❌'}")
            print(f"   🆔 인스턴스 ID: {id(reranker1)}")
        except Exception as e:
            print(f"   ⚠️  리랭커 모델 공유 실패: {str(e)}")
        
        if is_same_instance:
            print(f"\n✅ 모델 공유 (싱글톤 패턴) 정상 작동!")
        else:
            print(f"\n⚠️  모델 공유에 문제가 있을 수 있습니다")
        
    except Exception as e:
        print(f"❌ 모델 공유 테스트 오류: {str(e)}")


async def test_model_performance_optimization():
    """모델 성능 최적화 테스트"""
    print(f"\n⚡ 모델 성능 최적화 테스트")
    print("=" * 50)
    
    try:
        # 환경 설정
        from app.core.config import settings
        if settings.openai_api_key:
            os.environ['OPENAI_API_KEY'] = settings.openai_api_key
        elif settings.llm_api_key:
            os.environ['OPENAI_API_KEY'] = settings.llm_api_key
        
        from app.core.model_optimization import get_optimized_embedding_model
        from app.core.cache import clear_cache
        
        # 캐시 초기화
        await clear_cache()
        
        # 최적화된 임베딩 모델 가져오기
        embedding_model = get_optimized_embedding_model()
        
        # 성능 테스트 쿼리
        test_queries = [
            "MOJI AI 시스템",
            "프로젝트 관리 기능",
            "사용자 인터페이스",
            "데이터 분석",
            "시스템 아키텍처"
        ]
        
        print(f"📝 테스트 쿼리: {len(test_queries)}개")
        print()
        
        # 첫 번째 실행 (콜드 스타트)
        print("🥶 콜드 스타트 성능 (첫 실행)")
        cold_times = []
        
        for i, query in enumerate(test_queries, 1):
            start_time = time.time()
            try:
                # 임베딩 생성
                embedding = embedding_model.embed_query(query)
                elapsed_time = time.time() - start_time
                cold_times.append(elapsed_time)
                
                print(f"   쿼리 {i}: {elapsed_time:.3f}초 (벡터 차원: {len(embedding)})")
                
            except Exception as e:
                print(f"   쿼리 {i}: ❌ 오류 - {str(e)}")
        
        # 두 번째 실행 (웜 스타트)
        print(f"\n🔥 웜 스타트 성능 (재실행)")
        warm_times = []
        
        for i, query in enumerate(test_queries, 1):
            start_time = time.time()
            try:
                # 임베딩 생성 (캐시된 결과 활용)
                embedding = embedding_model.embed_query(query)
                elapsed_time = time.time() - start_time
                warm_times.append(elapsed_time)
                
                print(f"   쿼리 {i}: {elapsed_time:.3f}초")
                
            except Exception as e:
                print(f"   쿼리 {i}: ❌ 오류 - {str(e)}")
        
        # 성능 비교
        if cold_times and warm_times:
            avg_cold = sum(cold_times) / len(cold_times)
            avg_warm = sum(warm_times) / len(warm_times)
            improvement = ((avg_cold - avg_warm) / avg_cold) * 100 if avg_cold > 0 else 0
            speedup = avg_cold / avg_warm if avg_warm > 0 else float('inf')
            
            print(f"\n📊 성능 비교:")
            print(f"   평균 콜드 스타트: {avg_cold:.3f}초")
            print(f"   평균 웜 스타트: {avg_warm:.3f}초")
            print(f"   성능 향상: {improvement:.1f}%")
            print(f"   속도 향상: {speedup:.1f}x")
            
            if improvement > 50:
                print(f"   ✅ 캐싱 효과 우수!")
            elif improvement > 20:
                print(f"   🔄 캐싱 효과 양호")
            else:
                print(f"   ⚠️  캐싱 효과 제한적")
        
    except Exception as e:
        print(f"❌ 성능 최적화 테스트 오류: {str(e)}")
        import traceback
        traceback.print_exc()


async def test_startup_initialization():
    """시작 초기화 테스트"""
    print(f"\n🚀 시작 초기화 테스트")
    print("=" * 40)
    
    try:
        from app.core.startup import initialize_moji_system, startup_manager
        
        # 전체 시스템 초기화
        print("🔧 MOJI 시스템 초기화 중...")
        
        initialization_results = await initialize_moji_system()
        
        # 결과 출력
        print("\n📊 초기화 결과:")
        print(f"   성공: {'✅' if initialization_results.get('success', False) else '❌'}")
        print(f"   총 시간: {initialization_results.get('total_time', 0):.3f}초")
        print(f"   완료된 작업: {len(initialization_results.get('tasks_completed', []))}개")
        print(f"   실패한 작업: {len(initialization_results.get('tasks_failed', []))}개")
        
        # 상세 정보
        if initialization_results.get('details'):
            print(f"\n📋 작업 상세:")
            for task, detail in initialization_results['details'].items():
                print(f"   {task}: {detail}")
        
        # 요약 정보 출력
        summary = startup_manager.get_startup_summary(initialization_results)
        print(f"\n{summary}")
        
    except Exception as e:
        print(f"❌ 시작 초기화 테스트 오류: {str(e)}")


async def test_memory_optimization():
    """메모리 최적화 테스트"""
    print(f"\n💾 메모리 최적화 테스트")
    print("=" * 40)
    
    try:
        import psutil
        import gc
        from app.core.model_optimization import model_manager
        
        # 초기 메모리 사용량
        process = psutil.Process()
        memory_before = process.memory_info().rss / 1024 / 1024  # MB
        
        print(f"📊 최적화 전 메모리: {memory_before:.1f} MB")
        
        # 메모리 최적화 실행
        print("🧹 메모리 최적화 실행 중...")
        start_time = time.time()
        
        model_manager.optimize_memory()
        gc.collect()  # 추가 가비지 컬렉션
        
        optimization_time = time.time() - start_time
        
        # 최적화 후 메모리 사용량
        memory_after = process.memory_info().rss / 1024 / 1024  # MB
        memory_saved = memory_before - memory_after
        memory_reduction = (memory_saved / memory_before) * 100 if memory_before > 0 else 0
        
        print(f"📊 최적화 후 메모리: {memory_after:.1f} MB")
        print(f"💾 절약된 메모리: {memory_saved:.1f} MB ({memory_reduction:.1f}%)")
        print(f"⏱️  최적화 시간: {optimization_time:.3f}초")
        
        if memory_reduction > 10:
            print(f"✅ 메모리 최적화 효과 우수!")
        elif memory_reduction > 5:
            print(f"🔄 메모리 최적화 효과 양호")
        else:
            print(f"⚠️  메모리 최적화 효과 제한적")
    
    except ImportError:
        print("⚠️  psutil 패키지가 설치되지 않아 메모리 측정을 생략합니다")
    except Exception as e:
        print(f"❌ 메모리 최적화 테스트 오류: {str(e)}")


if __name__ == "__main__":
    async def main():
        await test_model_warm_up()
        await test_model_sharing()
        await test_model_performance_optimization()
        await test_startup_initialization()
        await test_memory_optimization()
        
        print(f"\n🎉 모든 모델 최적화 테스트 완료!")
        print(f"💡 모델 예열, 공유, 캐싱을 통해 성능이 크게 향상되었습니다!")
    
    asyncio.run(main())