"""
애플리케이션 시작 시 초기화 스크립트
모델 예열, 캐시 초기화, 성능 최적화 등을 수행
"""

import time
from typing import Dict, Any

from app.core.logging import logger
from app.core.model_optimization import (
    warm_up_all_models,
    optimize_model_performance,
    model_manager,
    initialize_model_configurations,
)
from app.core.cache import initialize_cache
from app.core.adaptive_features import adaptive_feature_manager


class StartupManager:
    """애플리케이션 시작 관리자"""

    def __init__(self):
        self.startup_tasks = []
        self.startup_stats = {}

    async def initialize_application(self) -> Dict[str, Any]:
        """애플리케이션 전체 초기화"""
        start_time = time.time()
        logger.info("🚀 Application initialization started")

        initialization_results = {
            "success": True,
            "tasks_completed": [],
            "tasks_failed": [],
            "total_time": 0.0,
            "details": {},
        }

        try:
            # 1. 기본 시스템 초기화
            await self._initialize_core_systems(initialization_results)

            # 2. 캐시 시스템 초기화
            await self._initialize_cache_system(initialization_results)

            # 3. 모델 시스템 초기화
            await self._initialize_model_system(initialization_results)

            # 4. 성능 최적화 적용
            await self._apply_performance_optimizations(initialization_results)

            # 5. 상태 검증
            await self._validate_system_health(initialization_results)

            total_time = time.time() - start_time
            initialization_results["total_time"] = total_time

            logger.info(f"✅ Application initialization completed in {total_time:.3f}s")
            logger.info(
                f"   Tasks completed: {len(initialization_results['tasks_completed'])}"
            )
            logger.info(
                f"   Tasks failed: {len(initialization_results['tasks_failed'])}"
            )

            return initialization_results

        except Exception as e:
            initialization_results["success"] = False
            initialization_results["error"] = str(e)
            logger.error(f"❌ Application initialization failed: {e}")
            return initialization_results

    async def _initialize_core_systems(self, results: Dict[str, Any]):
        """핵심 시스템 초기화"""
        try:
            logger.info("🔧 Initializing core systems...")

            # 로깅 시스템 확인
            logger.info("Logging system: ✅ Active")

            # 설정 시스템 확인
            from app.core.config import settings

            logger.info(f"Configuration loaded: ✅ {settings.app_name}")

            results["tasks_completed"].append("core_systems")
            results["details"]["core_systems"] = "✅ Success"

        except Exception as e:
            results["tasks_failed"].append("core_systems")
            results["details"]["core_systems"] = f"❌ {str(e)}"
            logger.error(f"Core systems initialization failed: {e}")

    async def _initialize_cache_system(self, results: Dict[str, Any]):
        """캐시 시스템 초기화"""
        try:
            logger.info("💾 Initializing cache system...")
            start_time = time.time()

            # 캐시 초기화
            cache_result = await initialize_cache()
            elapsed_time = time.time() - start_time

            if cache_result.get("success", False):
                logger.info(f"Cache system initialized: ✅ ({elapsed_time:.3f}s)")
                logger.info(
                    f"   Cache type: {cache_result.get('cache_type', 'unknown')}"
                )

                results["tasks_completed"].append("cache_system")
                results["details"][
                    "cache_system"
                ] = f"✅ {cache_result.get('cache_type')} ({elapsed_time:.3f}s)"
            else:
                raise Exception(cache_result.get("error", "Unknown cache error"))

        except Exception as e:
            results["tasks_failed"].append("cache_system")
            results["details"]["cache_system"] = f"❌ {str(e)}"
            logger.error(f"Cache system initialization failed: {e}")

    async def _initialize_model_system(self, results: Dict[str, Any]):
        """모델 시스템 초기화"""
        try:
            logger.info("🧠 Initializing model system...")
            start_time = time.time()

            # 모델 설정 초기화
            initialize_model_configurations()

            # 모델 예열
            await warm_up_all_models()

            elapsed_time = time.time() - start_time

            # 모델 통계 수집
            model_stats = model_manager.get_model_stats()
            model_count = len(
                [
                    m
                    for m in model_stats.keys()
                    if model_manager.warm_up_status.get(m, False)
                ]
            )

            logger.info(f"Model system initialized: ✅ ({elapsed_time:.3f}s)")
            logger.info(f"   Models warmed up: {model_count}")

            results["tasks_completed"].append("model_system")
            results["details"][
                "model_system"
            ] = f"✅ {model_count} models warmed up ({elapsed_time:.3f}s)"

        except Exception as e:
            results["tasks_failed"].append("model_system")
            results["details"]["model_system"] = f"❌ {str(e)}"
            logger.error(f"Model system initialization failed: {e}")

    async def _apply_performance_optimizations(self, results: Dict[str, Any]):
        """성능 최적화 적용"""
        try:
            logger.info("⚡ Applying performance optimizations...")
            start_time = time.time()

            # 모델 성능 최적화
            await optimize_model_performance()

            # 메모리 최적화
            model_manager.optimize_memory()

            elapsed_time = time.time() - start_time

            logger.info(f"Performance optimizations applied: ✅ ({elapsed_time:.3f}s)")

            results["tasks_completed"].append("performance_optimization")
            results["details"][
                "performance_optimization"
            ] = f"✅ Completed ({elapsed_time:.3f}s)"

        except Exception as e:
            results["tasks_failed"].append("performance_optimization")
            results["details"]["performance_optimization"] = f"❌ {str(e)}"
            logger.error(f"Performance optimization failed: {e}")

    async def _validate_system_health(self, results: Dict[str, Any]):
        """시스템 상태 검증"""
        try:
            logger.info("🏥 Validating system health...")
            start_time = time.time()

            health_checks = {
                "models_ready": False,
                "cache_accessible": False,
                "adaptive_features_active": False,
            }

            # 모델 상태 확인
            model_stats = model_manager.get_model_stats()
            if model_stats:
                health_checks["models_ready"] = True

            # 캐시 상태 확인
            try:
                from app.core.cache import cache_manager

                if cache_manager.cache is not None:
                    health_checks["cache_accessible"] = True
            except Exception:
                pass

            # 적응형 기능 상태 확인
            if adaptive_feature_manager:
                health_checks["adaptive_features_active"] = True

            elapsed_time = time.time() - start_time
            healthy_systems = sum(health_checks.values())
            total_systems = len(health_checks)

            logger.info(f"System health validation: ✅ ({elapsed_time:.3f}s)")
            logger.info(f"   Healthy systems: {healthy_systems}/{total_systems}")

            for system, status in health_checks.items():
                status_icon = "✅" if status else "⚠️"
                logger.info(f"   {system}: {status_icon}")

            results["tasks_completed"].append("health_validation")
            results["details"][
                "health_validation"
            ] = f"✅ {healthy_systems}/{total_systems} systems healthy"
            results["details"]["health_checks"] = health_checks

        except Exception as e:
            results["tasks_failed"].append("health_validation")
            results["details"]["health_validation"] = f"❌ {str(e)}"
            logger.error(f"System health validation failed: {e}")

    def get_startup_summary(self, results: Dict[str, Any]) -> str:
        """시작 요약 정보 반환"""
        summary_lines = [
            "🚀 MOJI AI System Startup Summary",
            "=" * 50,
            f"Total Time: {results.get('total_time', 0):.3f}s",
            f"Success: {'✅' if results.get('success', False) else '❌'}",
            "",
        ]

        if results.get("tasks_completed"):
            summary_lines.append("✅ Completed Tasks:")
            for task in results["tasks_completed"]:
                detail = results["details"].get(task, "")
                summary_lines.append(f"   • {task}: {detail}")
            summary_lines.append("")

        if results.get("tasks_failed"):
            summary_lines.append("❌ Failed Tasks:")
            for task in results["tasks_failed"]:
                detail = results["details"].get(task, "")
                summary_lines.append(f"   • {task}: {detail}")
            summary_lines.append("")

        # 시스템 상태
        health_checks = results["details"].get("health_checks", {})
        if health_checks:
            summary_lines.append("🏥 System Health:")
            for system, status in health_checks.items():
                status_icon = "✅" if status else "⚠️"
                summary_lines.append(f"   • {system}: {status_icon}")

        return "\n".join(summary_lines)


# 전역 인스턴스
startup_manager = StartupManager()


# 편의 함수들
async def initialize_moji_system() -> Dict[str, Any]:
    """MOJI 시스템 초기화"""
    return await startup_manager.initialize_application()


async def quick_health_check() -> Dict[str, bool]:
    """빠른 상태 점검"""
    try:
        health_status = {
            "cache_system": False,
            "model_system": False,
            "adaptive_features": False,
        }

        # 캐시 시스템 확인
        try:
            from app.core.cache import cache_manager

            health_status["cache_system"] = cache_manager.cache is not None
        except Exception:
            pass

        # 모델 시스템 확인
        try:
            model_stats = model_manager.get_model_stats()
            health_status["model_system"] = len(model_stats) > 0
        except Exception:
            pass

        # 적응형 기능 확인
        try:
            health_status["adaptive_features"] = adaptive_feature_manager is not None
        except Exception:
            pass

        return health_status

    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return {"error": str(e)}


# 자동 정리 함수
async def cleanup_on_shutdown():
    """종료 시 정리 작업"""
    try:
        logger.info("🧹 Starting cleanup on shutdown...")

        # 모델 정리
        model_manager.optimize_memory()

        # 캐시 정리 (필요시)
        # await cache_manager.cleanup()

        logger.info("✅ Cleanup completed")

    except Exception as e:
        logger.error(f"Cleanup failed: {e}")
