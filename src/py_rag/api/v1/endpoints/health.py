import asyncio
import logging
import os
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Any, List, Optional

import psutil
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from py_faiss.config import settings
from py_faiss.core.embedding import get_embedding_service
from py_faiss.core.vector_store import get_vector_store
from py_faiss.services.document_service import get_document_service
from py_faiss.services.search_service import get_search_service

logger = logging.getLogger(__name__)
router = APIRouter()


class HealthStatus(BaseModel):
    """健康状态模型"""
    status: str  # healthy, degraded, unhealthy
    timestamp: str
    uptime: float
    version: str = "v1.0.0"


class ComponentHealth(BaseModel):
    """组件健康状态"""
    name: str
    status: str
    response_time: Optional[float] = None
    error: Optional[str] = None
    details: Optional[Dict[str, Any]] = None


class SystemMetrics(BaseModel):
    """系统指标"""
    cpu_percent: float
    memory_percent: float
    disk_percent: float
    load_average: List[float]
    process_count: int


class DetailedHealthResponse(BaseModel):
    """详细健康检查响应"""
    status: str
    timestamp: str
    uptime: float
    version: str
    components: List[ComponentHealth]
    system_metrics: SystemMetrics
    performance_metrics: Dict[str, Any]


async def _check_vector_store() -> ComponentHealth:
    """检查向量存储"""
    start_time = time.time()

    try:
        vector_store = await get_vector_store()

        # 获取统计信息
        stats = await vector_store.get_stats()

        response_time = time.time() - start_time

        # 检查索引状态
        index_healthy = (
                vector_store.index is not None and
                stats.get('total_chunks', 0) >= 0
        )

        status = "healthy" if index_healthy else "degraded"

        return ComponentHealth(
            name="vector_store",
            status=status,
            response_time=response_time,
            details={
                'total_documents': stats.get('total_documents', 0),
                'total_chunks': stats.get('total_chunks', 0),
                'index_size': stats.get('index_size', 0),
                'dimension': vector_store.dimension,
                'index_type': vector_store.index_type
            }
        )

    except Exception as e:
        return ComponentHealth(
            name="vector_store",
            status="unhealthy",
            response_time=time.time() - start_time,
            error=str(e)
        )


async def _check_document_service() -> ComponentHealth:
    """检查文档服务"""
    start_time = time.time()

    try:
        document_service = await get_document_service()

        # 获取统计信息
        stats = await document_service.get_statistics()

        response_time = time.time() - start_time

        # 检查服务状态
        service_healthy = 'error' not in stats

        status = "healthy" if service_healthy else "degraded"

        return ComponentHealth(
            name="document_service",
            status=status,
            response_time=response_time,
            details={
                'processing_queue': len(document_service.processing_status),
                'supported_formats': len(document_service.document_processor.get_supported_types()),
                'temp_dir': str(document_service.document_processor.temp_dir)
            }
        )

    except Exception as e:
        return ComponentHealth(
            name="document_service",
            status="unhealthy",
            response_time=time.time() - start_time,
            error=str(e)
        )


async def _check_search_service() -> ComponentHealth:
    """检查搜索服务"""
    start_time = time.time()

    try:
        search_service = await get_search_service()

        # 获取搜索统计
        stats = await search_service.get_search_statistics()

        response_time = time.time() - start_time

        status = "healthy"

        return ComponentHealth(
            name="search_service",
            status=status,
            response_time=response_time,
            details={
                'total_searches': stats.get('total_searches', 0),
                'avg_search_time': stats.get('avg_search_time', 0),
                'cache_size': stats.get('cache_size', 0),
                'history_size': stats.get('history_size', 0)
            }
        )

    except Exception as e:
        return ComponentHealth(
            name="search_service",
            status="unhealthy",
            response_time=time.time() - start_time,
            error=str(e)
        )


async def _check_storage() -> ComponentHealth:
    """检查存储"""
    start_time = time.time()

    try:
        # 检查数据目录
        data_path = Path(settings.DATA_PATH)
        index_path = Path(settings.INDEX_PATH)
        temp_path = Path(settings.TEMP_PATH)

        storage_details = {}

        # 检查目录存在性和权限
        for name, path in [("data", data_path), ("index", index_path), ("temp", temp_path)]:
            if path.exists():
                storage_details[f"{name}_exists"] = True
                storage_details[f"{name}_writable"] = os.access(path, os.W_OK)

                # 计算目录大小
                total_size = sum(f.stat().st_size for f in path.rglob('*') if f.is_file())
                storage_details[f"{name}_size_mb"] = round(total_size / 1024 / 1024, 2)
            else:
                storage_details[f"{name}_exists"] = False
                storage_details[f"{name}_writable"] = False
                storage_details[f"{name}_size_mb"] = 0

        # 检查磁盘空间
        if data_path.exists():
            disk_usage = psutil.disk_usage(str(data_path))
            free_percent = (disk_usage.free / disk_usage.total) * 100
            storage_details['disk_free_percent'] = round(free_percent, 2)

            if free_percent < 10:
                status = "unhealthy"
            elif free_percent < 20:
                status = "degraded"
            else:
                status = "healthy"
        else:
            status = "unhealthy"

        response_time = time.time() - start_time

        return ComponentHealth(
            name="storage",
            status=status,
            response_time=response_time,
            details=storage_details
        )

    except Exception as e:
        return ComponentHealth(
            name="storage",
            status="unhealthy",
            response_time=time.time() - start_time,
            error=str(e)
        )


async def _check_dependencies() -> ComponentHealth:
    """检查依赖项"""
    start_time = time.time()

    try:
        dependencies = {}

        # 检查重要的Python包
        required_packages = ['faiss', 'numpy', 'torch', 'transformers', 'fastapi', 'uvicorn', 'pandas', 'aiofiles']

        for package in required_packages:
            try:
                __import__(package)
                dependencies[package] = "available"
            except ImportError:
                dependencies[package] = "missing"

        # 检查CUDA可用性（如果需要）
        try:
            import torch
            dependencies['cuda_available'] = torch.cuda.is_available()
            if torch.cuda.is_available():
                dependencies['cuda_devices'] = torch.cuda.device_count()
                dependencies['cuda_memory'] = torch.cuda.get_device_properties(0).total_memory
        except ImportError:
            torch = None
            dependencies['cuda_available'] = False

        missing_deps = [k for k, v in dependencies.items() if v == "missing"]

        if missing_deps:
            status = "degraded"
        else:
            status = "healthy"

        response_time = time.time() - start_time

        return ComponentHealth(
            name="dependencies",
            status=status,
            response_time=response_time,
            details=dependencies
        )

    except Exception as e:
        return ComponentHealth(
            name="dependencies",
            status="unhealthy",
            response_time=time.time() - start_time,
            error=str(e)
        )


async def _get_system_metrics() -> SystemMetrics:
    """获取系统指标"""
    try:
        # CPU使用率
        cpu_percent = psutil.cpu_percent(interval=1)

        # 内存使用率
        memory = psutil.virtual_memory()
        memory_percent = memory.percent

        # 磁盘使用率
        disk = psutil.disk_usage('/')
        disk_percent = disk.percent

        # 系统负载
        load_average = list(psutil.getloadavg()) if hasattr(psutil, 'getloadavg') else [0.0, 0.0, 0.0]

        # 进程数量
        process_count = len(psutil.pids())

        return SystemMetrics(
            cpu_percent=cpu_percent,
            memory_percent=memory_percent,
            disk_percent=disk_percent,
            load_average=load_average,
            process_count=process_count
        )

    except Exception as e:
        logger.error(f"获取系统指标失败: {e}")
        return SystemMetrics(
            cpu_percent=0.0,
            memory_percent=0.0,
            disk_percent=0.0,
            load_average=[0.0, 0.0, 0.0],
            process_count=0
        )


class HealthChecker:
    """健康检查器"""

    def __init__(self):
        self.start_time = time.time()
        self.health_history: List[Dict[str, Any]] = []
        self.max_history = 100  # 保留最近100次检查记录

        # 性能阈值
        self.thresholds = {
            'cpu_warning': 70.0,
            'cpu_critical': 90.0,
            'memory_warning': 80.0,
            'memory_critical': 95.0,
            'disk_warning': 85.0,
            'disk_critical': 95.0,
            'response_time_warning': 1.0,
            'response_time_critical': 5.0
        }

    async def check_basic_health(self) -> HealthStatus:
        """基础健康检查"""
        try:
            uptime = time.time() - self.start_time

            # 简单检查：能否正常响应
            status = "healthy"

            return HealthStatus(
                status=status,
                timestamp=datetime.now().isoformat(),
                uptime=uptime
            )

        except Exception as e:
            logger.error(f"基础健康检查失败: {e}")
            return HealthStatus(
                status="unhealthy",
                timestamp=datetime.now().isoformat(),
                uptime=0.0
            )

    async def check_detailed_health(self) -> DetailedHealthResponse:
        """详细健康检查"""
        start_time = time.time()

        try:
            # 检查各个组件
            components = await self._check_all_components()

            # 获取系统指标
            own_system_metrics = await _get_system_metrics()

            # 获取性能指标
            performance_metrics = await self._get_performance_metrics()

            # 计算总体状态
            overall_status = self._calculate_overall_status(components, own_system_metrics)

            # 记录健康检查历史
            health_record = {
                'timestamp': datetime.now().isoformat(),
                'status': overall_status,
                'check_duration': time.time() - start_time,
                'component_count': len(components),
                'healthy_components': len([c for c in components if c.status == 'healthy'])
            }

            self.health_history.append(health_record)
            if len(self.health_history) > self.max_history:
                self.health_history.pop(0)

            return DetailedHealthResponse(
                status=overall_status,
                timestamp=datetime.now().isoformat(),
                uptime=time.time() - self.start_time,
                version="1.0.0",
                components=components,
                system_metrics=own_system_metrics,
                performance_metrics=performance_metrics
            )

        except Exception as e:
            logger.error(f"详细健康检查失败: {e}")
            raise HTTPException(status_code=500, detail=f"健康检查失败: {str(e)}")

    async def _check_all_components(self) -> List[ComponentHealth]:
        """检查所有组件"""
        components = []

        # 并行检查所有组件
        tasks = [
            self._check_embedding_service(),
            _check_vector_store(),
            _check_document_service(),
            _check_search_service(),
            _check_storage(),
            _check_dependencies()
        ]

        results = await asyncio.gather(*tasks, return_exceptions=True)

        for result in results:
            if isinstance(result, ComponentHealth):
                components.append(result)
            elif isinstance(result, Exception):
                components.append(ComponentHealth(
                    name="unknown_component",
                    status="unhealthy",
                    error=str(result)
                ))

        return components

    async def _check_embedding_service(self) -> ComponentHealth:
        """检查嵌入服务"""
        start_time = time.time()

        try:
            embedding_service = await get_embedding_service()

            # 测试嵌入生成
            test_text = "健康检查测试文本"
            embedding = await embedding_service.get_embedding(test_text)

            response_time = time.time() - start_time

            # 验证嵌入向量
            if embedding is not None and len(embedding) == settings.EMBEDDING_DIMENSION:
                status = "healthy"
                if response_time > self.thresholds['response_time_warning']:
                    status = "degraded"
            else:
                status = "unhealthy"

            return ComponentHealth(
                name="embedding_service",
                status=status,
                response_time=response_time,
                details={
                    'model_name': embedding_service.model_name,
                    'dimension': len(embedding) if embedding is not None else 0,
                    'device': getattr(embedding_service, 'device', 'unknown')
                }
            )

        except Exception as e:
            return ComponentHealth(
                name="embedding_service",
                status="unhealthy",
                response_time=time.time() - start_time,
                error=str(e)
            )

    async def _get_performance_metrics(self) -> Dict[str, Any]:
        """获取性能指标"""
        try:
            # 计算平均响应时间
            if self.health_history:
                recent_checks = self.health_history[-10:]  # 最近10次检查
                avg_check_duration = sum(h['check_duration'] for h in recent_checks) / len(recent_checks)
            else:
                avg_check_duration = 0.0

            # 可用性计算（最近24小时）
            now = datetime.now()
            recent_history = [
                h for h in self.health_history
                if datetime.fromisoformat(h['timestamp']) > now - timedelta(hours=24)
            ]

            if recent_history:
                healthy_count = len([h for h in recent_history if h['status'] == 'healthy'])
                availability = (healthy_count / len(recent_history)) * 100
            else:
                availability = 100.0  # 假设新启动的系统是健康的

            return {
                'avg_health_check_duration': round(avg_check_duration, 3),
                'health_check_count': len(self.health_history),
                'availability_24h': round(availability, 2),
                'uptime_hours': round((time.time() - self.start_time) / 3600, 2)
            }

        except Exception as e:
            logger.error(f"获取性能指标失败: {e}")
            return {}

    def _calculate_overall_status(
            self,
            components: List[ComponentHealth],
            system_metrics: SystemMetrics
    ) -> str:
        """计算总体状态"""
        # 检查组件状态
        unhealthy_components = [c for c in components if c.status == "unhealthy"]
        degraded_components = [c for c in components if c.status == "degraded"]

        # 检查系统资源
        resource_critical = (
                system_metrics.cpu_percent > self.thresholds['cpu_critical'] or
                system_metrics.memory_percent > self.thresholds['memory_critical'] or
                system_metrics.disk_percent > self.thresholds['disk_critical']
        )

        resource_warning = (
                system_metrics.cpu_percent > self.thresholds['cpu_warning'] or
                system_metrics.memory_percent > self.thresholds['memory_warning'] or
                system_metrics.disk_percent > self.thresholds['disk_warning']
        )

        # 决定总体状态
        if unhealthy_components or resource_critical:
            return "unhealthy"
        elif degraded_components or resource_warning:
            return "degraded"
        else:
            return "healthy"

    def get_health_history(self, hours: int = 24) -> List[Dict[str, Any]]:
        """获取健康检查历史"""
        cutoff_time = datetime.now() - timedelta(hours=hours)

        return [
            h for h in self.health_history
            if datetime.fromisoformat(h['timestamp']) > cutoff_time
        ]


# 全局健康检查器实例
health_checker = HealthChecker()


@router.get("/", response_model=HealthStatus)
async def basic_health_check():
    """基础健康检查 - 快速响应"""
    return await health_checker.check_basic_health()


@router.get("/detailed", response_model=DetailedHealthResponse)
async def detailed_health_check():
    """详细健康检查 - 包含所有组件和指标"""
    return await health_checker.check_detailed_health()


@router.get("/components")
async def component_health_check():
    """组件健康检查"""
    components = await health_checker._check_all_components()

    return {
        'timestamp': datetime.now().isoformat(),
        'components': [component.dict() for component in components],
        'summary': {
            'total': len(components),
            'healthy': len([c for c in components if c.status == 'healthy']),
            'degraded': len([c for c in components if c.status == 'degraded']),
            'unhealthy': len([c for c in components if c.status == 'unhealthy'])
        }
    }


@router.get("/metrics")
async def system_metrics():
    """系统指标"""
    metrics = await _get_system_metrics()
    performance = await health_checker._get_performance_metrics()

    return {
        'timestamp': datetime.now().isoformat(),
        'system': metrics.dict(),
        'performance': performance
    }


@router.get("/history")
async def health_history(hours: int = 24):
    """健康检查历史"""
    history = health_checker.get_health_history(hours)

    return {
        'period_hours': hours,
        'total_records': len(history),
        'history': history
    }


@router.get("/readiness")
async def readiness_check():
    """就绪检查 - K8s readiness probe"""
    try:
        # 检查关键组件是否就绪
        embedding_service = await get_embedding_service()
        vector_store = await get_vector_store()

        if embedding_service is None or vector_store is None:
            raise HTTPException(status_code=503, detail="Service not ready")

        return {"status": "ready", "timestamp": datetime.now().isoformat()}

    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Service not ready: {str(e)}")


@router.get("/liveness")
async def liveness_check():
    """存活检查 - K8s liveness probe"""
    try:
        # 简单的存活检查
        return {"status": "alive", "timestamp": datetime.now().isoformat()}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Service not alive: {str(e)}")


@router.get("/startup")
async def startup_check():
    """启动检查 - K8s startup probe"""
    try:
        uptime = time.time() - health_checker.start_time

        # 检查是否已经启动足够长时间
        if uptime < 10:  # 启动至少10秒
            raise HTTPException(status_code=503, detail="Service still starting")

        # 检查关键服务是否初始化
        try:
            await get_embedding_service()
            await get_vector_store()
        except Exception:
            raise HTTPException(status_code=503, detail="Services not initialized")

        return {
            "status": "started",
            "uptime": uptime,
            "timestamp": datetime.now().isoformat()
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Startup check failed: {str(e)}")


@router.get("/version")
async def version_info():
    """版本信息"""
    import sys
    import platform

    return {
        "version": "v1.0.0",
        "python_version": sys.version,
        "platform": platform.platform(),
        "timestamp": datetime.now().isoformat()
    }
