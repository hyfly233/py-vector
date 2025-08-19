"""异步数据库引擎与会话管理"""

import logging
from collections.abc import AsyncGenerator, AsyncIterator
from contextlib import asynccontextmanager

from sqlalchemy.ext.asyncio import (
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from py_vector.config import settings

logger = logging.getLogger(__name__)

_engine = None
_async_session_maker = None


def _build_dsn() -> str:
    """从配置构建 asyncpg 连接串"""
    return (
        f"postgresql+asyncpg://{settings.PG_USER}:{settings.PG_PASSWORD}"
        f"@{settings.PG_HOST}:{settings.PG_PORT}/{settings.PG_DATABASE}"
    )


async def init_db():
    """初始化数据库引擎和表结构"""
    global _engine, _async_session_maker

    if not settings.PG_ENABLED:
        logger.info("PostgreSQL 未启用（PG_ENABLED=false），跳过数据库初始化")
        return

    dsn = _build_dsn()
    _engine = create_async_engine(
        dsn,
        echo=settings.PG_ECHO,
        pool_size=settings.PG_POOL_SIZE,
        max_overflow=settings.PG_MAX_OVERFLOW,
    )
    _async_session_maker = async_sessionmaker(
        _engine, class_=AsyncSession, expire_on_commit=False
    )

    # 自动建表
    from py_vector.models.file import Base

    async with _engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    logger.info("✅ 数据库引擎已初始化")


async def dispose_db():
    """关闭数据库连接池"""
    global _engine, _async_session_maker
    if _engine:
        await _engine.dispose()
        _engine = None
        _async_session_maker = None
        logger.info("数据库连接池已关闭")


async def get_session() -> AsyncGenerator[AsyncSession]:
    """获取异步数据库会话（FastAPI 依赖注入用）"""
    if not settings.PG_ENABLED or _async_session_maker is None:
        raise RuntimeError("PostgreSQL 未启用，请设置 PG_ENABLED=true")
    async with _async_session_maker() as session:
        yield session


@asynccontextmanager
async def get_standalone_session() -> AsyncIterator[AsyncSession]:
    """获取独立异步数据库会话（服务层用）

    用于 document_service 等非端点场景，自动管理 session 生命周期。
    使用方式:
        async with get_standalone_session() as session:
            ...
    """
    if not settings.PG_ENABLED or _async_session_maker is None:
        raise RuntimeError("PostgreSQL 未启用，请设置 PG_ENABLED=true")
    async with _async_session_maker() as session:
        try:
            yield session
        except Exception:
            await session.rollback()
            raise
        await session.commit()
