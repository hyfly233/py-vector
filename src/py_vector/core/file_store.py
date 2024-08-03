"""文件存储服务：S3 上传/下载 + PostgreSQL 元数据管理"""

import asyncio
import hashlib
import logging
from datetime import datetime
from pathlib import Path

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from py_vector.config import settings
from py_vector.core.s3 import _get_client
from py_vector.models.file import (
    FileRecord,
    FileUploadResponse,
    generate_media_id,
    s3_key_for,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# S3 操作（同步，跑在 executor 中）
# ---------------------------------------------------------------------------


def _upload_to_s3_sync(local_path: Path, s3_key: str) -> bool:
    """同步上传文件到 S3"""
    try:
        client = _get_client()
        client.upload_file(str(local_path), settings.S3_BUCKET, s3_key)
        logger.info("✅ S3 上传成功: %s/%s", settings.S3_BUCKET, s3_key)
        return True
    except Exception as e:
        logger.error("❌ S3 上传失败: %s", e)
        return False


def _delete_from_s3_sync(s3_key: str) -> bool:
    """同步从 S3 删除文件"""
    try:
        client = _get_client()
        client.delete_object(Bucket=settings.S3_BUCKET, Key=s3_key)
        logger.info("S3 已删除: %s/%s", settings.S3_BUCKET, s3_key)
        return True
    except Exception as e:
        logger.error("S3 删除失败: %s", e)
        return False


# ---------------------------------------------------------------------------
# 异步公共方法
# ---------------------------------------------------------------------------


async def upload_to_s3(local_path: Path, s3_key: str) -> bool:
    """异步上传本地文件到 S3"""
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, _upload_to_s3_sync, local_path, s3_key)


async def delete_from_s3(s3_key: str) -> bool:
    """异步从 S3 删除文件"""
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, _delete_from_s3_sync, s3_key)


# ---------------------------------------------------------------------------
# 文件入库（PG）
# ---------------------------------------------------------------------------


async def _compute_sha256(path: Path) -> str:
    """计算文件的 SHA256 哈希"""
    loop = asyncio.get_event_loop()

    def _hash():
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(65536), b""):
                h.update(chunk)
        return h.hexdigest()

    return await loop.run_in_executor(None, _hash)


async def store_file_record(
    session: AsyncSession,
    *,
    local_path: Path,
    file_name: str,
    content_type: str = "application/octet-stream",
    doc_id: str | None = None,
    storage_type: str = "s3",
) -> FileUploadResponse | None:
    """将文件存储到 S3 并写入 PG 元数据

    Args:
        session: 数据库会话
        local_path: 本地文件路径
        file_name: 原始文件名
        content_type: MIME 类型
        doc_id: 关联的文档 ID
        storage_type: 存储类型

    Returns:
        FileUploadResponse 或 None（失败时）
    """
    media_id = generate_media_id()
    s3_key = s3_key_for(media_id)

    # 上传到 S3
    if storage_type == "s3":
        ok = await upload_to_s3(local_path, s3_key)
        if not ok:
            return None
    else:
        s3_key = str(local_path)

    # 计算哈希
    sha256 = await _compute_sha256(local_path)
    file_size = local_path.stat().st_size

    # 写入 PG
    record = FileRecord(
        media_id=media_id,
        doc_id=doc_id,
        file_name=file_name,
        file_size=file_size,
        content_type=content_type,
        sha256=sha256,
        storage_type=storage_type,
        created_at=datetime.now(),
    )
    session.add(record)
    await session.commit()

    logger.info(
        "✅ 文件记录已创建: media_id=%s, bucket=%s, key=%s",
        media_id,
        settings.S3_BUCKET,
        s3_key,
    )

    return FileUploadResponse(
        media_id=media_id,
        file_name=file_name,
        file_size=file_size,
        storage_type=storage_type,
    )


async def get_file_record(session: AsyncSession, media_id: str) -> FileRecord | None:
    """按 media_id 查询文件记录"""
    result = await session.execute(
        select(FileRecord).where(FileRecord.media_id == media_id)
    )
    return result.scalar_one_or_none()


async def delete_file_record(session: AsyncSession, media_id: str) -> bool:
    """删除文件记录及 S3 对象"""
    record = await get_file_record(session, media_id)
    if record is None:
        return False

    if record.storage_type == "s3":  # type: ignore[comparison-overlap]
        s3_key = s3_key_for(media_id)
        await delete_from_s3(s3_key)

    await session.delete(record)
    await session.commit()
    logger.info("已删除文件记录: %s", media_id)
    return True
