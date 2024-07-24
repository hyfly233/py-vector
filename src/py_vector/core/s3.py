import asyncio
import logging
from pathlib import Path

import boto3
from botocore.exceptions import ClientError

from py_vector.config import settings

logger = logging.getLogger(__name__)


def _get_client():
    """创建 S3 客户端"""
    return boto3.client(
        "s3",
        endpoint_url=settings.S3_ENDPOINT,
        aws_access_key_id=settings.S3_ACCESS_KEY,
        aws_secret_access_key=settings.S3_SECRET_KEY,
        region_name=settings.S3_REGION,
        use_ssl=settings.S3_SECURE,
    )


# ---------------------------------------------------------------------------
# 桶检查
# ---------------------------------------------------------------------------


def _check_bucket_sync() -> bool:
    """同步检查 S3 桶"""
    client = _get_client()
    try:
        client.head_bucket(Bucket=settings.S3_BUCKET)
        logger.info("✅ S3 存储桶已存在: %s", settings.S3_BUCKET)
        return True
    except ClientError as e:
        if e.response["Error"]["Code"] == "404":
            client.create_bucket(Bucket=settings.S3_BUCKET)
            logger.info("✅ S3 存储桶已创建: %s", settings.S3_BUCKET)
            return True
        logger.error("❌ S3 存储桶检查失败: %s", e)
        return False
    except Exception as e:
        logger.error("❌ S3 连接失败: %s", e)
        return False


async def ensure_bucket_exists() -> bool:
    """检查 S3 存储桶是否存在

    在异步上下文中执行同步的 boto3 调用。

    Returns:
        True 表示桶就绪，False 表示连接失败
    """
    if not settings.S3_ENABLED:
        return True

    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, _check_bucket_sync)


# ---------------------------------------------------------------------------
# 文件上传 / 下载
# ---------------------------------------------------------------------------


def _download_file_sync(key: str, dest: Path) -> bool:
    """从 S3 下载文件到本地（同步，跑在 executor 中）

    Args:
        key: S3 对象键（例如 documents/uuid_filename.pdf）
        dest: 本地目标路径

    Returns:
        True 成功，False 失败
    """
    try:
        client = _get_client()
        client.download_file(settings.S3_BUCKET, key, str(dest))
        logger.info("✅ S3 下载成功: %s → %s", key, dest)
        return True
    except Exception as e:
        logger.error("❌ S3 下载失败: %s", e)
        return False


async def download_from_s3(key: str, dest: Path) -> bool:
    """从 S3 异步下载文件到本地"""
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, _download_file_sync, key, dest)
