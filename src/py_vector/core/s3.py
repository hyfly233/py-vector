import asyncio
import logging

import boto3
from botocore.exceptions import ClientError

from py_vector.config import settings

logger = logging.getLogger(__name__)


def _check_bucket_sync() -> bool:
    """同步检查 S3 桶"""
    client = boto3.client(
        "s3",
        endpoint_url=settings.S3_ENDPOINT,
        aws_access_key_id=settings.S3_ACCESS_KEY,
        aws_secret_access_key=settings.S3_SECRET_KEY,
        region_name=settings.S3_REGION,
        use_ssl=settings.S3_SECURE,
    )
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
