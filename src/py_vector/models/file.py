"""文件存储数据模型"""

import secrets
from datetime import datetime

from pydantic import BaseModel
from sqlalchemy import Column, DateTime, Integer, String
from sqlalchemy.orm import DeclarativeBase


class Base(DeclarativeBase):
    pass


class FileRecord(Base):
    """文件元数据表"""

    __tablename__ = "files"

    media_id = Column(String, primary_key=True, comment="文件唯一标识")
    doc_id = Column(String, nullable=True, index=True, comment="关联文档 ID")
    file_name = Column(String, nullable=False, comment="原始文件名")
    file_size = Column(Integer, default=0, comment="文件大小（字节）")
    content_type = Column(
        String, default="application/octet-stream", comment="MIME 类型"
    )
    sha256 = Column(String, nullable=True, comment="内容哈希，用于去重")
    storage_type = Column(String, default="s3", comment="存储类型: s3 | local")
    created_at = Column(DateTime, default=datetime.now, comment="上传时间")


# ---------------------------------------------------------------------------
# 辅助函数
# ---------------------------------------------------------------------------


def generate_media_id(length: int = 20) -> str:
    """生成短随机字符串作为 media_id"""
    return secrets.token_urlsafe(length)[:length]


def s3_key_for(media_id: str) -> str:
    """根据 media_id 生成 S3 路径

    media_id = "aB3xK9mPQr"
    S3 key  = "media/aB/3x/K9mPQr"
    """
    if len(media_id) < 4:
        return f"media/{media_id}"
    return f"media/{media_id[:2]}/{media_id[2:4]}/{media_id[4:]}"


# ---------------------------------------------------------------------------
# API 模型
# ---------------------------------------------------------------------------


class FileResponse(BaseModel):
    """文件元数据响应"""

    media_id: str
    doc_id: str | None = None
    file_name: str
    file_size: int
    content_type: str
    sha256: str | None = None
    storage_type: str
    created_at: datetime


class FileUploadResponse(BaseModel):
    """文件上传响应"""

    media_id: str
    file_name: str
    file_size: int
    storage_type: str
    message: str = "文件上传成功"
