"""文件上传 / 下载端点"""

import logging
import mimetypes
from pathlib import Path

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from sqlalchemy.ext.asyncio import AsyncSession
from starlette.responses import StreamingResponse

from py_vector.config import settings
from py_vector.core.database import get_session
from py_vector.core.file_store import (
    delete_file_record,
    get_file_record,
    store_file_record,
)
from py_vector.core.s3 import _get_client
from py_vector.models.file import FileResponse, FileUploadResponse, s3_key_for

logger = logging.getLogger(__name__)
router = APIRouter()


@router.post("/upload", response_model=FileUploadResponse, summary="上传文件")
async def upload_file(
    file: UploadFile = File(...),
    doc_id: str | None = Form(None),
    session: AsyncSession = Depends(get_session),
):
    """上传文件到 S3 / 本地存储，并记录元数据到 PostgreSQL

    完整处理链路：

    ```
    上传文件
    ├── 1. 读取文件内容
    ├── 2. 保存到临时文件
    ├── 3. store_file_record()
    │    ├── 上传到 S3（media_id → 两级散列路径）
    │    │     例: TtGr1gj1Usaf → media/Tt/Gr/1gj1Usaf
    │    ├── 计算 SHA256 哈希（去重）
    │    └── 写入 PostgreSQL files 表
    │         ├── media_id: 随机短字符串（主键）
    │         ├── file_name / file_size / content_type
    │         ├── sha256 / doc_id
    │         └── storage_type: s3 | local
    └── 4. 返回 media_id
          └── GET /files/{media_id} 获取文件内容
    ```

    返回 media_id，可通过 GET /files/{media_id} 获取文件。
    """
    if not file.filename:
        raise HTTPException(400, detail="文件名不能为空")

    content = await file.read()

    # 保存到临时文件
    temp_dir = Path(settings.TEMP_PATH)
    temp_dir.mkdir(parents=True, exist_ok=True)
    temp_path = temp_dir / f"upload_{file.filename}"
    temp_path.write_bytes(content)

    try:
        content_type = (
            file.content_type
            or mimetypes.guess_type(file.filename)[0]
            or "application/octet-stream"
        )
        storage_type = "s3" if settings.S3_ENABLED else "local"

        result = await store_file_record(
            session,
            local_path=temp_path,
            file_name=file.filename,
            content_type=content_type,
            doc_id=doc_id,
            storage_type=storage_type,
        )

        if result is None:
            raise HTTPException(500, detail="文件存储失败")

        return result

    finally:
        temp_path.unlink(missing_ok=True)


@router.get("/{media_id}", summary="获取文件")
async def download_file(
    media_id: str,
    session: AsyncSession = Depends(get_session),
):
    """根据 media_id 获取文件内容

    从 S3 或本地存储读取文件并流式返回。
    """
    record = await get_file_record(session, media_id)
    if record is None:
        raise HTTPException(404, detail="文件不存在")

    content_type = record.content_type or "application/octet-stream"  # type: ignore[union-attr]
    file_name = record.file_name or media_id  # type: ignore[union-attr]

    if record.storage_type == "s3":  # type: ignore[comparison-overlap]
        # 从 S3 流式返回
        s3_key = s3_key_for(media_id)
        try:
            client = _get_client()
            obj = client.get_object(Bucket=settings.S3_BUCKET, Key=s3_key)
            return StreamingResponse(
                obj["Body"].iter_chunks(),
                media_type=content_type,
                headers={
                    "Content-Disposition": f'attachment; filename="{file_name}"',
                    "Content-Length": str(record.file_size or ""),
                },
            )
        except Exception as e:
            raise HTTPException(500, detail=f"文件读取失败: {e}")

    else:
        # 本地文件
        path = Path(record.file_name)  # type: ignore[arg-type]
        if not path.exists():
            raise HTTPException(404, detail="文件不存在")
        return StreamingResponse(
            open(path, "rb"),
            media_type=content_type,
            headers={
                "Content-Disposition": f'attachment; filename="{file_name}"',
            },
        )


@router.delete("/{media_id}", summary="删除文件")
async def delete_file(
    media_id: str,
    session: AsyncSession = Depends(get_session),
):
    """删除文件（含 S3 对象和 PG 记录）"""
    ok = await delete_file_record(session, media_id)
    if not ok:
        raise HTTPException(404, detail="文件不存在")
    return {"message": "文件已删除", "media_id": media_id}


@router.get("/{media_id}/meta", response_model=FileResponse, summary="获取文件元数据")
async def get_file_meta(
    media_id: str,
    session: AsyncSession = Depends(get_session),
):
    """获取文件的元数据信息"""
    record = await get_file_record(session, media_id)
    if record is None:
        raise HTTPException(404, detail="文件不存在")
    return record
