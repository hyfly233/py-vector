import logging

from fastapi import APIRouter, File, Form, HTTPException, Query, UploadFile

from py_vector.services.document_service import get_document_service

router = APIRouter()

logger = logging.getLogger(__name__)


@router.get(
    path="/",
    summary="列出文档",
)
async def list_documents(
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
    include_deleted: bool = Query(False),
):
    """列出文档

    Args:
        page (int): 页码，从 1 开始
        page_size (int): 每页数量，范围 1-100
        include_deleted (bool): 是否包含已删除的文档

    Returns:
        dict: 文档列表，包含 total（总数）和 documents（文档列表）字段
    """
    document_service = await get_document_service()
    return await document_service.list_documents(page, page_size, include_deleted)


@router.post(
    path="/upload",
    summary="上传文档",
)
async def upload_document(
    file: UploadFile = File(...),
    index: bool = Form(
        True, description="是否存入向量库，false 时仅保存文件不建立索引"
    ),
    user_id: str | None = Form(None),
):
    """上传文档

    通过 `index` 参数控制是否将文档内容向量化并存入向量库：
    - `index=true`（默认）：上传 → 提取 → 切片 → 嵌入 → 索引
    - `index=false`：仅保存文件，不做向量化处理

    Args:
        file (UploadFile): 上传的文件
        index (bool): 是否存入向量库，false 时仅保存文件不建立索引
        user_id (str | None): 用户 ID，可选

    Returns:
        dict: 上传结果，包含 doc_id（文档 ID）和 status（处理状态）等字段
    """
    try:
        if not file.filename:
            raise HTTPException(status_code=400, detail="文件名不能为空")
        document_service = await get_document_service()

        if not document_service.document_processor.is_supported_file(file.filename):
            raise HTTPException(
                status_code=400, detail=f"不支持的文件类型: {file.filename}"
            )

        content = await file.read()

        result = await document_service.upload_and_process_document(
            file_content=content,
            filename=file.filename,
            user_id=user_id,
            index=index,
        )

        return result

    except Exception as e:
        logger.error(f"上传文档失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get(
    path="/{doc_id}",
    summary="获取文档详情",
)
async def get_document_details(doc_id: str):
    """获取文档详情

    Args:
        doc_id (str): 文档 ID

    Returns:
        dict: 文档详情信息，包含元数据、内容等字段；如果文档不存在则返回 404
    """
    document_service = await get_document_service()
    details = await document_service.get_document_details(doc_id)

    if details is None:
        raise HTTPException(status_code=404, detail="文档不存在")

    return details


@router.get(
    path="/{doc_id}/status",
    summary="获取文档处理状态",
)
async def get_document_status(doc_id: str):
    """获取文档处理状态

    Args:
        doc_id (str): 文档 ID

    Returns:
        dict: 文档处理状态信息；如果文档不存在则返回 404
    """
    document_service = await get_document_service()
    status = await document_service.get_processing_status(doc_id)

    if status is None:
        raise HTTPException(status_code=404, detail="文档不存在")

    return status


@router.delete(
    path="/{doc_id}",
    summary="删除文档",
)
async def delete_document(doc_id: str):
    """删除文档

    Args:
        doc_id (str): 文档 ID

    Returns:
        dict: 删除结果，包含 status（状态）和 message（消息）字段
    """
    document_service = await get_document_service()
    result = await document_service.delete_document(doc_id)

    if result["status"] == "error":
        raise HTTPException(status_code=400, detail=result["message"])

    return result


@router.get(
    path="/stats/overview",
    summary="获取文档统计概览",
)
async def get_statistics():
    """获取统计信息

    Returns:
        dict: 文档统计概览，包含文档总数、索引数量等字段
    """
    document_service = await get_document_service()
    return await document_service.get_statistics()


@router.post(
    path="/admin/rebuild-index",
    summary="重建索引（管理员功能）",
)
async def rebuild_index():
    """重建索引（管理员功能）

    Returns:
        dict: 重建索引结果，包含 status（状态）和 message（消息）等字段
    """
    document_service = await get_document_service()
    return await document_service.rebuild_index()


@router.post(
    path="/admin/backup",
    summary="备份数据（管理员功能）",
)
async def backup_data(backup_path: str):
    """备份数据（管理员功能）

    Args:
        backup_path (str): 备份文件路径

    Returns:
        dict: 备份结果，包含 status（状态）和 message（消息）等字段
    """
    document_service = await get_document_service()
    return await document_service.backup_data(backup_path)
