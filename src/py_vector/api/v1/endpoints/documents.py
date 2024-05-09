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
    """列出文档"""
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
    """
    try:
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
    """获取文档详情"""
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
    """获取文档处理状态"""
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
    """删除文档"""
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
    """获取统计信息"""
    document_service = await get_document_service()
    return await document_service.get_statistics()


@router.post(
    path="/admin/rebuild-index",
    summary="重建索引（管理员功能）",
)
async def rebuild_index():
    """重建索引（管理员功能）"""
    document_service = await get_document_service()
    return await document_service.rebuild_index()


@router.post(
    path="/admin/backup",
    summary="备份数据（管理员功能）",
)
async def backup_data(backup_path: str):
    """备份数据（管理员功能）"""
    document_service = await get_document_service()
    return await document_service.backup_data(backup_path)
