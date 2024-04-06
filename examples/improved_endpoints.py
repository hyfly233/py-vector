# FastAPI JSON 响应最佳实践示例

from typing import List, Optional
from fastapi import APIRouter, HTTPException, Query, Depends
from pydantic import BaseModel

from py_rag.models.responses import (
    BaseResponse, SearchResponse, DocumentUploadResponse, 
    DocumentDetailResponse, ProcessingStatusResponse
)
from py_rag.utils.response_helper import ResponseHelper
from py_rag.services.document_service import get_document_service

router = APIRouter()

# 示例1: 使用标准响应模型
@router.get(
    "/documents",
    response_model=BaseResponse[List[DocumentDetailResponse]],
    summary="获取文档列表"
)
async def list_documents_improved(
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100)
):
    """改进的文档列表端点 - 使用标准响应格式"""
    try:
        document_service = await get_document_service()
        documents = await document_service.list_documents(page, page_size, False)
        
        return ResponseHelper.success(
            data=documents,
            message="获取文档列表成功"
        )
    except Exception as e:
        return ResponseHelper.internal_error(f"获取文档列表失败: {str(e)}")


# 示例2: 使用分页响应
@router.get(
    "/documents/paginated",
    summary="获取分页文档列表"
)
async def list_documents_paginated(
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100)
):
    """分页文档列表"""
    try:
        document_service = await get_document_service()
        result = await document_service.list_documents_paginated(page, page_size)
        
        return ResponseHelper.paginated(
            items=result['documents'],
            total=result['total'],
            page=page,
            page_size=page_size,
            message="获取文档列表成功"
        )
    except Exception as e:
        return ResponseHelper.internal_error(f"获取文档列表失败: {str(e)}")


# 示例3: 处理不同状态码
@router.get(
    "/documents/{doc_id}",
    response_model=BaseResponse[DocumentDetailResponse],
    summary="获取文档详情"
)
async def get_document_details_improved(doc_id: str):
    """改进的文档详情端点"""
    try:
        document_service = await get_document_service()
        details = await document_service.get_document_details(doc_id)
        
        if details is None:
            return ResponseHelper.not_found("文档不存在")
        
        return ResponseHelper.success(
            data=details,
            message="获取文档详情成功"
        )
    except Exception as e:
        return ResponseHelper.internal_error(f"获取文档详情失败: {str(e)}")


# 示例4: 异步处理响应
@router.post(
    "/documents/upload",
    response_model=BaseResponse[DocumentUploadResponse],
    status_code=202,
    summary="上传文档"
)
async def upload_document_improved(
    # ... 参数定义
):
    """改进的文档上传端点 - 异步处理"""
    try:
        # 处理上传逻辑
        result = await document_service.upload_and_process_document(
            file_content=content,
            filename=filename,
            user_id=user_id
        )
        
        return ResponseHelper.accepted(
            data=result,
            message="文档上传成功，正在后台处理"
        )
    except Exception as e:
        return ResponseHelper.bad_request(f"文档上传失败: {str(e)}")


# 示例5: 搜索响应
@router.post(
    "/search",
    response_model=BaseResponse[SearchResponse],
    summary="搜索文档"
)
async def search_documents_improved(search_request: SearchRequest):
    """改进的搜索端点"""
    try:
        document_service = await get_document_service()
        search_result = await document_service.search_documents(
            query=search_request.query,
            top_k=search_request.top_k
        )
        
        return ResponseHelper.success(
            data=search_result,
            message="搜索完成"
        )
    except Exception as e:
        return ResponseHelper.internal_error(f"搜索失败: {str(e)}")
