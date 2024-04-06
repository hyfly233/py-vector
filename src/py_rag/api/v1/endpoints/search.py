import time
from typing import Optional, List, Dict, Any

from fastapi import APIRouter, Query
from fastapi import Depends, HTTPException
from pydantic import BaseModel

from py_faiss.core.search_engine import SearchEngine
from py_faiss.dependencies import get_search_engine
from py_faiss.models.requests import SearchResponse
from py_faiss.services.document_service import get_document_service
from py_faiss.services.search_service import SearchOptions, SearchFilter, get_search_service

router = APIRouter()


class SearchRequest(BaseModel):
    """搜索请求模型"""
    query: str
    top_k: Optional[int] = 10
    filter_doc_ids: Optional[List[str]] = None
    min_score: Optional[float] = 0.1


class AdvancedSearchRequest(BaseModel):
    query: str
    search_type: Optional[str] = "vector"  # vector, hybrid, keyword
    top_k: Optional[int] = 10
    enable_rerank: Optional[bool] = False
    enable_highlight: Optional[bool] = True
    enable_summary: Optional[bool] = False
    chunk_merge: Optional[bool] = True
    diversity_threshold: Optional[float] = 0.7

    # 过滤器
    doc_ids: Optional[List[str]] = None
    file_names: Optional[List[str]] = None
    file_types: Optional[List[str]] = None
    date_range: Optional[List[str]] = None  # [start_date, end_date]
    min_score: Optional[float] = 0.1
    metadata_filters: Optional[dict] = None


@router.post(path="/", response_model=SearchResponse)
async def search_documents(request: SearchRequest, search_engine: SearchEngine = Depends(get_search_engine)):
    """搜索文档"""
    start_time = time.time()

    try:
        results = await search_engine.search(request.query, request.top_k)
        processing_time = time.time() - start_time

        return SearchResponse(
            query=request.query,
            results=results,
            total_results=len(results),
            processing_time=processing_time
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"搜索失败: {str(e)}")


@router.get(
    path="/documents",
    summary="搜索文档（GET 方式）",
    response_model=Dict[str, Any],
    response_description="搜索文档结果"
)
async def search_documents_get(
        q: str = Query(..., description="搜索查询"),
        top_k: int = Query(10, ge=1, le=50, description="返回结果数量"),
        min_score: float = Query(0.1, ge=0.0, le=1.0, description="最小得分过滤")
):
    """搜索文档（GET 方式）"""
    document_service = await get_document_service()

    return await document_service.search_documents(
        query=q,
        top_k=top_k,
        min_score=min_score
    )


@router.post(
    path="/documents",
    summary="搜索文档",
    response_model=Dict[str, Any],
    response_description="搜索文档结果"
)
async def search_documents(request: SearchRequest):
    """搜索文档"""
    document_service = await get_document_service()

    return await document_service.search_documents(
        query=request.query,
        top_k=request.top_k,
        filter_doc_ids=request.filter_doc_ids,
        min_score=request.min_score
    )


@router.get(
    path="/stats",
    summary="获取搜索引擎统计信息",
    response_model=Dict[str, Any],
    response_description="搜索引擎统计信息"
)
async def get_search_stats(search_engine: SearchEngine = Depends(get_search_engine)):
    """获取搜索引擎统计信息"""
    try:
        stats = await search_engine.get_stats()
        return stats
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"获取统计信息失败: {str(e)}")


@router.post(
    path="/advanced",
    summary="高级搜索",
    response_model=Dict[str, Any],
    response_description="高级搜索结果"
)
async def advanced_search(request: AdvancedSearchRequest, user_id: Optional[str] = None):
    """高级搜索"""
    try:
        search_service = await get_search_service()

        # 构建搜索选项
        options = SearchOptions(
            search_type=request.search_type,
            top_k=request.top_k,
            enable_rerank=request.enable_rerank,
            enable_highlight=request.enable_highlight,
            enable_summary=request.enable_summary,
            chunk_merge=request.chunk_merge,
            diversity_threshold=request.diversity_threshold
        )

        # 构建过滤器
        filters = SearchFilter(
            doc_ids=request.doc_ids,
            file_names=request.file_names,
            file_types=request.file_types,
            date_range=tuple(request.date_range) if request.date_range else None,
            min_score=request.min_score,
            metadata_filters=request.metadata_filters
        )

        # 执行搜索
        result = await search_service.search(
            query=request.query,
            options=options,
            filters=filters,
            user_id=user_id
        )

        return result

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get(
    path="/suggestions",
    summary="获取搜索建议",
)
async def get_search_suggestions(q: str = Query(..., description="部分查询"), limit: int = Query(5, ge=1, le=20)):
    """获取搜索建议"""
    search_service = await get_search_service()
    suggestions = await search_service.get_search_suggestions(q, limit)

    return {
        'query': q,
        'suggestions': suggestions
    }


@router.get(
    path="/statistics",
    summary="获取搜索统计",
)
async def get_search_statistics():
    """获取搜索统计"""
    search_service = await get_search_service()
    stats = await search_service.get_search_statistics()

    return stats


@router.delete(
    path="/cache/clear",
    summary="清理搜索缓存",
)
async def clear_search_cache():
    """清理搜索缓存"""
    search_service = await get_search_service()
    await search_service.clear_cache()

    return {'message': '搜索缓存已清理'}
