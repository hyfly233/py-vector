from typing import List, Optional, Any, Generic, TypeVar
from datetime import datetime
from pydantic import BaseModel, Field

T = TypeVar('T')


class BaseResponse(BaseModel, Generic[T]):
    """基础响应模型"""
    success: bool = Field(default=True, description="请求是否成功")
    message: str = Field(default="操作成功", description="响应消息")
    data: Optional[T] = Field(default=None, description="响应数据")
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat(), description="响应时间戳")
    request_id: Optional[str] = Field(default=None, description="请求ID")


class ErrorResponse(BaseModel):
    """错误响应模型"""
    success: bool = Field(default=False, description="请求是否成功")
    message: str = Field(..., description="错误消息")
    error_code: Optional[str] = Field(default=None, description="错误代码")
    details: Optional[Any] = Field(default=None, description="错误详情")
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat(), description="响应时间戳")


class DocumentUploadResponse(BaseModel):
    """文档上传响应"""
    doc_id: str = Field(..., description="文档ID")
    filename: str = Field(..., description="文件名")
    status: str = Field(..., description="处理状态")
    message: str = Field(..., description="状态消息")
    chunks_count: Optional[int] = Field(default=None, description="文本块数量")


class SearchResult(BaseModel):
    """搜索结果项"""
    doc_id: str = Field(..., description="文档ID")
    file_name: str = Field(..., description="文件名")
    file_path: str = Field(..., description="文件路径")
    chunk_index: int = Field(..., description="文本块索引")
    text: str = Field(..., description="文本内容")
    score: float = Field(..., description="相似度分数")
    highlighted_text: Optional[str] = Field(default=None, description="高亮文本")
    metadata: Optional[dict] = Field(default=None, description="元数据")


class SearchResponse(BaseModel):
    """搜索响应"""
    query: str = Field(..., description="搜索查询")
    results: List[SearchResult] = Field(default=[], description="搜索结果")
    total_results: int = Field(default=0, description="结果总数")
    processing_time: float = Field(..., description="处理时间（秒）")
    search_time: Optional[float] = Field(default=None, description="搜索时间（秒）")


class DocumentListResponse(BaseModel):
    """文档列表响应"""
    documents: List[dict] = Field(default=[], description="文档列表")
    total: int = Field(default=0, description="文档总数")
    page: int = Field(default=1, description="当前页码")
    page_size: int = Field(default=20, description="每页大小")
    total_pages: int = Field(default=1, description="总页数")


class DocumentDetailResponse(BaseModel):
    """文档详情响应"""
    doc_id: str = Field(..., description="文档ID")
    filename: str = Field(..., description="文件名")
    file_path: str = Field(..., description="文件路径")
    file_size: int = Field(..., description="文件大小")
    chunks_count: int = Field(..., description="文本块数量")
    created_at: str = Field(..., description="创建时间")
    status: str = Field(..., description="处理状态")
    metadata: Optional[dict] = Field(default=None, description="元数据")


class ProcessingStatusResponse(BaseModel):
    """处理状态响应"""
    status: str = Field(..., description="处理状态")
    progress: int = Field(default=0, description="处理进度（0-100）")
    message: str = Field(..., description="状态消息")
    filename: str = Field(..., description="文件名")
    started_at: Optional[str] = Field(default=None, description="开始时间")
    completed_at: Optional[str] = Field(default=None, description="完成时间")
    failed_at: Optional[str] = Field(default=None, description="失败时间")
    error: Optional[str] = Field(default=None, description="错误信息")


class StatisticsResponse(BaseModel):
    """统计信息响应"""
    total_documents: int = Field(default=0, description="文档总数")
    total_chunks: int = Field(default=0, description="文本块总数")
    storage_used: int = Field(default=0, description="存储使用量（字节）")
    processing_queue: int = Field(default=0, description="处理队列长度")
    last_updated: str = Field(default_factory=lambda: datetime.now().isoformat(), description="最后更新时间")


class HealthResponse(BaseModel):
    """健康检查响应"""
    status: str = Field(..., description="服务状态")
    version: str = Field(..., description="版本号")
    search_engine_status: str = Field(..., description="搜索引擎状态")
    total_documents: int = Field(default=0, description="文档总数")
    total_chunks: int = Field(default=0, description="文本块总数")
    system_info: Optional[dict] = Field(default=None, description="系统信息")


class DeleteResponse(BaseModel):
    """删除响应"""
    doc_id: str = Field(..., description="文档ID")
    status: str = Field(..., description="删除状态")
    message: str = Field(..., description="状态消息")
    deleted_chunks: Optional[int] = Field(default=None, description="删除的文本块数量")