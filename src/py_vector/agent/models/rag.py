from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field


class Source(BaseModel):
    """引用来源

    Args:
        doc_id: 文档 ID
        file_name: 文件名
        text: 引用文本
        score: 相似度分数
        chunk_index: 文本块索引
    """

    doc_id: str = Field(..., description="文档 ID")
    file_name: str = Field(..., description="文件名")
    text: str = Field(..., description="引用文本")
    score: float = Field(..., description="相似度分数")
    chunk_index: int = Field(0, description="文本块索引")


class AnswerWithCitations(BaseModel):
    """带引用的 RAG 回答

    Args:
        answer: 最终回答
        sources: 引用来源列表
        confidence: 可信度：high / medium / low
        timestamp: 生成时间
    """

    answer: str = Field(..., description="最终回答")
    sources: list[Source] = Field(default=[], description="引用来源列表")
    confidence: str = Field(
        default="medium",
        description="可信度：high / medium / low",
    )
    timestamp: str = Field(
        default_factory=lambda: datetime.now().isoformat(),
        description="生成时间",
    )


class RAGQuery(BaseModel):
    """RAG 查询请求

    Args:
        query: 用户问题
        top_k: 检索chunk数量
    """

    query: str = Field(..., min_length=1, max_length=2000, description="用户问题")
    top_k: int = Field(default=10, ge=1, le=50, description="检索chunk数量")


class RAGResponse(BaseModel):
    """RAG 查询响应

    Args:
        query: 原始问题
        answer: 最终回答
        sources: 引用来源
        confidence: 可信度
        processing_time: 处理时间（秒）
        error: 错误信息（非空时表示降级结果）
    """

    query: str = Field(..., description="原始问题")
    answer: str = Field(..., description="最终回答")
    sources: list[dict[str, Any]] = Field(default=[], description="引用来源")
    confidence: str = Field(default="medium", description="可信度")
    processing_time: float = Field(..., description="处理时间（秒）")
    error: str | None = Field(
        default=None, description="错误信息（非空时表示降级结果）"
    )
