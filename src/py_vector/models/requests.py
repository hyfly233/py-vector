from typing import List

from pydantic import BaseModel, Field


class SearchRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=1000, description="搜索查询")
    top_k: int = Field(default=5, ge=1, le=20, description="返回结果数量")


class DocumentUploadResponse(BaseModel):
    document_id: str
    filename: str
    status: str
    chunks_count: int
    message: str


class SearchResult(BaseModel):
    score: float
    file_name: str
    file_path: str
    chunk_index: int
    text: str


class SearchResponse(BaseModel):
    query: str
    results: List[SearchResult]
    total_results: int
    processing_time: float


class HealthResponse(BaseModel):
    status: str
    version: str
    search_engine_status: str
    total_documents: int
    total_chunks: int
