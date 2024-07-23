import json
import os
from typing import Any

from pydantic import field_validator
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    PROJECT_NAME: str = "Document Search API"
    VERSION: str = "v1.0.0"
    API_V1_STR: str = "/api/v1"

    # LLM 配置（RAG 生成）
    LLM_BASE_URL: str = "http://localhost:11434/v1"
    LLM_MODEL: str = "qwen2.5"
    LLM_API_KEY: str = "ollama"
    LLM_TEMPERATURE: float = 0.7
    LLM_MAX_TOKENS: int = 2048
    LLM_CONTEXT_LENGTH: int = 8192

    # 多模态配置（预留）
    MULTIMODAL_ENABLED: bool = False
    MULTIMODAL_BASE_URL: str = ""
    MULTIMODAL_MODEL: str = ""
    MULTIMODAL_EMBEDDING_MODEL: str = ""

    # Embedding 配置
    EMBEDDING_BASE_URL: str = "http://localhost:11434/v1"
    EMBEDDING_MODEL: str = "bge-m3"
    EMBEDDING_DIMENSION: int = 1024
    EMBEDDING_API_KEY: str = "ollama"

    # Reranker 配置（重排序）
    RERANKER_BASE_URL: str = ""
    RERANKER_MODEL: str = "bge-reranker-v2-m3"
    RERANKER_API_KEY: str = "ollama"
    RERANKER_ENABLED: bool = False
    RERANKER_TOP_K: int = 10

    # 模型组配置（主备切换，Hermes 风格）
    # 空字典时退回到 LLM_* / EMBEDDING_* / RERANKER_* 单个字段。
    # 配置后每种模型按顺序尝试，失败自动切换到下一个。
    # JSON 格式示例：
    #   MODEL_GROUPS='{"embedding":[{"base_url":"http://localhost:11434/v1","model":"bge-m3","api_key":"ollama","dimension":1024}],"llm":[{"base_url":"http://localhost:11434/v1","model":"qwen2.5","api_key":"ollama","temperature":0.7}]}'
    MODEL_GROUPS: dict[str, list[dict[str, Any]]] = {}

    # 向量存储配置
    VECTOR_STORE_TYPE: str = "faiss"  # faiss | milvus
    MILVUS_URI: str = ""  # 空值时使用 STORAGE_PATH/milvus.db（本地模式）

    # 存储配置
    STORAGE_PATH: str = "./storage"
    INDEX_PATH: str = "./storage/indexes"
    DOCUMENTS_PATH: str = "./storage/documents"
    TEMP_PATH: str = "./storage/temp"

    # 文档处理配置
    MAX_FILE_SIZE: int = 50 * 1024 * 1024  # 50MB
    CHUNK_SIZE: int = 512
    CHUNK_OVERLAP: int = 50
    CHUNKING_STRATEGY: str = "recursive"

    # API 配置
    ALLOWED_HOSTS: list[str] = ["*"]
    MAX_SEARCH_RESULTS: int = 20

    # 日志配置
    LOG_LEVEL: str = "INFO"
    LOG_FILE: str = "./logs/app.log"

    # MinIO / S3 兼容对象存储配置
    S3_ENABLED: bool = False
    S3_ENDPOINT: str = "http://localhost:9000"
    S3_ACCESS_KEY: str = "minioadmin"
    S3_SECRET_KEY: str = "minioadmin"
    S3_BUCKET: str = "py-vector"
    S3_REGION: str = "us-east-1"
    S3_SECURE: bool = False  # HTTPS

    # PostgreSQL 数据库配置
    PG_ENABLED: bool = False
    PG_HOST: str = "localhost"
    PG_PORT: int = 5432
    PG_USER: str = "postgres"
    PG_PASSWORD: str = "password"
    PG_DATABASE: str = "mydb"
    PG_SCHEMA: str = "public"
    PG_POOL_SIZE: int = 10
    PG_MAX_OVERFLOW: int = 20
    PG_ECHO: bool = False  # SQL 日志

    @field_validator("MODEL_GROUPS", mode="before")
    @classmethod
    def parse_model_groups(cls, v: Any) -> dict[str, list[dict[str, Any]]]:
        """支持环境变量传入 JSON 字符串或 dict"""
        if isinstance(v, str):
            if not v.strip():
                return {}
            return json.loads(v)
        return v or {}

    class Config:
        env_file = ".env"
        case_sensitive = True


settings = Settings()

# 确保存储目录存在
for path in [
    settings.STORAGE_PATH,
    settings.INDEX_PATH,
    settings.DOCUMENTS_PATH,
    settings.TEMP_PATH,
]:
    os.makedirs(path, exist_ok=True)
