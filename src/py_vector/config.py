import os

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

    # 存储配置
    STORAGE_PATH: str = "./storage"
    INDEX_PATH: str = "./storage/indexes"
    DOCUMENTS_PATH: str = "./storage/documents"
    TEMP_PATH: str = "./storage/temp"

    # 文档处理配置
    MAX_FILE_SIZE: int = 50 * 1024 * 1024  # 50MB
    CHUNK_SIZE: int = 512
    CHUNK_OVERLAP: int = 50

    # API 配置
    ALLOWED_HOSTS: list[str] = ["*"]
    MAX_SEARCH_RESULTS: int = 20

    # 日志配置
    LOG_LEVEL: str = "INFO"

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
