import os

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    PROJECT_NAME: str = "Document Search API"
    VERSION: str = "v1.0.0"
    API_V1_STR: str = "/api/v1"

    # Embedding 配置
    EMBEDDING_BASE_URL: str = "http://localhost:11434/v1"
    EMBEDDING_MODEL: str = "bge-m3"
    EMBEDDING_DIMENSION: int = 1024
    EMBEDDING_API_KEY: str = "ollama"

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
