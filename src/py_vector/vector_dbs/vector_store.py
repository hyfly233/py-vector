import logging
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any

import numpy as np

from py_vector.config import settings

logger = logging.getLogger(__name__)


class Document:
    """文档元数据类"""

    def __init__(
        self,
        doc_id: str,
        file_path: str,
        file_name: str,
        chunk_index: int,
        text: str,
        embedding: np.ndarray | None = None,
        metadata: dict[str, Any] | None = None,
    ):
        self.doc_id = doc_id
        self.file_path = file_path
        self.file_name = file_name
        self.chunk_index = chunk_index
        self.text = text
        self.embedding = embedding
        self.metadata = metadata or {}
        self.created_at = datetime.now().isoformat()

    def to_dict(self) -> dict[str, Any]:
        return {
            "doc_id": self.doc_id,
            "file_path": self.file_path,
            "file_name": self.file_name,
            "chunk_index": self.chunk_index,
            "text": self.text,
            "metadata": self.metadata,
            "created_at": self.created_at,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Document":
        doc = cls(
            doc_id=data["doc_id"],
            file_path=data["file_path"],
            file_name=data["file_name"],
            chunk_index=data["chunk_index"],
            text=data["text"],
            metadata=data.get("metadata", {}),
        )
        doc.created_at = data.get("created_at", datetime.now().isoformat())
        return doc


class SearchResult:
    """搜索结果类"""

    def __init__(self, document: Document, score: float, rank: int):
        self.document = document
        self.score = score
        self.rank = rank

    def to_dict(self) -> dict[str, Any]:
        return {
            "doc_id": self.document.doc_id,
            "file_name": self.document.file_name,
            "file_path": self.document.file_path,
            "chunk_index": self.document.chunk_index,
            "text": self.document.text,
            "score": float(self.score),
            "rank": self.rank,
            "metadata": self.document.metadata,
            "created_at": self.document.created_at,
        }


class VectorStore(ABC):
    """向量存储抽象接口

    所有后端实现（FAISS、Milvus 等）必须继承此类并实现全部抽象方法。
    使用方通过全局工厂函数 get_vector_store() 获取实例，不直接感知具体实现。
    """

    dimension: int
    index_type: str

    @abstractmethod
    async def initialize(self) -> bool:
        """初始化存储引擎"""
        ...

    @abstractmethod
    async def add_documents(
        self, documents: list[Document], embeddings: np.ndarray, batch_size: int = 100
    ) -> bool:
        """批量添加文档及其嵌入向量"""
        ...

    @abstractmethod
    async def search(
        self,
        query_embedding: np.ndarray,
        top_k: int = 10,
        filter_doc_ids: list[str] | None = None,
        min_score: float = 0.0,
    ) -> list[SearchResult]:
        """搜索相似文档"""
        ...

    @abstractmethod
    async def delete_document(self, doc_id: str) -> dict:
        """删除文档"""
        ...

    @abstractmethod
    async def get_document(self, doc_id: str) -> list[Document] | None:
        """获取文档的所有块"""
        ...

    @abstractmethod
    async def list_documents(
        self, include_deleted: bool = False
    ) -> list[dict[str, Any]]:
        """列出所有文档"""
        ...

    @abstractmethod
    async def get_stats(self) -> dict[str, Any]:
        """获取统计信息"""
        ...

    async def rebuild_index(self) -> bool:
        """重建索引（默认空操作，仅 FAISS 等本地存储需要）"""
        logger.info("重建索引不被当前后端支持，已跳过")
        return True

    async def backup_index(self, backup_path: str) -> bool:
        """备份索引（默认空操作，仅 FAISS 需要）"""
        logger.info("索引备份不被当前后端支持（由服务端托管），已跳过")
        return True

    @abstractmethod
    async def cleanup(self):
        """清理资源"""
        ...


# ---------------------------------------------------------------------------
# 全局工厂函数
# ---------------------------------------------------------------------------

_vector_store: VectorStore | None = None


async def create_vector_store() -> VectorStore:
    """根据配置创建对应后端的向量存储实例（未初始化）"""
    backend = settings.VECTOR_STORE_TYPE.lower()

    if backend == "milvus":
        from py_vector.vector_dbs.milvus_vector_store import MilvusVectorStore

        logger.info("创建 Milvus 向量存储")
        return MilvusVectorStore()
    else:
        from py_vector.vector_dbs.faiss_vector_store import FAISSVectorStore

        logger.info(f"创建 FAISS 向量存储（类型：{backend}）")
        return FAISSVectorStore()


async def get_vector_store() -> VectorStore:
    """获取全局向量存储实例（单例，延迟初始化）"""
    global _vector_store

    if _vector_store is None:
        _vector_store = await create_vector_store()
        await _vector_store.initialize()

    return _vector_store


async def cleanup_vector_store():
    """清理全局向量存储"""
    global _vector_store

    if _vector_store:
        await _vector_store.cleanup()
        _vector_store = None
