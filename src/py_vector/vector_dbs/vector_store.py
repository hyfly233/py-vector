import logging
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any

import numpy as np

from py_vector.config import settings

logger = logging.getLogger(__name__)


class Document:
    """文档元数据类

    Args:
        doc_id: 文档唯一标识
        file_path: 文件路径
        file_name: 文件名
        chunk_index: 文本块索引
        text: 文本内容
        embedding: 嵌入向量（可选）
        metadata: 附加元数据（可选）
    """

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
        """将文档转换为字典

        Returns:
            包含文档所有字段的字典
        """
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
        """从字典创建文档实例

        Args:
            data: 包含文档字段的字典

        Returns:
            Document: 新创建的文档实例
        """
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
    """搜索结果类

    Args:
        document: 匹配的文档对象
        score: 相似度分数
        rank: 排序位置
    """

    def __init__(self, document: Document, score: float, rank: int):
        self.document = document
        self.score = score
        self.rank = rank

    def to_dict(self) -> dict[str, Any]:
        """将搜索结果转换为字典

        Returns:
            包含搜索结果所有字段的字典
        """
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
        """初始化存储引擎

        Returns:
            bool: 初始化是否成功
        """
        ...

    @abstractmethod
    async def add_documents(
        self, documents: list[Document], embeddings: np.ndarray, batch_size: int = 100
    ) -> bool:
        """批量添加文档及其嵌入向量

        Args:
            documents: 文档对象列表
            embeddings: 嵌入向量数组，形状为 (len(documents), dimension)
            batch_size: 每批处理的文档数量，默认 100

        Returns:
            bool: 添加是否成功
        """
        ...

    @abstractmethod
    async def search(
        self,
        query_embedding: np.ndarray,
        top_k: int = 10,
        filter_doc_ids: list[str] | None = None,
        min_score: float = 0.0,
    ) -> list[SearchResult]:
        """搜索相似文档

        Args:
            query_embedding: 查询向量
            top_k: 返回的最匹配结果数量，默认 10
            filter_doc_ids: 可选的文档 ID 过滤列表
            min_score: 最小相似度分数阈值，默认 0.0

        Returns:
            list[SearchResult]: 搜索结果列表，按相似度降序排列
        """
        ...

    @abstractmethod
    async def delete_document(self, doc_id: str) -> dict:
        """删除文档

        Args:
            doc_id: 要删除的文档 ID

        Returns:
            dict: 删除结果，包含 status（状态）和 message（消息）
        """
        ...

    @abstractmethod
    async def get_document(self, doc_id: str) -> list[Document] | None:
        """获取文档的所有块

        Args:
            doc_id: 文档 ID

        Returns:
            list[Document] | None: 文档的所有文本块列表，不存在则返回 None
        """
        ...

    @abstractmethod
    async def list_documents(
        self, include_deleted: bool = False
    ) -> list[dict[str, Any]]:
        """列出所有文档

        Args:
            include_deleted: 是否包含已删除的文档，默认 False

        Returns:
            list[dict[str, Any]]: 文档摘要信息列表
        """
        ...

    @abstractmethod
    async def get_stats(self) -> dict[str, Any]:
        """获取统计信息

        Returns:
            dict[str, Any]: 包含 total_documents、total_chunks 等统计数据的字典
        """
        ...

    async def rebuild_index(self) -> bool:
        """重建索引（默认空操作，仅 FAISS 等本地存储需要）

        Returns:
            bool: 重建是否成功
        """
        logger.info("重建索引不被当前后端支持，已跳过")
        return True

    async def backup_index(self, backup_path: str) -> bool:
        """备份索引（默认空操作，仅 FAISS 需要）

        Args:
            backup_path: 备份目标路径

        Returns:
            bool: 备份是否成功
        """
        logger.info("索引备份不被当前后端支持（由服务端托管），已跳过")
        return True

    @abstractmethod
    async def cleanup(self):
        """清理资源

        Returns:
            None
        """
        ...


# ---------------------------------------------------------------------------
# 全局工厂函数
# ---------------------------------------------------------------------------

_vector_store: VectorStore | None = None


async def create_vector_store() -> VectorStore:
    """根据配置创建对应后端的向量存储实例（未初始化）

    Returns:
        VectorStore: 创建的向量存储实例（未初始化，需调用 initialize()）
    """
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
    """获取全局向量存储实例（单例，延迟初始化）

    Returns:
        VectorStore: 全局向量存储单例
    """
    global _vector_store

    if _vector_store is None:
        _vector_store = await create_vector_store()
        await _vector_store.initialize()

    return _vector_store


async def cleanup_vector_store():
    """清理全局向量存储

    Returns:
        None
    """
    global _vector_store

    if _vector_store:
        await _vector_store.cleanup()
        _vector_store = None
