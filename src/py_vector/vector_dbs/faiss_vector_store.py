import asyncio
import json
import logging
import pickle
import shutil
import threading
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path
from typing import Any

import faiss
import numpy as np

from py_vector.config import settings
from py_vector.vector_dbs.vector_store import Document, SearchResult, VectorStore

logger = logging.getLogger(__name__)


class FAISSVectorStore(VectorStore):
    """FAISS 向量存储实现

    Args:
        dimension: 向量维度，默认使用配置值
        index_type: 索引类型，默认 IndexFlatIP
        storage_path: 索引存储路径，默认使用配置值
    """

    def __init__(
        self,
        dimension: int | None = None,
        index_type: str = "IndexFlatIP",
        storage_path: str | None = None,
    ):
        self.dimension = dimension or settings.EMBEDDING_DIMENSION
        self.index_type = index_type
        self.storage_path = Path(storage_path or settings.INDEX_PATH)

        # 确保存储目录存在
        self.storage_path.mkdir(parents=True, exist_ok=True)

        # 索引和元数据
        self.index: faiss.Index | None = None
        self.documents: list[Document] = []
        self.doc_id_to_idx: dict[str, list[int]] = {}
        self.idx_to_doc_idx: dict[int, int] = {}

        # 文件路径
        self.index_file = self.storage_path / "faiss_index.bin"
        self.metadata_file = self.storage_path / "metadata.pkl"
        self.config_file = self.storage_path / "config.json"

        # 线程安全
        self._lock = threading.RLock()
        self._executor = ThreadPoolExecutor(max_workers=2)

        # 统计信息
        self._stats: dict[str, Any] = {
            "total_documents": 0,
            "total_chunks": 0,
            "index_size": 0,
            "last_updated": None,
            "created_at": None,
        }

    async def initialize(self) -> bool:
        """初始化 FAISS 索引

        Returns:
            bool: 初始化是否成功
        """
        try:
            # 尝试加载已有索引
            if self.index_file.exists() and self.metadata_file.exists():
                await self._load_index()
                logger.info(f"✅ 已加载 FAISS 索引：{self.index.ntotal} 个向量")
            else:
                # 创建新索引
                self.index = faiss.IndexFlatIP(self.dimension)
                logger.info(
                    "✅ 已创建新 FAISS 索引（%s，维度=%s）",
                    self.index_type,
                    self.dimension,
                )

            return True

        except Exception as e:
            logger.error(f"❌ FAISS 初始化失败: {e}")
            raise

    async def add_documents(
        self, documents: list[Document], embeddings: np.ndarray, batch_size: int = 100
    ) -> bool:
        """批量添加文档及其嵌入向量到 FAISS 索引

        Args:
            documents: 文档对象列表
            embeddings: 嵌入向量数组，形状为 (len(documents), dimension)
            batch_size: 每批处理的文档数量，默认 100

        Returns:
            bool: 添加是否成功
        """
        if len(documents) != len(embeddings):
            raise ValueError(
                f"文档数量({len(documents)})与嵌入向量数量({len(embeddings)})不匹配"
            )

        try:
            with self._lock:
                if embeddings.shape[1] != self.dimension:
                    raise ValueError(
                        f"嵌入向量维度({embeddings.shape[1]})与索引维度({self.dimension})不匹配"
                    )

                # 归一化向量（内积索引需要归一化后的余弦相似度）
                if isinstance(self.index, faiss.IndexFlatIP):
                    faiss.normalize_L2(embeddings)

                # 批量添加到索引
                loop = asyncio.get_event_loop()
                await loop.run_in_executor(
                    self._executor,
                    lambda: self.index.add(embeddings.astype("float32")),
                )

                # 更新文档列表和映射
                start_idx = len(self.documents)
                for i, doc in enumerate(documents):
                    doc.embedding = embeddings[i]
                    self.documents.append(doc)
                    if doc.doc_id not in self.doc_id_to_idx:
                        self.doc_id_to_idx[doc.doc_id] = []
                    faiss_idx = start_idx + i
                    self.doc_id_to_idx[doc.doc_id].append(faiss_idx)
                    self.idx_to_doc_idx[faiss_idx] = start_idx + i

                # 持久化
                await self._save_index()
                await self._update_stats()

                logger.info(f"✅ 成功添加 {len(documents)} 个文档到 FAISS")
                return True

        except Exception as e:
            logger.error(f"添加文档失败: {e}")
            return False

    async def search(
        self,
        query_embedding: np.ndarray,
        top_k: int = 10,
        filter_doc_ids: list[str] | None = None,
        min_score: float = 0.0,
    ) -> list[SearchResult]:
        """在 FAISS 索引中搜索相似文档

        Args:
            query_embedding: 查询向量
            top_k: 返回的最匹配结果数量，默认 10
            filter_doc_ids: 可选的文档 ID 过滤列表
            min_score: 最小相似度分数阈值，默认 0.0

        Returns:
            list[SearchResult]: 搜索结果列表，按相似度降序排列
        """
        if self.index is None or self.index.ntotal == 0:
            return []

        query_vec = query_embedding.reshape(1, -1).astype("float32")
        if isinstance(self.index, faiss.IndexFlatIP):
            faiss.normalize_L2(query_vec)

        k = min(top_k, self.index.ntotal)
        scores, indices = self.index.search(query_vec, k)

        results = []
        for score, faiss_idx in zip(scores[0], indices[0]):
            if faiss_idx < 0 or score < min_score:
                continue
            if faiss_idx in self.idx_to_doc_idx:
                doc_idx = self.idx_to_doc_idx[faiss_idx]
                doc = self.documents[doc_idx]
                if doc.metadata.get("deleted", False):
                    continue
                if filter_doc_ids and doc.doc_id not in filter_doc_ids:
                    continue
                results.append(
                    SearchResult(document=doc, score=float(score), rank=len(results))
                )

        return results

    async def delete_document(self, doc_id: str) -> dict:
        """从 FAISS 索引删除文档（标记删除）

        Args:
            doc_id: 要删除的文档 ID

        Returns:
            dict: 删除结果，包含 status、deleted_chunks 和 message
        """
        try:
            with self._lock:
                if doc_id not in self.doc_id_to_idx:
                    return {"status": "error", "message": "文档不存在"}

                # 标记删除
                for faiss_idx in self.doc_id_to_idx[doc_id]:
                    if faiss_idx in self.idx_to_doc_idx:
                        doc_idx = self.idx_to_doc_idx[faiss_idx]
                        self.documents[doc_idx].metadata["deleted"] = True

                deleted_count = len(self.doc_id_to_idx[doc_id])
                del self.doc_id_to_idx[doc_id]

            await self._save_index()
            await self._update_stats()

            logger.info(f"✅ 已标记删除文档 {doc_id}（{deleted_count} 个块）")
            return {
                "status": "success",
                "deleted_chunks": deleted_count,
                "message": f"已删除 {deleted_count} 个文本块",
            }

        except Exception as e:
            logger.error(f"删除文档失败: {e}")
            return {"status": "error", "message": str(e)}

    async def get_document(self, doc_id: str) -> list[Document] | None:
        """获取 FAISS 中文档的所有块

        Args:
            doc_id: 文档 ID

        Returns:
            list[Document] | None: 文档的文本块列表（已排序）
        """
        if doc_id not in self.doc_id_to_idx:
            return None

        chunks = []
        for faiss_idx in self.doc_id_to_idx[doc_id]:
            if faiss_idx in self.idx_to_doc_idx:
                doc_idx = self.idx_to_doc_idx[faiss_idx]
                if doc_idx < len(self.documents):
                    doc = self.documents[doc_idx]
                    if not doc.metadata.get("deleted", False):
                        chunks.append(doc)

        return sorted(chunks, key=lambda x: x.chunk_index) if chunks else None

    async def list_documents(
        self, include_deleted: bool = False
    ) -> list[dict[str, Any]]:
        """列出 FAISS 索引中的所有文档

        Args:
            include_deleted: 是否包含已标记删除的文档，默认 False

        Returns:
            list[dict[str, Any]]: 文档摘要信息列表
        """
        doc_map: dict[str, dict[str, Any]] = {}

        for doc in self.documents:
            if not include_deleted and doc.metadata.get("deleted", False):
                continue
            if doc.doc_id not in doc_map:
                doc_map[doc.doc_id] = {
                    "doc_id": doc.doc_id,
                    "file_name": doc.file_name,
                    "file_path": doc.file_path,
                    "chunk_count": 0,
                    "created_at": doc.created_at,
                    "metadata": doc.metadata,
                }
            doc_map[doc.doc_id]["chunk_count"] += 1

        return list(doc_map.values())

    async def get_stats(self) -> dict[str, Any]:
        """获取 FAISS 索引统计信息

        Returns:
            dict[str, Any]: 向量存储统计信息
        """
        await self._update_stats()
        return {
            **self._stats.copy(),
            "dimension": self.dimension,
            "index_type": self.index_type,
        }

    async def _update_stats(self):
        """更新内部统计信息

        Returns:
            None
        """
        active_docs = [
            doc for doc in self.documents if not doc.metadata.get("deleted", False)
        ]
        unique_doc_ids = set(doc.doc_id for doc in active_docs)

        self._stats.update(
            {
                "total_documents": len(unique_doc_ids),
                "total_chunks": len(active_docs),
                "index_size": self.index.ntotal if self.index else 0,
                "last_updated": datetime.now().isoformat(),
            }
        )
        if self._stats["created_at"] is None:
            self._stats["created_at"] = datetime.now().isoformat()

    async def rebuild_index(self) -> bool:
        """重建索引（清理标记删除的数据）

        Returns:
            bool: 重建是否成功
        """
        try:
            with self._lock:
                active_docs = [
                    doc
                    for doc in self.documents
                    if not doc.metadata.get("deleted", False)
                ]
                if not active_docs:
                    self.index = faiss.IndexFlatIP(self.dimension)
                    self.documents = []
                    self.doc_id_to_idx.clear()
                    self.idx_to_doc_idx.clear()
                    await self._save_index()
                    await self._update_stats()
                    logger.info("索引已重置（无有效文档）")
                    return True

                # 收集有效嵌入
                valid_embeddings = []
                valid_docs = []
                for doc in active_docs:
                    if doc.embedding is not None:
                        valid_embeddings.append(doc.embedding)
                        valid_docs.append(doc)

                if not valid_embeddings:
                    logger.warning("没有带嵌入向量的有效文档，跳过重建")
                    return False

                embeddings = np.array(valid_embeddings, dtype=np.float32)

                # 重建索引
                new_index = faiss.IndexFlatIP(self.dimension)
                if isinstance(self.index, faiss.IndexFlatIP):
                    faiss.normalize_L2(embeddings)
                new_index.add(embeddings)

                self.index = new_index
                self.documents = valid_docs

                # 重建映射
                self.doc_id_to_idx.clear()
                self.idx_to_doc_idx.clear()
                for i, doc in enumerate(valid_docs):
                    if doc.doc_id not in self.doc_id_to_idx:
                        self.doc_id_to_idx[doc.doc_id] = []
                    self.doc_id_to_idx[doc.doc_id].append(i)
                    self.idx_to_doc_idx[i] = i

                await self._save_index()
                await self._update_stats()

                logger.info(f"✅ 索引重建完成：{len(valid_docs)} 个有效块")
                return True

        except Exception as e:
            logger.error(f"重建索引失败: {e}")
            return False

    async def backup_index(self, backup_path: str) -> bool:
        """备份索引文件

        Args:
            backup_path: 备份目标目录路径

        Returns:
            bool: 备份是否成功
        """
        try:
            backup_dir = Path(backup_path)
            backup_dir.mkdir(parents=True, exist_ok=True)

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_subdir = backup_dir / f"backup_{timestamp}"
            backup_subdir.mkdir(exist_ok=True)

            if self.index_file.exists():
                shutil.copy2(self.index_file, backup_subdir / "faiss_index.bin")
            if self.metadata_file.exists():
                shutil.copy2(self.metadata_file, backup_subdir / "metadata.pkl")
            if self.config_file.exists():
                shutil.copy2(self.config_file, backup_subdir / "config.json")

            logger.info(f"FAISS 索引备份完成: {backup_subdir}")
            return True

        except Exception as e:
            logger.error(f"备份索引失败: {e}")
            return False

    # ------------------------------------------------------------------
    # 内部持久化
    # ------------------------------------------------------------------

    async def _save_index(self):
        """保存索引到磁盘

        Returns:
            None
        """
        try:
            loop = asyncio.get_event_loop()

            def _save():
                # 保存 FAISS 索引
                faiss.write_index(self.index, str(self.index_file))

                # 保存文档元数据（不含嵌入向量）
                metadata = {
                    "documents": [
                        {
                            "doc_id": d.doc_id,
                            "file_path": d.file_path,
                            "file_name": d.file_name,
                            "chunk_index": d.chunk_index,
                            "text": d.text,
                            "metadata": d.metadata,
                            "created_at": d.created_at,
                        }
                        for d in self.documents
                    ],
                    "doc_id_to_idx": {
                        doc_id: indices
                        for doc_id, indices in self.doc_id_to_idx.items()
                    },
                    "idx_to_doc_idx": {
                        str(faiss_idx): doc_idx
                        for faiss_idx, doc_idx in self.idx_to_doc_idx.items()
                    },
                    "total_chunks": len(
                        [
                            doc
                            for doc in self.documents
                            if not doc.metadata.get("deleted", False)
                        ]
                    ),
                    "saved_at": datetime.now().isoformat(),
                    "version": "1.0",
                }

                with open(self.metadata_file, "wb") as f:
                    pickle.dump(metadata, f)

                # 保存配置
                config = {
                    "dimension": self.dimension,
                    "index_type": self.index_type,
                    "storage_path": str(self.storage_path),
                }
                with open(self.config_file, "w", encoding="utf-8") as f:
                    json.dump(config, f, indent=2, ensure_ascii=False)

            await loop.run_in_executor(self._executor, _save)

        except Exception as e:
            logger.error(f"保存索引失败: {e}")
            raise

    async def _load_index(self):
        """从磁盘加载索引

        Returns:
            None
        """
        loop = asyncio.get_event_loop()

        def _load():
            # 加载 FAISS 索引
            index = faiss.read_index(str(self.index_file))

            # 加载元数据
            with open(self.metadata_file, "rb") as f:
                metadata = pickle.load(f)

            # 加载配置
            if self.config_file.exists():
                with open(self.config_file, encoding="utf-8") as f:
                    config = json.load(f)
                self.dimension = config.get("dimension", self.dimension)
                self.index_type = config.get("index_type", self.index_type)

            documents = []
            for doc_data in metadata.get("documents", []):
                doc = Document.from_dict(doc_data)
                documents.append(doc)

            return index, documents, metadata

        self.index, self.documents, metadata = await loop.run_in_executor(
            self._executor, _load
        )

        # 重建映射
        self.doc_id_to_idx = {
            doc_id: indices
            for doc_id, indices in metadata.get("doc_id_to_idx", {}).items()
        }
        self.idx_to_doc_idx = {
            int(faiss_idx): doc_idx
            for faiss_idx, doc_idx in metadata.get("idx_to_doc_idx", {}).items()
        }

    async def cleanup(self):
        """清理 FAISS 资源

        Returns:
            None
        """
        try:
            if self._executor:
                self._executor.shutdown(wait=True)
            logger.info("FAISS 向量存储资源清理完成")
        except Exception as e:
            logger.error(f"清理资源失败: {e}")

    def __del__(self):
        try:
            if hasattr(self, "_executor") and self._executor:
                self._executor.shutdown(wait=False)
        except Exception:
            pass
