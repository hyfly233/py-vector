import asyncio
import json
import logging
import pickle
import shutil
import threading
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional

import faiss
import numpy as np

from py_faiss.config import settings

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
            embedding: Optional[np.ndarray] = None,
            metadata: Optional[Dict[str, Any]] = None
    ):
        self.doc_id = doc_id
        self.file_path = file_path
        self.file_name = file_name
        self.chunk_index = chunk_index
        self.text = text
        self.embedding = embedding
        self.metadata = metadata or {}
        self.created_at = datetime.now().isoformat()

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'doc_id': self.doc_id,
            'file_path': self.file_path,
            'file_name': self.file_name,
            'chunk_index': self.chunk_index,
            'text': self.text,
            'metadata': self.metadata,
            'created_at': self.created_at
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Document':
        """从字典创建"""
        doc = cls(
            doc_id=data['doc_id'],
            file_path=data['file_path'],
            file_name=data['file_name'],
            chunk_index=data['chunk_index'],
            text=data['text'],
            metadata=data.get('metadata', {})
        )
        doc.created_at = data.get('created_at', datetime.now().isoformat())
        return doc


class SearchResult:
    """搜索结果类"""

    def __init__(self, document: Document, score: float, rank: int):
        self.document = document
        self.score = score
        self.rank = rank

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'doc_id': self.document.doc_id,
            'file_name': self.document.file_name,
            'file_path': self.document.file_path,
            'chunk_index': self.document.chunk_index,
            'text': self.document.text,
            'score': float(self.score),
            'rank': self.rank,
            'metadata': self.document.metadata,
            'created_at': self.document.created_at
        }


class VectorStore:
    """FAISS 向量存储实现"""

    def __init__(
            self,
            dimension: int = None,
            index_type: str = "IndexFlatIP",  # 内积索引，适合归一化向量
            storage_path: str = None
    ):
        self.dimension = dimension or settings.EMBEDDING_DIMENSION
        self.index_type = index_type
        self.storage_path = Path(storage_path or settings.INDEX_PATH)

        # 确保存储目录存在
        self.storage_path.mkdir(parents=True, exist_ok=True)

        # 索引和元数据
        self.index: Optional[faiss.Index] = None
        self.documents: List[Document] = []
        self.doc_id_to_idx: Dict[str, List[int]] = {}  # doc_id -> [indices]
        self.idx_to_doc_idx: Dict[int, int] = {}  # faiss_index -> document_index

        # 文件路径
        self.index_file = self.storage_path / "faiss_index.bin"
        self.metadata_file = self.storage_path / "metadata.pkl"
        self.config_file = self.storage_path / "config.json"

        # 线程安全
        self._lock = threading.RLock()
        self._executor = ThreadPoolExecutor(max_workers=4)

        # 统计信息
        self._stats = {
            'total_documents': 0,
            'total_chunks': 0,
            'index_size': 0,
            'created_at': None,
            'last_updated': None
        }

    async def initialize(self) -> bool:
        """初始化向量存储"""
        try:
            with self._lock:
                # 尝试加载现有索引
                if await self._load_existing_index():
                    logger.info(f"加载现有索引成功: {len(self.documents)} 个文档")
                else:
                    # 创建新索引
                    await self._create_new_index()
                    logger.info(f"创建新索引成功: 维度 {self.dimension}")

                # 更新统计信息
                await self._update_stats()

            return True

        except Exception as e:
            logger.error(f"向量存储初始化失败: {e}")
            raise

    async def _create_new_index(self):
        """创建新的 FAISS 索引"""
        loop = asyncio.get_event_loop()

        def _create():
            if self.index_type == "IndexFlatIP":
                # 内积索引，适合归一化向量的余弦相似度
                self.index = faiss.IndexFlatIP(self.dimension)
            elif self.index_type == "IndexFlatL2":
                # L2 距离索引
                self.index = faiss.IndexFlatL2(self.dimension)
            elif self.index_type == "IndexIVFFlat":
                # 倒排文件索引，适合大量数据
                quantizer = faiss.IndexFlatIP(self.dimension)
                self.index = faiss.IndexIVFFlat(quantizer, self.dimension, 100)
            elif self.index_type == "IndexHNSW":
                # HNSW 索引，平衡速度和精度
                self.index = faiss.IndexHNSWFlat(self.dimension, 32)
            else:
                # 默认使用 IndexFlatIP
                self.index = faiss.IndexFlatIP(self.dimension)

            logger.info(f"创建索引类型: {type(self.index).__name__}")

        await loop.run_in_executor(self._executor, _create)

    async def _load_existing_index(self) -> bool:
        """加载现有索引"""
        if not (self.index_file.exists() and self.metadata_file.exists()):
            return False

        try:
            loop = asyncio.get_event_loop()

            def _load() -> bool:  # 明确返回类型
                # 加载 FAISS 索引
                self.index = faiss.read_index(str(self.index_file))

                # 加载元数据
                with open(self.metadata_file, 'rb') as f:
                    metadata = pickle.load(f)

                # 恢复文档列表
                self.documents = [Document.from_dict(doc_data) for doc_data in metadata['documents']]

                # 重建索引映射
                self.doc_id_to_idx = {}
                self.idx_to_doc_idx = {}

                for doc_idx, doc in enumerate(self.documents):
                    if doc.doc_id not in self.doc_id_to_idx:
                        self.doc_id_to_idx[doc.doc_id] = []

                    faiss_idx = len(self.idx_to_doc_idx)
                    self.doc_id_to_idx[doc.doc_id].append(faiss_idx)
                    self.idx_to_doc_idx[faiss_idx] = doc_idx

                # 验证维度
                if hasattr(metadata, 'dimension') and metadata['dimension'] != self.dimension:
                    logger.warning(f"索引维度不匹配: {metadata['dimension']} vs {self.dimension}")

                return True

            return await asyncio.to_thread(_load)

        except Exception as e:
            logger.error(f"加载索引失败: {e}")
            return False

    async def add_documents(
            self,
            documents: List[Document],
            embeddings: np.ndarray,
            batch_size: int = 100
    ) -> bool:
        """
        批量添加文档

        Args:
            documents: 文档列表
            embeddings: 对应的嵌入向量 [n_docs, dimension]
            batch_size: 批处理大小

        Returns:
            是否成功
        """
        if len(documents) != len(embeddings):
            raise ValueError(f"文档数量({len(documents)})与嵌入向量数量({len(embeddings)})不匹配")

        try:
            with self._lock:
                # 验证嵌入向量维度
                if embeddings.shape[1] != self.dimension:
                    raise ValueError(f"嵌入向量维度({embeddings.shape[1]})与索引维度({self.dimension})不匹配")

                # 归一化向量（如果使用内积索引）
                if isinstance(self.index, faiss.IndexFlatIP):
                    faiss.normalize_L2(embeddings)

                # 批量添加到索引
                loop = asyncio.get_event_loop()
                await loop.run_in_executor(
                    self._executor,
                    lambda: self.index.add(embeddings.astype('float32'))
                )

                # 更新文档列表和映射
                start_idx = len(self.documents)

                for i, doc in enumerate(documents):
                    # 添加到文档列表
                    doc.embedding = embeddings[i]
                    self.documents.append(doc)

                    # 更新映射
                    if doc.doc_id not in self.doc_id_to_idx:
                        self.doc_id_to_idx[doc.doc_id] = []

                    faiss_idx = start_idx + i
                    self.doc_id_to_idx[doc.doc_id].append(faiss_idx)
                    self.idx_to_doc_idx[faiss_idx] = start_idx + i

                # 保存索引
                await self._save_index()
                await self._update_stats()

                logger.info(f"✅ 成功添加 {len(documents)} 个文档到向量存储")
                return True

        except Exception as e:
            logger.error(f"添加文档失败: {e}")
            return False

    async def search(
            self,
            query_embedding: np.ndarray,
            top_k: int = 10,
            filter_doc_ids: Optional[List[str]] = None,
            min_score: float = 0.0
    ) -> List[SearchResult]:
        """
        搜索相似文档

        Args:
            query_embedding: 查询向量 [1, dimension]
            top_k: 返回结果数量
            filter_doc_ids: 过滤特定文档ID
            min_score: 最小相似度分数

        Returns:
            搜索结果列表
        """
        if self.index.ntotal == 0:
            return []

        try:
            with self._lock:
                # 确保查询向量是二维的
                if query_embedding.ndim == 1:
                    query_embedding = query_embedding.reshape(1, -1)

                # 验证维度
                if query_embedding.shape[1] != self.dimension:
                    raise ValueError(f"查询向量维度({query_embedding.shape[1]})与索引维度({self.dimension})不匹配")

                # 归一化查询向量
                if isinstance(self.index, faiss.IndexFlatIP):
                    faiss.normalize_L2(query_embedding)

                # 执行搜索
                search_k = min(top_k * 2, self.index.ntotal)  # 搜索更多结果用于过滤

                loop = asyncio.get_event_loop()
                scores, indices = await loop.run_in_executor(
                    self._executor,
                    lambda: self.index.search(query_embedding.astype('float32'), search_k)
                )

                # 处理搜索结果
                results = []
                for rank, (score, idx) in enumerate(zip(scores[0], indices[0])):
                    if idx < 0 or score < min_score:  # 无效索引或分数过低
                        continue

                    if idx not in self.idx_to_doc_idx:
                        logger.warning(f"索引 {idx} 不在文档映射中")
                        continue

                    doc_idx = self.idx_to_doc_idx[idx]
                    if doc_idx >= len(self.documents):
                        logger.warning(f"文档索引 {doc_idx} 超出范围")
                        continue

                    document = self.documents[doc_idx]

                    # 应用文档ID过滤
                    if filter_doc_ids and document.doc_id not in filter_doc_ids:
                        continue

                    results.append(SearchResult(document, float(score), rank))

                    if len(results) >= top_k:
                        break

                return results

        except Exception as e:
            logger.error(f"搜索失败: {e}")
            return []

    async def delete_document(self, doc_id: str) -> bool:
        """
        删除文档（标记删除，不重建索引）

        Args:
            doc_id: 文档ID

        Returns:
            是否成功
        """
        try:
            with self._lock:
                if doc_id not in self.doc_id_to_idx:
                    logger.warning(f"文档 {doc_id} 不存在")
                    return False

                # 标记文档为已删除
                indices_to_remove = self.doc_id_to_idx[doc_id]
                for faiss_idx in indices_to_remove:
                    if faiss_idx in self.idx_to_doc_idx:
                        doc_idx = self.idx_to_doc_idx[faiss_idx]
                        if doc_idx < len(self.documents):
                            self.documents[doc_idx].metadata['deleted'] = True
                            self.documents[doc_idx].metadata['deleted_at'] = datetime.now().isoformat()

                logger.info(f"标记删除文档: {doc_id}")
                await self._save_index()
                await self._update_stats()

                return True

        except Exception as e:
            logger.error(f"删除文档失败: {e}")
            return False

    async def rebuild_index(self) -> bool:
        """重建索引（清理已删除的文档）"""
        try:
            with self._lock:
                logger.info("开始重建索引...")

                # 过滤未删除的文档
                active_documents = []
                active_embeddings = []

                for doc in self.documents:
                    if not doc.metadata.get('deleted', False):
                        active_documents.append(doc)
                        if doc.embedding is not None:
                            active_embeddings.append(doc.embedding)

                if not active_documents:
                    logger.info("没有活跃文档，创建空索引")
                    await self._create_new_index()
                    self.documents = []
                    self.doc_id_to_idx = {}
                    self.idx_to_doc_idx = {}
                else:
                    # 重建索引
                    await self._create_new_index()

                    # 重新添加活跃文档
                    embeddings_array = np.array(active_embeddings)
                    await self.add_documents(active_documents, embeddings_array)

                logger.info(f"索引重建完成: {len(active_documents)} 个活跃文档")
                return True

        except Exception as e:
            logger.error(f"重建索引失败: {e}")
            return False

    async def _save_index(self):
        """保存索引和元数据"""
        try:
            loop = asyncio.get_event_loop()

            def _save():
                # 保存 FAISS 索引
                faiss.write_index(self.index, str(self.index_file))

                # 准备元数据
                metadata = {
                    'documents': [doc.to_dict() for doc in self.documents],
                    'dimension': self.dimension,
                    'index_type': self.index_type,
                    'total_documents': len(
                        set(doc.doc_id for doc in self.documents if not doc.metadata.get('deleted', False))),
                    'total_chunks': len([doc for doc in self.documents if not doc.metadata.get('deleted', False)]),
                    'saved_at': datetime.now().isoformat(),
                    'version': '1.0'
                }

                # 保存元数据
                with open(self.metadata_file, 'wb') as f:
                    pickle.dump(metadata, f)

                # 保存配置
                config = {
                    'dimension': self.dimension,
                    'index_type': self.index_type,
                    'storage_path': str(self.storage_path)
                }

                with open(self.config_file, 'w', encoding='utf-8') as f:
                    json.dump(config, f, indent=2, ensure_ascii=False)

            await loop.run_in_executor(self._executor, _save)

        except Exception as e:
            logger.error(f"保存索引失败: {e}")
            raise

    async def _update_stats(self):
        """更新统计信息"""
        active_docs = [doc for doc in self.documents if not doc.metadata.get('deleted', False)]
        unique_doc_ids = set(doc.doc_id for doc in active_docs)

        self._stats.update({
            'total_documents': len(unique_doc_ids),
            'total_chunks': len(active_docs),
            'index_size': self.index.ntotal if self.index else 0,
            'last_updated': datetime.now().isoformat()
        })

        if self._stats['created_at'] is None:
            self._stats['created_at'] = datetime.now().isoformat()

    async def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        await self._update_stats()
        return self._stats.copy()

    async def get_document(self, doc_id: str) -> Optional[List[Document]]:
        """获取文档的所有块"""
        if doc_id not in self.doc_id_to_idx:
            return None

        chunks = []
        for faiss_idx in self.doc_id_to_idx[doc_id]:
            if faiss_idx in self.idx_to_doc_idx:
                doc_idx = self.idx_to_doc_idx[faiss_idx]
                if doc_idx < len(self.documents):
                    doc = self.documents[doc_idx]
                    if not doc.metadata.get('deleted', False):
                        chunks.append(doc)

        return sorted(chunks, key=lambda x: x.chunk_index) if chunks else None

    async def list_documents(self, include_deleted: bool = False) -> List[Dict[str, Any]]:
        """列出所有文档"""
        doc_map = {}

        for doc in self.documents:
            if not include_deleted and doc.metadata.get('deleted', False):
                continue

            if doc.doc_id not in doc_map:
                doc_map[doc.doc_id] = {
                    'doc_id': doc.doc_id,
                    'file_name': doc.file_name,
                    'file_path': doc.file_path,
                    'chunk_count': 0,
                    'created_at': doc.created_at,
                    'metadata': doc.metadata
                }

            doc_map[doc.doc_id]['chunk_count'] += 1

        return list(doc_map.values())

    async def backup_index(self, backup_path: str) -> bool:
        """备份索引"""
        try:
            backup_dir = Path(backup_path)
            backup_dir.mkdir(parents=True, exist_ok=True)

            # 创建带时间戳的备份目录
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_subdir = backup_dir / f"backup_{timestamp}"
            backup_subdir.mkdir(exist_ok=True)

            # 复制文件
            if self.index_file.exists():
                shutil.copy2(self.index_file, backup_subdir / "faiss_index.bin")

            if self.metadata_file.exists():
                shutil.copy2(self.metadata_file, backup_subdir / "metadata.pkl")

            if self.config_file.exists():
                shutil.copy2(self.config_file, backup_subdir / "config.json")

            logger.info(f"索引备份完成: {backup_subdir}")
            return True

        except Exception as e:
            logger.error(f"备份索引失败: {e}")
            return False

    async def cleanup(self):
        """清理资源"""
        try:
            if self._executor:
                self._executor.shutdown(wait=True)
            logger.info("向量存储资源清理完成")
        except Exception as e:
            logger.error(f"清理资源失败: {e}")

    def __del__(self):
        """析构函数"""
        try:
            if hasattr(self, '_executor') and self._executor:
                self._executor.shutdown(wait=False)
        except:
            pass


# 全局实例
_vector_store: Optional[VectorStore] = None


async def get_vector_store() -> VectorStore:
    """获取全局向量存储实例"""
    global _vector_store

    if _vector_store is None:
        _vector_store = VectorStore()
        # 初始化向量存储
        await _vector_store.initialize()

    return _vector_store


async def cleanup_vector_store():
    """清理全局向量存储"""
    global _vector_store

    if _vector_store:
        await _vector_store.cleanup()
        _vector_store = None
