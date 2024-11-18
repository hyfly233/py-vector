import asyncio
import logging
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

from py_vector.config import settings
from py_vector.core.vector_store import Document, SearchResult, VectorStore

logger = logging.getLogger(__name__)

# Milvus 字段名
_VECTOR_FIELD = "vector"
_DOC_ID_FIELD = "doc_id"
_FILE_NAME_FIELD = "file_name"
_FILE_PATH_FIELD = "file_path"
_CHUNK_INDEX_FIELD = "chunk_index"
_TEXT_FIELD = "text"
_CREATED_AT_FIELD = "created_at"
_METADATA_FIELD = "metadata"


class MilvusVectorStore(VectorStore):
    """Milvus 向量存储实现

    支持两种模式：
    - **本地模式**（默认）：使用 ``<STORAGE_PATH>/milvus.db``，
      基于 LanceDB，无需运行 Milvus 服务
    - **服务端模式**：设置 ``MILVUS_URI`` 为 ``http://host:19530``
    """

    COLLECTION_NAME = "py_vector"

    def __init__(self, uri: str | None = None, dimension: int | None = None):
        self.dimension = dimension or settings.EMBEDDING_DIMENSION
        self.index_type = "milvus"

        # URI 解析
        configured_uri = uri or settings.MILVUS_URI
        if configured_uri:
            self._uri = configured_uri
        else:
            self._uri = str(Path(settings.STORAGE_PATH) / "milvus.db")

        self._client: Any = None  # MilvusClient
        self._collection_ready = False
        self._executor = ThreadPoolExecutor(max_workers=2)

    # ------------------------------------------------------------------
    # 内部工具
    # ------------------------------------------------------------------

    def _assert_client(self):
        if self._client is None:
            raise RuntimeError("MilvusVectorStore 未初始化，请先调用 initialize()")

    # ------------------------------------------------------------------
    # 生命周期
    # ------------------------------------------------------------------

    async def initialize(self) -> bool:
        try:
            from pymilvus import MilvusClient

            self._client = MilvusClient(self._uri)

            loop = asyncio.get_event_loop()
            self._collection_ready = await loop.run_in_executor(
                self._executor, self._init_collection
            )
            return self._collection_ready

        except Exception as e:
            logger.error(f"❌ Milvus 初始化失败: {e}")
            self._client = None
            raise

    def _init_collection(self) -> bool:
        self._assert_client()

        if self._client.has_collection(self.COLLECTION_NAME):
            desc = self._client.describe_collection(self.COLLECTION_NAME)
            coll_dim = desc.get("params", {}).get("dimension", 0)
            if coll_dim and coll_dim != self.dimension:
                logger.warning(f"集合维度({coll_dim})与配置({self.dimension})不一致")
            logger.info(
                f"✅ 已连接 Milvus 集合「{self.COLLECTION_NAME}」（URI={self._uri}）"
            )
            return True

        self._client.create_collection(
            collection_name=self.COLLECTION_NAME,
            dimension=self.dimension,
            primary_field_name="id",
            vector_field_name=_VECTOR_FIELD,
            metric_type="IP",
            auto_id=True,
            enable_dynamic_field=False,
        )
        logger.info(
            f"✅ 已创建 Milvus 集合「{self.COLLECTION_NAME}」"
            f"（维度={self.dimension}，URI={self._uri}）"
        )
        return True

    async def cleanup(self):
        if self._client:
            self._client.close()
            self._client = None
        if self._executor:
            self._executor.shutdown(wait=True)
        self._collection_ready = False
        logger.info("Milvus 连接已关闭")

    # ------------------------------------------------------------------
    # CRUD
    # ------------------------------------------------------------------

    async def add_documents(
        self, documents: list[Document], embeddings: np.ndarray, batch_size: int = 100
    ) -> bool:
        if len(documents) != len(embeddings):
            raise ValueError(
                f"文档数量({len(documents)})与嵌入向量数量({len(embeddings)})不匹配"
            )

        try:
            rows = [
                {
                    _VECTOR_FIELD: embeddings[i].tolist(),
                    _DOC_ID_FIELD: doc.doc_id,
                    _FILE_NAME_FIELD: doc.file_name,
                    _FILE_PATH_FIELD: doc.file_path,
                    _CHUNK_INDEX_FIELD: doc.chunk_index,
                    _TEXT_FIELD: doc.text,
                    _CREATED_AT_FIELD: doc.created_at,
                    _METADATA_FIELD: doc.metadata,
                }
                for i, doc in enumerate(documents)
            ]

            loop = asyncio.get_event_loop()

            def _insert():
                self._assert_client()
                for start in range(0, len(rows), batch_size):
                    self._client.insert(
                        self.COLLECTION_NAME, rows[start : start + batch_size]
                    )
                return True

            result = await loop.run_in_executor(self._executor, _insert)
            logger.info(f"✅ 成功插入 {len(rows)} 个文档到 Milvus")
            return result

        except Exception as e:
            logger.error(f"Milvus 插入失败: {e}")
            return False

    async def search(
        self,
        query_embedding: np.ndarray,
        top_k: int = 10,
        filter_doc_ids: list[str] | None = None,
        min_score: float = 0.0,
    ) -> list[SearchResult]:
        if not self._collection_ready:
            return []

        query_vec = query_embedding.reshape(1, -1).astype("float32").tolist()

        expr = None
        if filter_doc_ids:
            quoted = ",".join(f'"{d}"' for d in filter_doc_ids)
            expr = f"{_DOC_ID_FIELD} in [{quoted}]"

        def _search():
            self._assert_client()
            return self._client.search(
                collection_name=self.COLLECTION_NAME,
                data=query_vec,
                anns_field=_VECTOR_FIELD,
                limit=top_k,
                output_fields=[
                    _DOC_ID_FIELD,
                    _FILE_NAME_FIELD,
                    _FILE_PATH_FIELD,
                    _CHUNK_INDEX_FIELD,
                    _TEXT_FIELD,
                    _CREATED_AT_FIELD,
                    _METADATA_FIELD,
                ],
                search_params={"metric_type": "IP"},
                expr=expr,
            )

        try:
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(self._executor, _search)

            if not result:
                return []

            search_results = []
            for hits in result:
                for hit in hits:
                    score = hit.get("distance", 0.0)
                    if score < min_score:
                        continue
                    entity = hit.get("entity") or hit
                    doc = Document(
                        doc_id=entity.get(_DOC_ID_FIELD, ""),
                        file_path=entity.get(_FILE_PATH_FIELD, ""),
                        file_name=entity.get(_FILE_NAME_FIELD, ""),
                        chunk_index=entity.get(_CHUNK_INDEX_FIELD, 0),
                        text=entity.get(_TEXT_FIELD, ""),
                        metadata=entity.get(_METADATA_FIELD, {}),
                    )
                    doc.created_at = entity.get(
                        _CREATED_AT_FIELD, datetime.now().isoformat()
                    )
                    search_results.append(
                        SearchResult(
                            document=doc,
                            score=float(score),
                            rank=len(search_results),
                        )
                    )

            return search_results

        except Exception as e:
            logger.error(f"Milvus 搜索失败: {e}")
            return []

    async def delete_document(self, doc_id: str) -> dict:
        def _delete():
            self._assert_client()
            expr = f'{_DOC_ID_FIELD} == "{doc_id}"'
            result = self._client.delete(self.COLLECTION_NAME, filter=expr)
            if isinstance(result, dict):
                return result.get("delete_count", 0)
            return 0

        try:
            loop = asyncio.get_event_loop()
            deleted = await loop.run_in_executor(self._executor, _delete)
            logger.info(f"✅ 已从 Milvus 删除文档 {doc_id}（{deleted} 条）")
            return {
                "status": "success",
                "deleted_chunks": deleted,
                "message": f"已删除 {deleted} 条记录",
            }
        except Exception as e:
            logger.error(f"Milvus 删除失败: {e}")
            return {"status": "error", "message": str(e)}

    async def get_document(self, doc_id: str) -> list[Document] | None:
        def _query():
            self._assert_client()
            expr = f'{_DOC_ID_FIELD} == "{doc_id}"'
            return self._client.query(
                collection_name=self.COLLECTION_NAME,
                filter=expr,
                output_fields=[
                    _DOC_ID_FIELD,
                    _FILE_NAME_FIELD,
                    _FILE_PATH_FIELD,
                    _CHUNK_INDEX_FIELD,
                    _TEXT_FIELD,
                    _CREATED_AT_FIELD,
                    _METADATA_FIELD,
                ],
            )

        try:
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(self._executor, _query)

            if not result:
                return None

            docs = [
                Document(
                    doc_id=row.get(_DOC_ID_FIELD, ""),
                    file_path=row.get(_FILE_PATH_FIELD, ""),
                    file_name=row.get(_FILE_NAME_FIELD, ""),
                    chunk_index=row.get(_CHUNK_INDEX_FIELD, 0),
                    text=row.get(_TEXT_FIELD, ""),
                    metadata=row.get(_METADATA_FIELD, {}),
                )
                for row in result
            ]
            for row, doc in zip(result, docs):
                doc.created_at = row.get(_CREATED_AT_FIELD, datetime.now().isoformat())

            return sorted(docs, key=lambda x: x.chunk_index)

        except Exception as e:
            logger.error(f"Milvus 查询失败: {e}")
            return None

    async def list_documents(
        self, include_deleted: bool = False
    ) -> list[dict[str, Any]]:
        def _list():
            self._assert_client()
            return self._client.query(
                collection_name=self.COLLECTION_NAME,
                filter="",
                output_fields=[
                    _DOC_ID_FIELD,
                    _FILE_NAME_FIELD,
                    _FILE_PATH_FIELD,
                    _CREATED_AT_FIELD,
                ],
                limit=10000,
            )

        try:
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(self._executor, _list)

            doc_map: dict[str, dict[str, Any]] = {}
            for row in result:
                did = row.get(_DOC_ID_FIELD, "")
                if did not in doc_map:
                    doc_map[did] = {
                        "doc_id": did,
                        "file_name": row.get(_FILE_NAME_FIELD, ""),
                        "file_path": row.get(_FILE_PATH_FIELD, ""),
                        "chunk_count": 0,
                        "created_at": row.get(_CREATED_AT_FIELD, ""),
                        "metadata": {},
                    }
                doc_map[did]["chunk_count"] += 1

            return list(doc_map.values())

        except Exception as e:
            logger.error(f"Milvus 列出文档失败: {e}")
            return []

    async def get_stats(self) -> dict[str, Any]:
        def _stats():
            self._assert_client()
            desc = self._client.describe_collection(self.COLLECTION_NAME)
            row_count = desc.get("row_count", 0)

            rows = self._client.query(
                collection_name=self.COLLECTION_NAME,
                filter="",
                output_fields=[_DOC_ID_FIELD],
                limit=10000,
            )
            unique_docs = len(set(row.get(_DOC_ID_FIELD, "") for row in rows))

            return {
                "total_documents": unique_docs,
                "total_chunks": row_count,
                "index_size": row_count,
                "dimension": self.dimension,
                "index_type": self.index_type,
                "uri": self._uri,
                "last_updated": datetime.now().isoformat(),
            }

        try:
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(self._executor, _stats)
        except Exception as e:
            logger.error(f"Milvus 统计失败: {e}")
            return {
                "total_documents": 0,
                "total_chunks": 0,
                "index_size": 0,
                "dimension": self.dimension,
                "index_type": self.index_type,
                "error": str(e),
            }
