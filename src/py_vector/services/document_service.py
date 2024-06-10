import asyncio
import hashlib
import logging
import shutil
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

from py_vector.config import settings
from py_vector.core.document_processor import document_processor
from py_vector.core.embedding import get_embedding_service
from py_vector.vector_dbs.vector_store import Document, get_vector_store

logger = logging.getLogger(__name__)


class DocumentService:
    """文档服务 - 提供完整的文档管理功能"""

    def __init__(self):
        """初始化文档服务

        Args:
            无

        Returns:
            None
        """
        self.document_processor = document_processor
        self.embedding_service = None
        self.vector_store = None

        # 文档状态
        self.processing_status: dict[str, dict[str, Any]] = {}

    async def initialize(self):
        """初始化服务

        初始化嵌入向量服务和向量存储后端。

        Args:
            无

        Returns:
            None
        """
        try:
            self.embedding_service = await get_embedding_service()
            self.vector_store = await get_vector_store()
            logger.info("文档服务初始化完成")
        except Exception as e:
            logger.error(f"文档服务初始化失败: {e}")
            raise

    async def upload_and_process_document(
        self,
        file_content: bytes,
        filename: str,
        user_id: str | None = None,
        metadata: dict[str, Any] | None = None,
        index: bool = True,
    ) -> dict[str, Any]:
        """
        上传并处理文档

        Args:
            file_content: 文件内容（二进制数据）
            filename: 文件名
            user_id: 用户ID
            metadata: 额外元数据
            index: 是否存入向量库。False 时仅保存文件，跳过嵌入和索引

        Returns:
            返回包含处理结果的字典，关键字段：
            - doc_id: 文档ID
            - status: 处理状态（"processing" 处理中 / "completed" 完成 / "error" 失败）
            - message: 状态描述信息
            - filename: 文件名
            - indexed: 是否已索引（仅 index=False 时返回）
            - error: 错误信息（仅 status 为 "error" 时返回）
        """
        # 生成文档ID
        doc_id = str(uuid.uuid4())

        try:
            # 保存临时文件
            temp_file_path = await self.document_processor.save_temp_file(
                file_content, filename
            )

            # 特殊保存原始文件（即使不索引也保留）
            file_storage_path = Path(settings.DOCUMENTS_PATH) / f"{doc_id}_{filename}"
            file_storage_path.parent.mkdir(parents=True, exist_ok=True)
            # 复制临时文件到永久存储
            shutil.copy2(temp_file_path, file_storage_path)

            if not index:
                # 仅保存，不向量化
                self.processing_status[doc_id] = {
                    "status": "completed",
                    "filename": filename,
                    "user_id": user_id,
                    "started_at": datetime.now().isoformat(),
                    "completed_at": datetime.now().isoformat(),
                    "progress": 100,
                    "message": "文件已保存（未索引）",
                    "indexed": False,
                }
                logger.info(f"✅ 文件已保存（未索引）: {doc_id} ({filename})")
                return {
                    "doc_id": doc_id,
                    "status": "completed",
                    "message": "文件上传成功（未存入向量库）",
                    "filename": filename,
                    "indexed": False,
                }

            # 初始化处理状态
            self.processing_status[doc_id] = {
                "status": "processing",
                "filename": filename,
                "user_id": user_id,
                "started_at": datetime.now().isoformat(),
                "progress": 0,
                "message": "开始处理文档...",
            }

            # 异步处理文档
            asyncio.create_task(
                self._process_document_async(doc_id, temp_file_path, metadata)
            )

            return {
                "doc_id": doc_id,
                "status": "processing",
                "message": "文档上传成功，正在处理中...",
                "filename": filename,
            }

        except Exception as e:
            logger.error(f"上传文档失败: {e}")
            self.processing_status[doc_id] = {
                "status": "error",
                "error": str(e),
                "filename": filename,
                "failed_at": datetime.now().isoformat(),
            }
            return {
                "doc_id": doc_id,
                "status": "error",
                "error": str(e),
                "message": "文档上传失败",
            }

    async def _process_document_async(
        self, doc_id: str, file_path: Path, metadata: dict[str, Any] | None = None
    ):
        """异步处理文档

        在后台执行文档文本提取、分块、生成嵌入向量并存入向量存储。

        Args:
            doc_id: 文档ID
            file_path: 临时文件路径
            metadata: 额外元数据

        Returns:
            None（结果通过 processing_status 跟踪）
        """
        try:
            # 更新状态：文本提取
            self._update_processing_status(doc_id, 10, "正在提取文档文本...")

            # 处理文档
            process_result = await self.document_processor.process_document(file_path)

            if process_result["status"] != "success":
                raise Exception(
                    f"文档处理失败: {process_result.get('error', 'Unknown error')}"
                )

            chunks = process_result["chunks"]

            # 更新状态：生成嵌入向量
            self._update_processing_status(
                doc_id, 30, f"正在生成嵌入向量... ({len(chunks)} 个文本块)"
            )

            # 生成嵌入向量
            embeddings = await self.embedding_service.get_embeddings_batch(
                chunks, show_progress=False
            )

            # 更新状态：创建文档对象
            self._update_processing_status(doc_id, 70, "正在创建文档索引...")

            # 创建文档对象
            documents = []
            file_hash = hashlib.md5(str(file_path).encode()).hexdigest()

            for i, chunk in enumerate(chunks):
                doc_metadata = {
                    "file_size": process_result["file_size"],
                    "processing_time": process_result["processing_time"],
                    "document_hash": process_result["document_hash"],
                    "file_hash": file_hash,
                    "chunk_length": len(chunk),
                    **(metadata or {}),
                }

                doc = Document(
                    doc_id=doc_id,
                    file_path=str(file_path),
                    file_name=process_result["file_name"],
                    chunk_index=i,
                    text=chunk,
                    metadata=doc_metadata,
                )
                documents.append(doc)

            # 更新状态：添加到向量存储
            self._update_processing_status(doc_id, 90, "正在添加到向量数据库...")

            # 添加到向量存储
            success = await self.vector_store.add_documents(documents, embeddings)

            if not success:
                raise Exception("添加到向量存储失败")

            # 完成处理
            self.processing_status[doc_id] = {
                "status": "completed",
                "filename": process_result["file_name"],
                "started_at": self.processing_status[doc_id]["started_at"],
                "completed_at": datetime.now().isoformat(),
                "progress": 100,
                "message": "文档处理完成",
                "chunks_count": len(chunks),
                "file_size": process_result["file_size"],
                "document_hash": process_result["document_hash"],
            }

            logger.info(f"✅ 文档处理完成: {doc_id} ({process_result['file_name']})")

        except Exception as e:
            logger.error(f"文档处理失败 {doc_id}: {e}")
            self.processing_status[doc_id] = {
                "status": "error",
                "error": str(e),
                "filename": self.processing_status[doc_id].get("filename", "Unknown"),
                "started_at": self.processing_status[doc_id]["started_at"],
                "failed_at": datetime.now().isoformat(),
                "message": f"文档处理失败: {str(e)}",
            }
        finally:
            # 清理临时文件
            try:
                if file_path.exists():
                    file_path.unlink()
            except Exception as e:
                logger.warning(f"清理临时文件失败: {e}")

    def _update_processing_status(self, doc_id: str, progress: int, message: str):
        """更新处理状态

        更新指定文档的处理进度和状态信息。

        Args:
            doc_id: 文档ID
            progress: 处理进度（0-100）
            message: 状态描述信息

        Returns:
            None
        """
        if doc_id in self.processing_status:
            self.processing_status[doc_id].update(
                {
                    "progress": progress,
                    "message": message,
                    "updated_at": datetime.now().isoformat(),
                }
            )

    async def get_processing_status(self, doc_id: str) -> dict[str, Any] | None:
        """获取文档处理状态

        Args:
            doc_id: 文档ID

        Returns:
            返回处理状态字典，关键字段：
            - status: 状态（"processing"/"completed"/"error"）
            - progress: 处理进度（0-100）
            - message: 状态描述
            - started_at: 开始时间
            - completed_at: 完成时间（仅已完成时返回）
            - failed_at: 失败时间（仅失败时返回）
            - error: 错误信息（仅失败时返回）
            - filename: 文件名
            - chunks_count: 分块数量（仅已完成时返回）
            - file_size: 文件大小（仅已完成时返回）
            - document_hash: 文档哈希（仅已完成时返回）
            如果文档不存在则返回 None
        """
        return self.processing_status.get(doc_id)

    async def search_documents(
        self,
        query: str,
        top_k: int = 10,
        filter_doc_ids: list[str] | None = None,
        min_score: float = 0.1,
    ) -> dict[str, Any]:
        """
        搜索文档

        Args:
            query: 搜索查询文本
            top_k: 返回结果数量
            filter_doc_ids: 过滤特定文档ID列表
            min_score: 最小相似度分数（低于该分数的结果将被过滤）

        Returns:
            返回包含搜索结果的字典，关键字段：
            - query: 原始查询文本
            - results: 搜索结果列表，每项包含文档详情和高亮文本
            - total_results: 结果总数
            - search_time: 搜索耗时（秒）
            - error: 错误信息（搜索失败时返回）
            - timestamp: 时间戳
        """
        try:
            start_time = datetime.now()

            # 生成查询向量
            query_embedding = await self.embedding_service.get_embedding(query)

            # 执行搜索
            search_results = await self.vector_store.search(
                query_embedding=query_embedding,
                top_k=top_k,
                filter_doc_ids=filter_doc_ids,
                min_score=min_score,
            )

            # 处理搜索结果
            results = []
            for result in search_results:
                result_dict = result.to_dict()

                # 添加高亮信息
                result_dict["highlighted_text"] = self._highlight_text(
                    result.document.text, query
                )

                results.append(result_dict)

            # 计算搜索时间
            search_time = (datetime.now() - start_time).total_seconds()

            return {
                "query": query,
                "results": results,
                "total_results": len(results),
                "search_time": search_time,
                "timestamp": datetime.now().isoformat(),
            }

        except Exception as e:
            logger.error(f"搜索失败: {e}")
            return {
                "query": query,
                "results": [],
                "total_results": 0,
                "error": str(e),
                "timestamp": datetime.now().isoformat(),
            }

    def _highlight_text(self, text: str, query: str, max_length: int = 200) -> str:
        """高亮搜索关键词

        在文本中标记搜索关键词，并返回包含关键词高亮的摘要片段。

        Args:
            text: 原始文本
            query: 搜索查询（用于提取关键词）
            max_length: 返回文本的最大长度

        Returns:
            包含高亮标记（**关键词**）的文本摘要片段，前后可能带有省略号
        """
        try:
            # 简单的关键词高亮
            query_words = query.lower().split()
            text_lower = text.lower()

            # 找到第一个匹配的位置
            first_match_pos = len(text)
            for word in query_words:
                pos = text_lower.find(word)
                if pos != -1:
                    first_match_pos = min(first_match_pos, pos)

            # 如果没有找到匹配，返回前段文本
            if first_match_pos == len(text):
                return text[:max_length] + ("..." if len(text) > max_length else "")

            # 计算摘要范围
            start = max(0, first_match_pos - max_length // 3)
            end = min(len(text), start + max_length)

            highlighted_text = text[start:end]

            # 高亮关键词
            for word in query_words:
                if len(word) > 1:  # 忽略单字符
                    highlighted_text = highlighted_text.replace(word, f"**{word}**")
                    highlighted_text = highlighted_text.replace(
                        word.capitalize(), f"**{word.capitalize()}**"
                    )

            # 添加省略号
            prefix = "..." if start > 0 else ""
            suffix = "..." if end < len(text) else ""

            return prefix + highlighted_text + suffix

        except Exception:
            # 如果高亮失败，返回原始文本片段
            return text[:max_length] + ("..." if len(text) > max_length else "")

    async def get_document_details(self, doc_id: str) -> dict[str, Any] | None:
        """获取文档详细信息

        Args:
            doc_id: 文档ID

        Returns:
            返回文档详情字典，关键字段：
            - doc_id: 文档ID
            - file_name: 文件名
            - file_path: 文件路径
            - chunks_count: 分块数量
            - created_at: 创建时间
            - metadata: 文档元数据
            - chunks: 文本块列表，每项包含 chunk_index（块索引）、text（文本内容）、
              text_length（文本长度）
            如果文档不存在则返回 None
        """
        try:
            # 从向量存储获取文档
            document_chunks = await self.vector_store.get_document(doc_id)

            if not document_chunks:
                return None

            # 组织文档信息
            first_chunk = document_chunks[0]

            return {
                "doc_id": doc_id,
                "file_name": first_chunk.file_name,
                "file_path": first_chunk.file_path,
                "chunks_count": len(document_chunks),
                "created_at": first_chunk.created_at,
                "metadata": first_chunk.metadata,
                "chunks": [
                    {
                        "chunk_index": chunk.chunk_index,
                        "text": chunk.text,
                        "text_length": len(chunk.text),
                    }
                    for chunk in document_chunks
                ],
            }

        except Exception as e:
            logger.error(f"获取文档详情失败 {doc_id}: {e}")
            return None

    async def delete_document(self, doc_id: str) -> dict[str, Any]:
        """删除文档

        从向量存储中删除指定文档及其所有分块，并清理处理状态。

        Args:
            doc_id: 要删除的文档ID

        Returns:
            返回删除结果字典，关键字段：
            - doc_id: 文档ID
            - status: 删除状态（"deleted" 成功 / "error" 失败）
            - message: 状态描述信息
            - deleted_at: 删除时间（仅删除成功时返回）
            - error: 错误信息（仅删除失败时返回）
        """
        try:
            # 从向量存储删除
            success = await self.vector_store.delete_document(doc_id)

            if success:
                # 清理处理状态
                if doc_id in self.processing_status:
                    del self.processing_status[doc_id]

                return {
                    "doc_id": doc_id,
                    "status": "deleted",
                    "message": "文档删除成功",
                    "deleted_at": datetime.now().isoformat(),
                }
            else:
                return {
                    "doc_id": doc_id,
                    "status": "error",
                    "error": "文档删除失败",
                    "message": "文档不存在或删除失败",
                }

        except Exception as e:
            logger.error(f"删除文档失败 {doc_id}: {e}")
            return {
                "doc_id": doc_id,
                "status": "error",
                "error": str(e),
                "message": "文档删除失败",
            }

    async def list_documents(
        self, page: int = 1, page_size: int = 20, include_deleted: bool = False
    ) -> dict[str, Any]:
        """列出文档

        分页获取文档列表，可选择是否包含已删除的文档。

        Args:
            page: 页码（从 1 开始）
            page_size: 每页文档数量
            include_deleted: 是否包含已删除的文档

        Returns:
            返回文档列表和分页信息字典，关键字段：
            - documents: 文档列表，每项包含文档基本信息
            - pagination: 分页信息，包含 page（当前页码）、page_size（每页数量）、
              total（总数）、total_pages（总页数）
            - error: 错误信息（获取失败时返回）
            - timestamp: 时间戳
        """
        try:
            # 从向量存储获取文档列表
            all_documents = await self.vector_store.list_documents(include_deleted)

            # 分页
            total = len(all_documents)
            start = (page - 1) * page_size
            end = start + page_size
            documents = all_documents[start:end]

            # 添加处理状态信息
            for doc in documents:
                doc_id = doc["doc_id"]
                if doc_id in self.processing_status:
                    doc["processing_status"] = self.processing_status[doc_id]

            return {
                "documents": documents,
                "pagination": {
                    "page": page,
                    "page_size": page_size,
                    "total": total,
                    "total_pages": (total + page_size - 1) // page_size,
                },
                "timestamp": datetime.now().isoformat(),
            }

        except Exception as e:
            logger.error(f"获取文档列表失败: {e}")
            return {
                "documents": [],
                "pagination": {
                    "page": page,
                    "page_size": page_size,
                    "total": 0,
                    "total_pages": 0,
                },
                "error": str(e),
                "timestamp": datetime.now().isoformat(),
            }

    async def get_statistics(self) -> dict[str, Any]:
        """获取统计信息

        获取向量存储统计、处理状态统计、支持的文档格式等信息。

        Args:
            无

        Returns:
            返回统计信息字典，关键字段：
            - vector_store: 向量存储统计信息
            - processing: 处理状态统计，包含 processing（处理中数量）、
              completed（已完成数量）、error（失败数量）
            - supported_formats: 支持的文档格式列表
            - error: 错误信息（获取失败时返回）
            - timestamp: 时间戳
        """
        try:
            # 向量存储统计
            vector_stats = await self.vector_store.get_stats()

            # 处理状态统计
            processing_stats = {"processing": 0, "completed": 0, "error": 0}

            for status_info in self.processing_status.values():
                status = status_info.get("status", "unknown")
                if status in processing_stats:
                    processing_stats[status] += 1

            return {
                "vector_store": vector_stats,
                "processing": processing_stats,
                "supported_formats": self.document_processor.get_supported_types(),
                "timestamp": datetime.now().isoformat(),
            }

        except Exception as e:
            logger.error(f"获取统计信息失败: {e}")
            return {"error": str(e), "timestamp": datetime.now().isoformat()}

    async def rebuild_index(self) -> dict[str, Any]:
        """重建索引

        重新构建向量存储的索引。

        Args:
            无

        Returns:
            返回重建结果字典，关键字段：
            - status: 状态（"success" 成功 / "error" 失败）
            - message: 状态描述信息
            - processing_time: 处理耗时（秒）
            - error: 错误信息（仅失败时返回）
            - timestamp: 时间戳
        """
        try:
            logger.info("开始重建文档索引...")
            start_time = datetime.now()

            success = await self.vector_store.rebuild_index()

            processing_time = (datetime.now() - start_time).total_seconds()

            if success:
                return {
                    "status": "success",
                    "message": "索引重建完成",
                    "processing_time": processing_time,
                    "timestamp": datetime.now().isoformat(),
                }
            else:
                return {
                    "status": "error",
                    "message": "索引重建失败",
                    "processing_time": processing_time,
                    "timestamp": datetime.now().isoformat(),
                }

        except Exception as e:
            logger.error(f"重建索引失败: {e}")
            return {
                "status": "error",
                "error": str(e),
                "message": "索引重建失败",
                "timestamp": datetime.now().isoformat(),
            }

    async def backup_data(self, backup_path: str) -> dict[str, Any]:
        """备份数据

        将向量存储数据备份到指定路径。

        Args:
            backup_path: 备份目标路径

        Returns:
            返回备份结果字典，关键字段：
            - status: 状态（"success" 成功 / "error" 失败）
            - message: 状态描述信息
            - backup_path: 备份路径（仅成功时返回）
            - error: 错误信息（仅失败时返回）
            - timestamp: 时间戳
        """
        try:
            success = await self.vector_store.backup_index(backup_path)

            if success:
                return {
                    "status": "success",
                    "message": "数据备份完成",
                    "backup_path": backup_path,
                    "timestamp": datetime.now().isoformat(),
                }
            else:
                return {
                    "status": "error",
                    "message": "数据备份失败",
                    "timestamp": datetime.now().isoformat(),
                }

        except Exception as e:
            logger.error(f"备份数据失败: {e}")
            return {
                "status": "error",
                "error": str(e),
                "message": "数据备份失败",
                "timestamp": datetime.now().isoformat(),
            }

    async def cleanup_temp_files(self):
        """清理临时文件

        清理文档处理过程中产生的临时文件。

        Args:
            无

        Returns:
            None
        """
        try:
            await self.document_processor.cleanup_temp_files()
            logger.info("临时文件清理完成")
        except Exception as e:
            logger.error(f"清理临时文件失败: {e}")

    async def cleanup(self):
        """清理服务资源

        清理临时文件和处理状态，释放服务资源。

        Args:
            无

        Returns:
            None
        """
        try:
            # 清理临时文件
            await self.cleanup_temp_files()

            # 清理处理状态
            self.processing_status.clear()

            logger.info("文档服务清理完成")
        except Exception as e:
            logger.error(f"文档服务清理失败: {e}")


# 全局实例
_document_service: DocumentService | None = None


async def get_document_service() -> DocumentService:
    """获取全局文档服务实例

    获取或创建全局唯一的 DocumentService 单例实例。

    Args:
        无

    Returns:
        DocumentService: 已初始化的文档服务实例
    """
    global _document_service

    if _document_service is None:
        _document_service = DocumentService()
        # 初始化文档服务
        await _document_service.initialize()

    return _document_service


async def cleanup_document_service():
    """清理全局文档服务

    清理并销毁全局 DocumentService 实例，释放资源。

    Args:
        无

    Returns:
        None
    """
    global _document_service

    if _document_service:
        await _document_service.cleanup()
        _document_service = None
