import logging
import os
import pickle
from datetime import datetime
from typing import List, Dict, Any

import faiss

from py_faiss.config import settings
from py_faiss.core.document_processor import DocumentProcessor
from py_faiss.core.embedding import EmbeddingService
from py_faiss.models.requests import SearchResult

logger = logging.getLogger(__name__)


class SearchEngine:
    """搜索引擎类，负责文档索引和搜索功能"""

    def __init__(self):
        """初始化搜索引擎"""
        # embedding服务，用于生成文本嵌入向量
        self.embedding_service = EmbeddingService()
        # 文档处理器，用于提取和分割文档内容
        self.document_processor = DocumentProcessor()
        # FAISS索引实例
        self.index = None
        # 存储文档和文本块信息
        self.documents = []
        self.chunks = []
        # 索引和元数据文件路径
        self.index_file = os.path.join(settings.INDEX_PATH, "faiss_index.bin")
        self.metadata_file = os.path.join(settings.INDEX_PATH, "metadata.pkl")

    async def initialize(self):
        """初始化搜索引擎"""
        await self.embedding_service.initialize()

        # 创建或加载索引
        if os.path.exists(self.index_file) and os.path.exists(self.metadata_file):
            await self.load_index()
        else:
            self.index = faiss.IndexFlatIP(settings.EMBEDDING_DIMENSION)

    async def add_document(self, file_path: str, document_id: str) -> Dict[str, Any]:
        """添加文档到索引"""
        try:
            # 处理文档
            text_content = await self.document_processor.extract_text(file_path)
            if not text_content:
                return {"status": "error", "message": "无法提取文档内容"}

            # 分割文本
            chunks = self.document_processor.split_text(text_content)

            # 生成嵌入向量
            embeddings = await self.embedding_service.get_embeddings_batch(chunks)

            # 标准化向量
            faiss.normalize_L2(embeddings)

            # 添加到索引
            self.index.add(embeddings.astype('float32'))

            # 存储文档信息
            filename = os.path.basename(file_path)
            for i, chunk in enumerate(chunks):
                self.documents.append({
                    'document_id': document_id,
                    'file_path': file_path,
                    'file_name': filename,
                    'chunk_index': i,
                    'text': chunk,
                    'created_at': datetime.now().isoformat()
                })
                self.chunks.append(chunk)

            # 保存索引
            await self.save_index()

            return {
                "status": "success",
                "chunks_count": len(chunks),
                "message": f"成功添加 {len(chunks)} 个文本块"
            }

        except Exception as e:
            return {"status": "error", "message": f"处理文档失败: {str(e)}"}

    async def search(self, query: str, top_k: int = 5) -> List[SearchResult]:
        """搜索文档"""
        if self.index.ntotal == 0:
            return []

        # 生成查询向量
        query_embedding = await self.embedding_service.get_embedding(query)
        query_embedding = query_embedding.reshape(1, -1)
        faiss.normalize_L2(query_embedding)

        # 搜索
        scores, indices = self.index.search(
            query_embedding.astype('float32'),
            min(top_k, self.index.ntotal)
        )

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx >= 0:
                doc_info = self.documents[idx]
                results.append(SearchResult(
                    score=float(score),
                    file_name=doc_info['file_name'],
                    file_path=doc_info['file_path'],
                    chunk_index=doc_info['chunk_index'],
                    text=doc_info['text']
                ))

        return results

    async def save_index(self):
        """保存索引"""
        try:
            # 保存 FAISS 索引
            faiss.write_index(self.index, self.index_file)

            # 保存元数据
            metadata = {
                'documents': self.documents,
                'chunks': self.chunks,
                'dimension': settings.EMBEDDING_DIMENSION,
                'model_name': settings.EMBEDDING_MODEL,
                'saved_at': datetime.now().isoformat(),
                'version': settings.VERSION
            }

            with open(self.metadata_file, 'wb') as f:
                pickle.dump(metadata, f)

            logger.info(f"索引保存成功，包含 {len(self.documents)} 个文档块")

        except Exception as e:
            raise Exception(f"保存索引失败: {str(e)}")

    async def load_index(self) -> bool:
        """加载索引"""
        try:
            # 加载 FAISS 索引
            self.index = faiss.read_index(self.index_file)

            # 加载元数据
            with open(self.metadata_file, 'rb') as f:
                metadata = pickle.load(f)

            # 兼容性处理：处理不同版本的元数据格式
            self.documents = metadata.get('documents', [])
            self.chunks = metadata.get('chunks', [])

            # 如果 chunks 不存在，从 documents 中重建
            if not self.chunks and self.documents:
                self.chunks = [doc.get('text', '') for doc in self.documents]
                logger.info("从文档信息重建 chunks 列表")

            # 验证数据一致性
            if len(self.documents) != self.index.ntotal:
                logger.warning(f"文档数量 ({len(self.documents)}) 与索引大小 ({self.index.ntotal}) 不匹配")

            # 检查维度兼容性
            expected_dim = settings.EMBEDDING_DIMENSION
            if 'dimension' in metadata and metadata['dimension'] != expected_dim:
                logger.warning(f"索引维度不匹配: {metadata['dimension']} vs {expected_dim}")

            logger.info(f"✅ 索引加载成功，包含 {len(self.documents)} 个文档块")
            return True

        except Exception as e:
            logger.error(f"❌ 加载索引失败: {e}")
            # 清理可能部分加载的数据
            self.index = None
            self.documents = []
            self.chunks = []
            return False

    async def get_stats(self) -> Dict[str, Any]:
        """获取搜索引擎统计信息"""
        return {
            "total_documents": len(set(doc['document_id'] for doc in self.documents)) if self.documents else 0,
            "total_chunks": len(self.documents),
            "index_size": self.index.ntotal if self.index else 0,
            "embedding_model": settings.EMBEDDING_MODEL,
            "index_exists": self.index is not None
        }

    async def cleanup(self):
        """清理资源"""
        await self.embedding_service.cleanup()

    async def reset_index(self):
        """重置索引（删除所有数据）"""
        try:
            self.index = faiss.IndexFlatIP(settings.EMBEDDING_DIMENSION)
            self.documents = []
            self.chunks = []

            # 删除索引文件
            if os.path.exists(self.index_file):
                os.remove(self.index_file)
            if os.path.exists(self.metadata_file):
                os.remove(self.metadata_file)

            logger.info("索引已重置")

        except Exception as e:
            logger.error(f"重置索引失败: {e}")
            raise Exception(f"重置索引失败: {str(e)}")
