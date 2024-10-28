import asyncio
import logging
import time

import numpy as np
from openai import APIError, AsyncOpenAI, OpenAI

from py_vector.config import settings

logger = logging.getLogger(__name__)


class EmbeddingService:
    """Embedding 服务——通过 OpenAI 兼容 API 生成文本嵌入向量"""

    def __init__(
        self,
        base_url: str = None,
        model_name: str = None,
        dimension: int = None,
        api_key: str = None,
        timeout: int = 30,
        max_retries: int = 3,
    ):
        """初始化 embedding 服务"""
        self.base_url = base_url or settings.EMBEDDING_BASE_URL
        self.model_name = model_name or settings.EMBEDDING_MODEL
        self.dimension = dimension or settings.EMBEDDING_DIMENSION
        self.api_key = api_key or settings.EMBEDDING_API_KEY
        self.timeout = timeout
        self.max_retries = max_retries

        # OpenAI 客户端（延迟初始化）
        self.client: AsyncOpenAI | None = None
        self._sync_client: OpenAI | None = None

    async def initialize(self):
        """初始化 embedding 服务"""
        try:
            # 创建异步 OpenAI 客户端
            self.client = AsyncOpenAI(
                base_url=self.base_url,
                api_key=self.api_key,
                timeout=self.timeout,
                max_retries=max(0, self.max_retries - 1),
            )

            # 测试 embedding 生成（验证连接和模型可用性）
            await self._test_embedding()

            logger.info(
                f"✅ Embedding service initialized with model: {self.model_name}"
            )
            return True

        except Exception as e:
            logger.error(f"❌ Failed to initialize embedding service: {e}")
            if self.client:
                await self.client.close()
                self.client = None
            raise

    async def _test_embedding(self):
        """测试 embedding 生成"""
        try:
            test_text = "测试文本"
            embedding = await self.get_embedding(test_text)

            if len(embedding) != self.dimension:
                logger.warning(
                    f"Expected dimension {self.dimension}, got {len(embedding)}"
                )
                # 更新实际维度
                self.dimension = len(embedding)

            logger.info(f"Embedding test successful, dimension: {self.dimension}")

        except Exception as e:
            raise Exception(f"Embedding test failed: {e}")

    async def get_embedding(self, text: str) -> np.ndarray:
        """
        获取单个文本的 embedding

        Args:
            text: 输入文本

        Returns:
            embedding 向量
        """
        if not self.client:
            raise Exception("Embedding service not initialized")

        if not text or not text.strip():
            raise ValueError("Text cannot be empty")

        for retry_num in range(self.max_retries):
            try:
                response = await self.client.embeddings.create(
                    model=self.model_name,
                    input=text.strip(),
                )

                embedding = np.array(response.data[0].embedding, dtype=np.float32)

                # 验证维度
                if len(embedding) != self.dimension:
                    embedding = self._adjust_dimension(embedding)

                return embedding

            except APIError as e:
                logger.warning(f"❌ Embedding retry {retry_num + 1} failed: {e}")
                if retry_num < self.max_retries - 1:
                    wait_time = (2**retry_num) * 0.5  # 指数退避
                    await asyncio.sleep(wait_time)
                else:
                    raise Exception(
                        "Failed to get embedding after"
                        f" {self.max_retries} attempts: {e}"
                    )
            except Exception as e:
                logger.warning(f"❌ Embedding retry {retry_num + 1} failed: {e}")
                if retry_num < self.max_retries - 1:
                    wait_time = (2**retry_num) * 0.5
                    await asyncio.sleep(wait_time)
                else:
                    raise

        return np.zeros(self.dimension, dtype=np.float32)

    async def get_embeddings_batch(
        self, texts: list[str], batch_size: int = 10, show_progress: bool = True
    ) -> np.ndarray:
        """
        批量获取文本的 embedding

        Args:
            texts: 文本列表
            batch_size: 每批发送的文本数
            show_progress: 是否显示进度

        Returns:
            embedding 向量数组
        """
        if not texts:
            return np.array([])

        # 过滤空文本
        valid_texts = [text.strip() for text in texts if text and text.strip()]
        if not valid_texts:
            raise ValueError("No valid texts provided")

        embeddings = []
        total_batches = (len(valid_texts) + batch_size - 1) // batch_size

        for i in range(0, len(valid_texts), batch_size):
            batch = valid_texts[i : i + batch_size]
            batch_num = i // batch_size + 1

            if show_progress:
                logger.info(
                    f"Processing batch {batch_num}/{total_batches} ({len(batch)} texts)"
                )

            try:
                # OpenAI API 原生支持批量输入，一次请求发送整个 batch
                response = await self.client.embeddings.create(
                    model=self.model_name,
                    input=batch,
                )

                # response.data 按输入索引排序
                batch_embeddings = [
                    np.array(item.embedding, dtype=np.float32) for item in response.data
                ]
                embeddings.extend(batch_embeddings)

                # 批次间小延迟，避免过载
                if batch_num < total_batches:
                    await asyncio.sleep(0.1)

            except APIError as e:
                logger.error(f"Failed to process batch {batch_num}: {e}")
                for _ in batch:
                    embeddings.append(np.zeros(self.dimension, dtype=np.float32))

        return np.array(embeddings)

    def get_embedding_sync(self, text: str) -> np.ndarray:
        """
        同步获取 embedding（用于非异步环境）

        Args:
            text: 输入文本

        Returns:
            embedding 向量
        """
        if not text or not text.strip():
            raise ValueError("Text cannot be empty")

        # 延迟创建同步客户端
        if self._sync_client is None:
            self._sync_client = OpenAI(
                base_url=self.base_url,
                api_key=self.api_key,
                timeout=self.timeout,
                max_retries=max(0, self.max_retries - 1),
            )

        for attempt in range(self.max_retries):
            try:
                response = self._sync_client.embeddings.create(
                    model=self.model_name,
                    input=text.strip(),
                )

                embedding = np.array(response.data[0].embedding, dtype=np.float32)

                if len(embedding) != self.dimension:
                    embedding = self._adjust_dimension(embedding)

                return embedding

            except APIError as e:
                logger.warning(f"Sync embedding attempt {attempt + 1} failed: {e}")
                if attempt < self.max_retries - 1:
                    time.sleep((2**attempt) * 0.5)
                else:
                    raise Exception(
                        f"Sync embedding failed after {self.max_retries} attempts: {e}"
                    )
            except Exception as e:
                logger.warning(f"Sync embedding attempt {attempt + 1} failed: {e}")
                if attempt < self.max_retries - 1:
                    time.sleep((2**attempt) * 0.5)
                else:
                    raise

        return np.zeros(self.dimension, dtype=np.float32)

    def _adjust_dimension(self, embedding: np.ndarray) -> np.ndarray:
        """调整向量维度"""
        current_dim = len(embedding)

        if current_dim > self.dimension:
            # 截断
            return embedding[: self.dimension]
        elif current_dim < self.dimension:
            # 填充零值
            padded = np.zeros(self.dimension, dtype=np.float32)
            padded[:current_dim] = embedding
            return padded
        else:
            return embedding

    async def get_service_info(self) -> dict:
        """获取服务信息"""
        try:
            models_response = await self.client.models.list()
            return {
                "status": "healthy",
                "base_url": self.base_url,
                "model": self.model_name,
                "dimension": self.dimension,
                "available_models": [m.id for m in models_response.data],
            }
        except Exception as e:
            return {
                "status": "unhealthy",
                "error": str(e),
                "base_url": self.base_url,
                "model": self.model_name,
            }

    async def cleanup(self):
        """清理资源"""
        if self.client:
            await self.client.close()
            self.client = None

        if self._sync_client:
            self._sync_client.close()
            self._sync_client = None

        logger.info("Embedding service cleaned up")


# 全局实例（可选）
_embedding_service: EmbeddingService | None = None


async def get_embedding_service() -> EmbeddingService:
    """获取全局 embedding 服务实例"""
    global _embedding_service

    if _embedding_service is None:
        _embedding_service = EmbeddingService()
        # 初始化服务
        await _embedding_service.initialize()

    return _embedding_service


async def cleanup_embedding_service():
    """清理全局 embedding 服务"""
    global _embedding_service

    if _embedding_service:
        await _embedding_service.cleanup()
        _embedding_service = None
