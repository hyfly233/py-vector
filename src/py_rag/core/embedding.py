import asyncio
import logging
import time
from concurrent.futures import ThreadPoolExecutor
from typing import List, Optional

import aiohttp
import numpy as np
import requests

from py_faiss.config import settings

logger = logging.getLogger(__name__)


class EmbeddingService:
    def __init__(self, base_url: str = None, model_name: str = None, dimension: int = None, timeout: int = 30,
                 max_retries: int = 3):
        """ 初始化 embedding 服务"""
        self.base_url = base_url or settings.OLLAMA_BASE_URL
        self.model_name = model_name or settings.EMBEDDING_MODEL
        self.dimension = dimension or settings.EMBEDDING_DIMENSION
        self.timeout = timeout
        self.max_retries = max_retries

        # 构建 API URL
        self.embeddings_url = f"{self.base_url}/api/embeddings"
        self.tags_url = f"{self.base_url}/api/tags"

        # 连接池配置
        self.session = None
        self._executor = ThreadPoolExecutor(max_workers=4)

    async def initialize(self):
        """初始化 embedding 服务"""
        try:
            # 创建 aiohttp 会话
            self.session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=self.timeout)
            )

            # 检查 Ollama 连接
            await self._check_ollama_connection()

            # 验证模型可用性
            await self._verify_model()

            # 测试 embedding 生成
            await self._test_embedding()

            logger.info(f"✅ Embedding service initialized with model: {self.model_name}")
            return True

        except Exception as e:
            logger.error(f"❌ Failed to initialize embedding service: {e}")
            if self.session:
                await self.session.close()
                self.session = None
            raise

    async def _check_ollama_connection(self):
        """测试与 Ollama 的连接"""
        try:
            # 获取 tags 列表以验证连接
            async with self.session.get(self.tags_url) as response:
                if response.status == 200:
                    logger.info("✅ Ollama connection successful")
                else:
                    raise Exception(f"Ollama returned status {response.status}")
        except Exception as e:
            raise Exception(f"Cannot connect to Ollama at {self.base_url}: {e}")

    async def _verify_model(self):
        """验证模型是否可用"""
        try:
            async with self.session.get(self.tags_url) as response:
                if response.status == 200:
                    # 获取模型列表
                    data = await response.json()
                    models = data.get('models', [])
                    model_names = [model['name'] for model in models]

                    # 检查模型是否存在（支持 model:latest 格式）
                    if self.model_name in model_names or f"{self.model_name}:latest" in model_names:
                        logger.info(f"✅ Model {self.model_name} is available")
                    else:
                        available_models = ", ".join(model_names)
                        raise Exception(
                            f"Model {self.model_name} not found. "
                            f"Available models: {available_models}"
                        )
                else:
                    raise Exception("Cannot retrieve model list")
        except Exception as e:
            raise Exception(f"Model verification failed: {e}")

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
        if not self.session:
            raise Exception("Embedding service not initialized")

        if not text or not text.strip():
            raise ValueError("Text cannot be empty")

        for retry_num in range(self.max_retries):
            try:
                payload = {
                    "model": self.model_name,
                    "prompt": text.strip()
                }

                # 异步发送请求 embedding
                async with self.session.post(self.embeddings_url, json=payload) as response:
                    if response.status == 200:
                        result = await response.json()
                        embedding = np.array(result['embedding'], dtype=np.float32)

                        # 验证维度
                        if len(embedding) != self.dimension:
                            # 调整维度
                            embedding = self._adjust_dimension(embedding)

                        return embedding
                    else:
                        error_text = await response.text()
                        raise Exception(f"HTTP {response.status}: {error_text}")

            except Exception as e:
                logger.warning(f"❌ Embedding retry_num {retry_num + 1} failed: {e}")
                if retry_num < self.max_retries - 1:
                    wait_time = (2 ** retry_num) * 0.5  # 指数退避
                    await asyncio.sleep(wait_time)
                else:
                    raise Exception(f"Failed to get embedding after {self.max_retries} attempts: {e}")

        return np.zeros(self.dimension, dtype=np.float32)

    async def get_embeddings_batch(self, texts: List[str], batch_size: int = 10,
                                   show_progress: bool = True) -> np.ndarray:
        """
        批量获取文本的 embedding

        Args:
            texts: 文本列表
            batch_size: 批处理大小
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
            batch = valid_texts[i:i + batch_size]
            batch_num = i // batch_size + 1

            if show_progress:
                logger.info(f"Processing batch {batch_num}/{total_batches} ({len(batch)} texts)")

            try:
                # 并发处理批次中的文本
                batch_embeddings = await self._process_batch_concurrent(batch)
                embeddings.extend(batch_embeddings)

                # 批次间延迟，避免过载
                if batch_num < total_batches:
                    await asyncio.sleep(0.1)

            except Exception as e:
                logger.error(f"Failed to process batch {batch_num}: {e}")
                # 使用零向量作为后备
                for _ in batch:
                    embeddings.append(np.zeros(self.dimension, dtype=np.float32))

        return np.array(embeddings)

    async def _process_batch_concurrent(self, texts: List[str]) -> List[np.ndarray]:
        """并发处理批次中的文本"""
        tasks = [self.get_embedding(text) for text in texts]

        try:
            embeddings = await asyncio.gather(*tasks, return_exceptions=True)

            results = []
            for i, embedding in enumerate(embeddings):
                if isinstance(embedding, Exception):
                    logger.warning(f"Failed to get embedding for text {i}: {embedding}")
                    results.append(np.zeros(self.dimension, dtype=np.float32))
                else:
                    results.append(embedding)

            return results

        except Exception as e:
            logger.error(f"Concurrent processing failed: {e}")
            # 回退到顺序处理
            return await self._process_batch_sequential(texts)

    async def _process_batch_sequential(self, texts: List[str]) -> List[np.ndarray]:
        """顺序处理批次中的文本（回退方案）"""
        embeddings = []
        for text in texts:
            try:
                embedding = await self.get_embedding(text)
                embeddings.append(embedding)
                await asyncio.sleep(0.05)  # 小延迟
            except Exception as e:
                logger.warning(f"Failed to get embedding: {e}")
                embeddings.append(np.zeros(self.dimension, dtype=np.float32))
        return embeddings

    def _adjust_dimension(self, embedding: np.ndarray) -> np.ndarray:
        """调整向量维度"""
        current_dim = len(embedding)

        if current_dim > self.dimension:
            # 截断
            return embedding[:self.dimension]
        elif current_dim < self.dimension:
            # 填充零值
            padded = np.zeros(self.dimension, dtype=np.float32)
            padded[:current_dim] = embedding
            return padded
        else:
            return embedding

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

        for attempt in range(self.max_retries):
            try:
                payload = {
                    "model": self.model_name,
                    "prompt": text.strip()
                }

                response = requests.post(
                    self.embeddings_url,
                    json=payload,
                    timeout=self.timeout
                )

                if response.status_code == 200:
                    result = response.json()
                    embedding = np.array(result['embedding'], dtype=np.float32)

                    if len(embedding) != self.dimension:
                        embedding = self._adjust_dimension(embedding)

                    return embedding
                else:
                    raise Exception(f"HTTP {response.status_code}: {response.text}")

            except Exception as e:
                logger.warning(f"Sync embedding attempt {attempt + 1} failed: {e}")
                if attempt < self.max_retries - 1:
                    time.sleep((2 ** attempt) * 0.5)
                else:
                    raise

    async def get_service_info(self) -> dict:
        """获取服务信息"""
        try:
            async with self.session.get(self.tags_url) as response:
                if response.status == 200:
                    data = await response.json()
                    return {
                        "status": "healthy",
                        "base_url": self.base_url,
                        "model": self.model_name,
                        "dimension": self.dimension,
                        "available_models": [model['name'] for model in data.get('models', [])]
                    }
        except Exception as e:
            return {
                "status": "unhealthy",
                "error": str(e),
                "base_url": self.base_url,
                "model": self.model_name
            }

    async def cleanup(self):
        """清理资源"""
        if self.session:
            await self.session.close()
            self.session = None

        if self._executor:
            self._executor.shutdown(wait=True)

        logger.info("Embedding service cleaned up")


# 全局实例（可选）
_embedding_service: Optional[EmbeddingService] = None


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
