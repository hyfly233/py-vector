"""语义分块策略（预留）

后续实现思路：
1. 将文本切为候选片段（段落/句子）
2. 用 EmbeddingService 生成每个候选片段的向量
3. 计算相邻片段的余弦相似度
4. 在相似度突变处（主题切换点）分块
5. 合并相似片段直到达到 chunk_size

依赖：EmbeddingService（需在工厂方法中注入）
"""

import logging

from py_vector.core.chunking.base import Chunker

logger = logging.getLogger(__name__)


class SemanticChunker(Chunker):
    """语义分块——根据主题变化边界分割（预留）"""

    def chunk(self, text: str, chunk_size: int = 0, overlap: int = 0) -> list[str]:
        raise NotImplementedError(
            "SemanticChunker 尚未实现。"
            "后续接入 EmbeddingService 后根据向量相似度确定分割点。"
        )
