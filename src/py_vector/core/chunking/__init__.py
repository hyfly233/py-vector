"""分块策略工厂

通过 CHUNKING_STRATEGY 配置选择分块方案。
支持扩展，新增策略只需在 _STRATEGIES 字典中注册。
"""

import logging

from py_vector.config import settings
from py_vector.core.chunking.base import Chunker
from py_vector.core.chunking.fixed_size import FixedSizeChunker
from py_vector.core.chunking.recursive import RecursiveChunker

logger = logging.getLogger(__name__)

_STRATEGIES: dict[str, type[Chunker] | None] = {
    "fixed_size": FixedSizeChunker,
    "recursive": RecursiveChunker,
    "semantic": None,  # 预留
    "structure": None,  # 预留
    "agent": None,  # 预留
}


def create_chunker(strategy: str | None = None) -> Chunker:
    """根据策略名称创建分块器实例

    Args:
        strategy: 策略名，可选值：
                  fixed_size — 固定长度（按句子边界优先）
                  recursive — 递归分割（段落→行→句子→字符）
                  semantic — 语义分块（预留）
                  structure — 文档结构分块（预留）
                  agent — Agent/LLM 分块（预留）
                  传 None 时从 settings.CHUNKING_STRATEGY 读取

    Returns:
        Chunker 实例
    """
    name = (strategy or settings.CHUNKING_STRATEGY).lower()

    cls = _STRATEGIES.get(name)
    if cls is None:
        if name in _STRATEGIES:
            logger.warning("分块策略 '%s' 尚未实现，回退到 recursive", name)
        else:
            logger.warning("未知分块策略 '%s'，回退到 recursive", name)
        return RecursiveChunker()

    logger.info("使用分块策略: %s", name)
    return cls()
