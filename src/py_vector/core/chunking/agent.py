"""Agent / LLM 分块策略（预留）

后续实现思路：
1. 将文本分页或分段送入 LLM
2. Prompt 要求 LLM 识别语义段落边界
3. 返回结构化的分割点位置
4. 根据分割点切分文本

适用场景：
- 法律条文（条款边界明确，行文严谨）
- 财务报告（表格与文本混合，需要保留上下文）
- 需要高精度分块的垂直领域

注意：调用 LLM 有成本和延迟，建议仅在对分割精度要求高的场景启用。
"""

import logging

from py_vector.core.chunking.base import Chunker

logger = logging.getLogger(__name__)


class AgentChunker(Chunker):
    """Agent / LLM 分块——利用 AI 确定语义分割点（预留）"""

    def chunk(self, text: str, chunk_size: int = 0, overlap: int = 0) -> list[str]:
        raise NotImplementedError(
            "AgentChunker 尚未实现。"
            "后续通过 pydantic-ai Agent 调用 LLM 识别语义段落边界。"
            "参考: py_vector.agent 模块"
        )
