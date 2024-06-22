"""文档结构分块策略（预留）

后续实现思路：
1. 识别文档的层级结构（Markdown 标题、PDF 章节书签、HTML heading）
2. 以标题/章节为单位分割
3. 保留层级关系（section 1.1 属于 section 1）
4. 大章节内部降级到段落分块

适用场景：技术文档、Markdown 笔记、有明确章节结构的 PDF
"""

import logging

from py_vector.core.chunking.base import Chunker

logger = logging.getLogger(__name__)


class StructureChunker(Chunker):
    """文档结构分块——按标题/章节层级分割（预留）"""

    def chunk(self, text: str, chunk_size: int = 0, overlap: int = 0) -> list[str]:
        raise NotImplementedError(
            "StructureChunker 尚未实现。"
            "后续需结合文档结构提取（标题、章节、列表）确定分割点。"
        )
