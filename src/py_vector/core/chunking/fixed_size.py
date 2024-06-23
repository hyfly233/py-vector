"""固定长度分块策略（按句子边界优先）

从旧的 smart_split_text 迁移而来。
按句子分隔符（. ! ? 。！？ \\n\\n）标记边界，尽量在句子处分割；
超长句子退回到字符级硬切。
"""

import logging

from py_vector.config import settings
from py_vector.core.chunking.base import Chunker

logger = logging.getLogger(__name__)

# 句子分隔符
_SENTENCE_SEPARATORS = [".", "!", "?", "。", "！", "？", "\n\n"]
_SPLIT_TAG = "<SPLIT>"


class FixedSizeChunker(Chunker):
    """固定长度分块

    优先在句子边界分割，避免句子被切断。
    当句子超过 chunk_size 时退回到按字符分割。
    """

    def chunk(self, text: str, chunk_size: int = 0, overlap: int = 0) -> list[str]:
        chunk_size = chunk_size or settings.CHUNK_SIZE
        overlap = overlap or settings.CHUNK_OVERLAP

        if not text or not text.strip():
            return []

        # 标记句子边界
        marked = text
        for sep in _SENTENCE_SEPARATORS:
            marked = marked.replace(sep, sep + _SPLIT_TAG)
        parts = [p.strip() for p in marked.split(_SPLIT_TAG) if p.strip()]

        chunks = []
        current = ""

        for part in parts:
            # 当前块 + 新句子 ≤ chunk_size → 追加
            if len(current) + len(part) <= chunk_size:
                current += part
            else:
                if current.strip():
                    chunks.append(current.strip())

                if len(part) > chunk_size:
                    # 超长句子 → 按字符硬切
                    sub = self._split_by_chars(part, chunk_size, overlap)
                    chunks.extend(sub[:-1])
                    current = sub[-1] if sub else ""
                else:
                    current = part

        if current.strip():
            chunks.append(current.strip())

        return self._apply_overlap(chunks, overlap) if overlap > 0 else chunks

    def _split_by_chars(self, text: str, chunk_size: int, overlap: int) -> list[str]:
        """纯按字符长度分割（兜底方案）"""
        chunks = []
        start = 0
        while start < len(text):
            end = min(start + chunk_size, len(text))
            chunks.append(text[start:end].strip())
            start += chunk_size - overlap
        return [c for c in chunks if c]

    @staticmethod
    def _apply_overlap(chunks: list[str], overlap: int) -> list[str]:
        """在相邻块之间补重叠"""
        if len(chunks) <= 1:
            return chunks
        result = []
        for i, chunk in enumerate(chunks):
            if i == 0:
                result.append(chunk)
            else:
                prev = chunks[i - 1]
                overlap_text = prev[-overlap:] if len(prev) > overlap else prev
                result.append(overlap_text + " " + chunk)
        return result
