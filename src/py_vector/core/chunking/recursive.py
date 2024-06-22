"""递归分块策略

按段落（\\n\\n）→ 行（\\n）→ 句子（. ! ?）→ 字符的优先级降级分割，
尽可能保持语义完整。每一级只切超大的块，小块保留。
"""

import logging
import re

from py_vector.config import settings
from py_vector.core.chunking.base import Chunker

logger = logging.getLogger(__name__)


class RecursiveChunker(Chunker):
    """递归分块

    按以下优先级降级分割：
    1. 段落（连续两个换行）
    2. 行（单个换行）
    3. 句子（中英文句号、感叹号、问号）
    4. 字符（兜底）
    """

    # 分割层级：每级是一个 (名称, 正则)
    _SEPARATORS = [
        ("paragraph", re.compile(r"\n\n+")),
        ("line", re.compile(r"\n")),
        ("sentence", re.compile(r"(?<=[.!?。！？])\s+")),
    ]

    def chunk(self, text: str, chunk_size: int = 0, overlap: int = 0) -> list[str]:
        chunk_size = chunk_size or settings.CHUNK_SIZE
        overlap = overlap or settings.CHUNK_OVERLAP

        if not text or not text.strip():
            return []

        chunks = self._recursive_split(text.strip(), chunk_size, 0)
        return self._apply_overlap(chunks, overlap) if overlap > 0 else chunks

    def _recursive_split(
        self, text: str, chunk_size: int, level: int
    ) -> list[str]:
        """递归分割

        Args:
            text: 待分割文本
            chunk_size: 目标块大小
            level: 当前层级索引（0=段落, 1=行, 2=句子）

        Returns:
            文本块列表
        """
        # 如果文本已经足够短，直接返回
        if len(text) <= chunk_size:
            return [text]

        # 如果已经到最后一层（字符级），按字符硬切
        if level >= len(self._SEPARATORS):
            return self._split_by_chars(text, chunk_size)

        # 使用当前层级的分隔符分割
        separator = self._SEPARATORS[level]
        parts = [
            p.strip() for p in separator[1].split(text) if p.strip()
        ]

        # 逐段合并到 chunk_size
        merged = self._merge_parts(parts, chunk_size)

        # 仍有超过 chunk_size 的块 → 递归到下一级
        result = []
        for chunk in merged:
            if len(chunk) > chunk_size:
                result.extend(self._recursive_split(chunk, chunk_size, level + 1))
            else:
                result.append(chunk)
        return result

    @staticmethod
    def _merge_parts(parts: list[str], chunk_size: int) -> list[str]:
        """将片段合并成不超过 chunk_size 的块"""
        merged = []
        current = ""
        for part in parts:
            if not current:
                current = part
            elif len(current) + 1 + len(part) <= chunk_size:
                current += " " + part
            else:
                merged.append(current)
                current = part
        if current:
            merged.append(current)
        return merged

    @staticmethod
    def _split_by_chars(text: str, chunk_size: int) -> list[str]:
        """按字符长度硬切（兜底）"""
        return [
            text[i : i + chunk_size].strip()
            for i in range(0, len(text), chunk_size)
            if text[i : i + chunk_size].strip()
        ]

    @staticmethod
    def _apply_overlap(chunks: list[str], overlap: int) -> list[str]:
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
