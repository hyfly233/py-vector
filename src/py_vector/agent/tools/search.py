import logging
from typing import Any

from pydantic_ai import RunContext

logger = logging.getLogger(__name__)


class RAGDeps:
    """RAG Agent 依赖"""

    def __init__(
        self,
        search_service: Any,
        top_k: int = 10,
        min_score: float = 0.1,
    ):
        self.search_service = search_service
        self.top_k = top_k
        self.min_score = min_score


async def search_docs(
    ctx: RunContext[RAGDeps],
    query: str,
    top_k: int = 10,
) -> str:
    """搜索文档库，返回最相关的文本块

    当需要查找信息来回答用户问题时使用这个工具。
    会返回匹配的文档片段及其来源文件名和相似度分数。

    Args:
        ctx: 运行上下文（自动注入）
        query: 搜索查询语句
        top_k: 返回的文本块数量（默认 10）

    Returns:
        格式化的搜索结果文本
    """
    from py_vector.services.search_service import SearchOptions

    try:
        results_dict = await ctx.deps.search_service.search(
            query=query,
            options=SearchOptions(
                top_k=top_k,
                search_type="vector",
                enable_highlight=False,
                enable_summary=False,
                chunk_merge=False,
                diversity_threshold=0.7,
            ),
        )

        results = results_dict.get("results", [])
        if not results:
            return "未找到相关文档。"

        chunks_text = []
        for i, r in enumerate(results):
            file_name = r.get("file_name", "未知文件")
            score = r.get("max_score", 0)
            chunks_info = r.get("chunks", [])
            chunk_texts = [c.get("text", "") for c in chunks_info if c.get("text")]
            if not chunk_texts:
                continue
            text = chunk_texts[0]  # 使用第一段文本
            chunks_text.append(
                f"[来源 {i + 1}] {file_name}（相似度={score:.3f}）\n{text}"
            )

        if not chunks_text:
            return "未找到相关文档。"

        return "\n\n---\n\n".join(chunks_text)

    except Exception as e:
        logger.error(f"搜索工具调用失败: {e}")
        return f"搜索过程出现错误：{e}"
