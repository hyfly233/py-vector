"""重排序服务

支持多种重排序方式，按优先级：
1. **模型重排序** — 调用 Reranker API（OpenAI 兼容 /v1/rerank）
2. **启发式重排序** — 基于查询词覆盖率的回退方案（原有逻辑迁移至此）
"""

import logging

import httpx

from py_vector.config import settings

logger = logging.getLogger(__name__)


async def _model_rerank(query: str, texts: list[str], top_k: int = 10) -> list[float]:
    """调用 Reranker API 计算每条文本与查询的相关性分数

    通过 OpenAI 兼容的 rerank 端点（如 Jina / Cohere / 自定义服务）。
    POST /v1/rerank
    Body: {"model": "...", "query": "...", "documents": [...], "top_k": N}
    Returns: {"results": [{"index": 0, "relevance_score": 0.95}, ...]}
    """
    base_url = settings.RERANKER_BASE_URL or settings.EMBEDDING_BASE_URL
    api_key = settings.RERANKER_API_KEY or settings.EMBEDDING_API_KEY
    model = settings.RERANKER_MODEL

    url = f"{base_url.rstrip('/')}/rerank"
    payload = {
        "model": model,
        "query": query,
        "documents": texts,
        "top_k": top_k,
    }

    try:
        async with httpx.AsyncClient(timeout=60) as client:
            resp = await client.post(
                url,
                json=payload,
                headers={"Authorization": f"Bearer {api_key}"},
            )
            resp.raise_for_status()
            data = resp.json()

        results = data.get("results", [])
        scores = [0.0] * len(texts)
        for item in results:
            idx = item.get("index")
            score = item.get("relevance_score", item.get("score", 0.0))
            if idx is not None and 0 <= idx < len(scores):
                scores[idx] = score
        return scores

    except Exception as e:
        logger.warning("模型重排序失败（%s），回退到启发式", e)
        return []


async def _heuristic_rerank(query: str, texts: list[str]) -> list[float]:
    """基于查询词覆盖率的启发式重排序（回退方案）"""
    query_words = set(query.lower().split())
    if not query_words:
        return [0.0] * len(texts)

    scores = []
    for text in texts:
        text_words = set(text.lower().split())
        coverage = len(query_words & text_words) / len(query_words)
        scores.append(coverage)
    return scores


async def rerank(
    query: str,
    texts: list[str],
    initial_scores: list[float] | None = None,
    top_k: int = 10,
    weight_model: float = 0.5,
    weight_initial: float = 0.3,
    weight_heuristic: float = 0.2,
) -> list[tuple[int, float]]:
    """对文本列表进行重排序，返回 (原始索引, 最终分数) 的降序列表

    Args:
        query: 查询文本
        texts: 待排序的文本列表
        initial_scores: 初始分数（来自向量检索的相似度），None 时忽略
        top_k: 排序返回的数量
        weight_model: 模型重排序分数的权重
        weight_initial: 初始分数的权重
        weight_heuristic: 启发式分数的权重

    Returns:
        按最终分数降序排列的 [(index, score), ...]
    """
    n = len(texts)
    if n == 0:
        return []

    model_scores = []
    if settings.RERANKER_ENABLED:
        model_scores = await _model_rerank(query, texts, top_k)

    heuristic_scores = await _heuristic_rerank(query, texts)

    # 综合评分
    final = []
    for i in range(n):
        score = heuristic_scores[i] * weight_heuristic
        if model_scores and i < len(model_scores):
            score += model_scores[i] * weight_model
        if initial_scores and i < len(initial_scores):
            score += initial_scores[i] * weight_initial
        final.append((i, score))

    final.sort(key=lambda x: x[1], reverse=True)
    return final[:top_k]
