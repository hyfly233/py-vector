import logging
from typing import Any

from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider

from py_vector.agent.models.rag import AnswerWithCitations
from py_vector.agent.tools.search import RAGDeps, search_docs
from py_vector.config import settings

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# 默认系统提示词
# ---------------------------------------------------------------------------

DEFAULT_SYSTEM_PROMPT = """你是一个基于文档的智能问答助手。你的工作流程如下：

1. **检索** — 使用 search_docs 工具在文档库中搜索与用户问题相关的内容
2. **阅读** — 仔细阅读检索到的文档片段
3. **回答** — 基于检索到的内容给出简洁准确的回答
4. **引用** — 在回答中引用信息来源 [文件名]

## 规则
- 如果文档中没有相关信息，明确告知用户，不要编造
- 如果信息不足以回答，可以追问更多细节
- 回答使用与用户相同的语言
- 相同来源的引用可以合并"""

# ---------------------------------------------------------------------------
# Agent + 全局工厂
# ---------------------------------------------------------------------------

_rag_agent: Agent[RAGDeps, AnswerWithCitations] | None = None
_rag_deps: RAGDeps | None = None

SYSTEM_PROMPT = getattr(settings, "LLM_SYSTEM_PROMPT", None) or DEFAULT_SYSTEM_PROMPT


def _build_model():
    """根据配置创建 OpenAI 兼容模型实例"""
    return OpenAIChatModel(
        model_name=settings.LLM_MODEL,
        provider=OpenAIProvider(
            base_url=settings.LLM_BASE_URL,
            api_key=settings.LLM_API_KEY,
        ),
    )


def _build_agent() -> Agent[RAGDeps, AnswerWithCitations]:
    """创建一个新的 RAG Agent 实例"""
    model = _build_model()
    agent = Agent[RAGDeps, AnswerWithCitations](
        model=model,
        system_prompt=SYSTEM_PROMPT,
        output_type=AnswerWithCitations,
        deps_type=RAGDeps,
    )
    # 显式注册外部工具函数，防止在单个文件膨胀
    agent.tool(search_docs)
    return agent


def get_rag_agent() -> Agent[RAGDeps, AnswerWithCitations]:
    """获取全局 RAG Agent 实例（单例，延迟初始化）"""
    global _rag_agent
    if _rag_agent is None:
        _rag_agent = _build_agent()
    assert _rag_agent is not None
    return _rag_agent


def get_rag_deps_sync(search_service: Any) -> RAGDeps:
    """同步获取 RAG 依赖（在已经拿到 search_service 的场景使用）"""
    global _rag_deps
    if _rag_deps is None:
        _rag_deps = RAGDeps(search_service=search_service)
    assert _rag_deps is not None
    return _rag_deps


async def get_rag_deps() -> RAGDeps:
    """异步获取 RAG 依赖（延迟初始化）"""
    global _rag_deps
    if _rag_deps is None:
        from py_vector.services.search_service import get_search_service

        search_service = await get_search_service()
        _rag_deps = RAGDeps(search_service=search_service)
    assert _rag_deps is not None  # 消除 Pyright 的 global 类型收窄限制
    return _rag_deps


async def cleanup_rag():
    """清理 RAG Agent 资源"""
    global _rag_agent, _rag_deps
    _rag_agent = None
    _rag_deps = None
    logger.info("RAG Agent 资源已清理")
