import logging
import time

from fastapi import APIRouter, HTTPException

from py_vector.agent.models.rag import RAGQuery, RAGResponse
from py_vector.agent.rag import get_rag_agent, get_rag_deps

logger = logging.getLogger(__name__)
router = APIRouter()


@router.post(
    path="/ask",
    summary="RAG 问答",
    response_model=RAGResponse,
)
async def ask_rag(request: RAGQuery):
    """基于文档库内容进行问答。

    流程：
    1. 接收用户问题
    2. Agent 调用 search_docs 工具搜索相关文档片段
    3. LLM 根据检索结果生成带引用的回答
    """
    start = time.time()

    try:
        agent = get_rag_agent()
        deps = await get_rag_deps()

        result = await agent.run(
            user_prompt=request.query,
            deps=deps,
        )

        answer = result.output  # AnswerWithCitations 实例
        processing_time = time.time() - start

        return RAGResponse(
            query=request.query,
            answer=answer.answer,
            sources=[s.model_dump() for s in answer.sources],
            confidence=answer.confidence,
            processing_time=round(processing_time, 3),
        )

    except Exception as e:
        logger.error(f"RAG 问答失败: {e}")
        processing_time = time.time() - start

        # Agent 调用失败时降级返回纯检索结果
        try:
            from py_vector.services.search_service import SearchOptions

            deps = await get_rag_deps()
            results = await deps.search_service.search(
                query=request.query,
                options=SearchOptions(top_k=request.top_k),
            )
            sources = results.get("results", [])
            return RAGResponse(
                query=request.query,
                answer="抱歉，生成回答时出现错误。以下是检索到的相关文档片段，请自行查阅。",
                sources=sources,
                confidence="low",
                processing_time=round(processing_time, 3),
                error=str(e),
            )
        except Exception as fallback_e:
            raise HTTPException(
                status_code=500,
                detail=f"RAG 问答失败（降级检索也失败）: {fallback_e}",
            )
