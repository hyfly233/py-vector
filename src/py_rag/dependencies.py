from fastapi import Request

from py_faiss.core.search_engine import SearchEngine


async def get_search_engine(request: Request) -> SearchEngine:
    """获取搜索引擎实例"""
    return request.app.state.search_engine
