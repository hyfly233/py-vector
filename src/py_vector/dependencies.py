from py_vector.services.document_service import get_document_service
from py_vector.services.search_service import get_search_service
from py_vector.vector_dbs.vector_store import get_vector_store


async def get_vector_store_dep():
    """获取向量存储实例（依赖注入用）"""
    return await get_vector_store()


async def get_search_service_dep():
    """获取搜索服务实例（依赖注入用）"""
    return await get_search_service()


async def get_document_service_dep():
    """获取文档服务实例（依赖注入用）"""
    return await get_document_service()
