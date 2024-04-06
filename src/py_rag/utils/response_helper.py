from typing import Any, Optional, TypeVar, Generic
from datetime import datetime
from fastapi import Response
from fastapi.responses import JSONResponse
import uuid

from py_faiss.models.responses import BaseResponse, ErrorResponse

T = TypeVar('T')


class ResponseHelper:
    """响应助手类 - 提供统一的响应格式"""
    
    @staticmethod
    def success(
        data: Any = None,
        message: str = "操作成功",
        status_code: int = 200,
        request_id: Optional[str] = None
    ) -> JSONResponse:
        """返回成功响应"""
        response_data = BaseResponse[Any](
            success=True,
            message=message,
            data=data,
            timestamp=datetime.now().isoformat(),
            request_id=request_id or str(uuid.uuid4())[:8]
        )
        
        return JSONResponse(
            content=response_data.model_dump(),
            status_code=status_code
        )
    
    @staticmethod
    def error(
        message: str,
        error_code: Optional[str] = None,
        details: Any = None,
        status_code: int = 400,
        request_id: Optional[str] = None
    ) -> JSONResponse:
        """返回错误响应"""
        response_data = ErrorResponse(
            success=False,
            message=message,
            error_code=error_code,
            details=details,
            timestamp=datetime.now().isoformat()
        )
        
        return JSONResponse(
            content=response_data.model_dump(),
            status_code=status_code
        )
    
    @staticmethod
    def paginated(
        items: list,
        total: int,
        page: int,
        page_size: int,
        message: str = "获取成功"
    ) -> JSONResponse:
        """返回分页响应"""
        total_pages = (total + page_size - 1) // page_size
        
        data = {
            "items": items,
            "pagination": {
                "total": total,
                "page": page,
                "page_size": page_size,
                "total_pages": total_pages,
                "has_next": page < total_pages,
                "has_prev": page > 1
            }
        }
        
        return ResponseHelper.success(data=data, message=message)
    
    @staticmethod
    def not_found(message: str = "资源不存在") -> JSONResponse:
        """返回404响应"""
        return ResponseHelper.error(
            message=message,
            error_code="NOT_FOUND",
            status_code=404
        )
    
    @staticmethod
    def bad_request(message: str = "请求参数错误") -> JSONResponse:
        """返回400响应"""
        return ResponseHelper.error(
            message=message,
            error_code="BAD_REQUEST",
            status_code=400
        )
    
    @staticmethod
    def internal_error(message: str = "服务器内部错误") -> JSONResponse:
        """返回500响应"""
        return ResponseHelper.error(
            message=message,
            error_code="INTERNAL_ERROR",
            status_code=500
        )
    
    @staticmethod
    def created(
        data: Any = None,
        message: str = "创建成功"
    ) -> JSONResponse:
        """返回201响应"""
        return ResponseHelper.success(
            data=data,
            message=message,
            status_code=201
        )
    
    @staticmethod
    def accepted(
        data: Any = None,
        message: str = "请求已接受"
    ) -> JSONResponse:
        """返回202响应（异步处理）"""
        return ResponseHelper.success(
            data=data,
            message=message,
            status_code=202
        )
