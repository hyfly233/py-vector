FROM python:3.12-slim

# ... todo uv ...

# 健康检查
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/health/ || exit 1

EXPOSE 8000

CMD ["uvicorn", "py_rag.main:app", "--host", "0.0.0.0", "--port", "8000"]