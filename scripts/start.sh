#!/bin/bash
set -e

echo "Starting Document Search API..."

# 启动应用
uv run uvicorn py_vector.main:app --host 0.0.0.0 --port 8000 --reload
