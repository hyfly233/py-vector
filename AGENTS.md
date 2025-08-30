# Repository Guidelines

## Project Overview

**py-vector** is a FastAPI-based vector similarity search API service. It ingests documents (PDF, DOCX, TXT, Excel, CSV,
JSON, XML), chunks them, generates embeddings via Ollama (bge-m3), stores vectors in FAISS or Milvus indexes, and serves
search queries over the embedded content. Designed for RAG, multimodal search, and recommendation workloads. Includes a
pydantic-ai Agent for LLM-powered Q&A.

## Architecture & Data Flow

```
HTTP Request → Router → Endpoint Handler → Service Layer → Core Layer → Vector Store / Ollama / S3 / PostgreSQL
```

**Layered architecture** (all under `src/py_vector/`):

| Layer            | Directory                           | Responsibility                                                                 |
|------------------|-------------------------------------|--------------------------------------------------------------------------------|
| **API**          | `api/v1/endpoints/`                 | Route handlers, request validation, HTTP status codes                          |
| **Services**     | `services/`                         | Business logic: document lifecycle, search orchestration, caching              |
| **Core**         | `core/`                             | Infrastructure: embedding, document parsing, S3, database, chunking strategies |
| **Vector Store** | `vector_dbs/vector_store.py`        | Abstract `VectorStore` class + factory `get_vector_store()`                    |
| **FAISS**        | `vector_dbs/faiss_vector_store.py`  | `FAISSVectorStore(VectorStore)` — default backend                              |
| **Milvus**       | `vector_dbs/milvus_vector_store.py` | `MilvusVectorStore(VectorStore)` — optional backend                            |
| **Models**       | `models/`                           | Pydantic schemas for request/response shape + SQLAlchemy ORM                   |
| **Utils**        | `utils/`                            | Cross-cutting helpers (response formatting)                                    |

**Data flow**:

1. Document upload → `DocumentService` saves to S3+PG → `DocumentProcessor` extracts text → chunking strategies split
   text → `EmbeddingService` generates vectors → `VectorStore` indexes them
2. Search query → same embedding → `VectorStore` similarity search → optional reranking → ranked results
3. RAG Q&A → pydantic-ai Agent with `search_docs` tool → LLM-generated answer (falls back to plain retrieval)

**Lifespan** (`main.py:AppLifespan`): Startup initializes embedding service, vector store, S3 bucket, database; shutdown
disposes DB, cleans up embedding service + vector store. All services follow init/cleanup lifecycle.

## Key Directories

| Path                              | Contents                                                                                                      |
|-----------------------------------|---------------------------------------------------------------------------------------------------------------|
| `src/py_vector/`                  | Package root                                                                                                  |
| `src/py_vector/api/v1/endpoints/` | `search.py`, `documents.py`, `files.py`, `rag.py`, `health.py` — FastAPI route handlers                       |
| `src/py_vector/core/`             | `embedding.py`, `document_processor.py`, `reranker.py`, `database.py`, `s3.py`, `file_store.py`, `chunking/`  |
| `src/py_vector/core/chunking/`    | Strategy pattern: `base.py` (ABC), `fixed_size.py`, `recursive.py`, `semantic.py`, `agent.py`, `structure.py` |
| `src/py_vector/vector_dbs/`       | `vector_store.py` (ABC), `faiss_vector_store.py`, `milvus_vector_store.py`, `faiss_persistence.py`            |
| `src/py_vector/services/`         | `document_service.py`, `search_service.py`                                                                    |
| `src/py_vector/models/`           | `requests.py`, `responses.py`, `file.py` — Pydantic + SQLAlchemy                                              |
| `src/py_vector/utils/`            | `response_helper.py` — `ResponseHelper` singleton for uniform JSON responses                                  |
| `src/py_vector/agent/`            | `rag.py` (RAG Agent), `tools/search.py`, `models/rag.py` — pydantic-ai Agent                                  |
| `tests/`                          | `test_document_processor.py`, `test_document_processor_extract.py`, `test_embedding.py`                       |
| `deploy/`                         | `Dockerfile`, `k8s-deployment.yaml`                                                                           |
| `scripts/`                        | `start.sh` — checks Ollama, pulls bge-m3, starts uvicorn                                                      |
| `examples/`                       | `improved_endpoints.py` — example of standard response patterns                                               |
| `depends/`                        | Placeholder for docker/k8s dependency configs                                                                 |

## Development Commands

```bash
# Run the API server
uv run uvicorn py_vector.main:app --host 0.0.0.0 --port 8000 --reload

# Run via start script (checks Ollama first)
bash scripts/start.sh

# Run tests
uv run pytest

# Run tests with verbose output
uv run pytest -v

# Lint
uv run ruff check .

# Format
uv run ruff format .
```

`uv` is the package manager (replaces pip/poetry). `uv.lock` pins all dependencies.

## Code Conventions & Common Patterns

### Language & Style
- **Python 3.13** required (`.python-version`).
- **Chinese docstrings** and comments throughout the codebase.
- **ruff** for linting and formatting (`.ruff_cache/` present).
- Pydantic `BaseModel` / `BaseSettings` for all data schemas and config.

### Async-First
- All I/O-bound code is `async def`: HTTP handlers, service methods, core operations.
- File I/O via `aiofiles`; HTTP calls via `aiohttp` / `httpx`.
- `run_in_executor` used for CPU/blocking calls: FAISS operations, boto3 S3 calls, document parsing (PDF/DOCX/Excel).

### Singleton Services (lazy-init pattern)
```python
# Global instance at module level
_service: Optional[SomeService] = None

# Async factory called once during lifespan startup
async def get_service() -> SomeService:
    global _service
    if _service is None:
        _service = SomeService()
        await _service.initialize()
    return _service

# Cleanup during shutdown
async def cleanup_service():
    global _service
    if _service is not None:
        await _service.cleanup()
        _service = None
```

Followed by: `EmbeddingService`, `VectorStore` (abstract factory dispatching to FAISS/Milvus), `DocumentService`,
`SearchService`, `Database` (asyncpg/SQLAlchemy). Services obtain peer services via these factories at init time.

### Dependency Injection

FastAPI routes mostly call service factories directly (`await get_document_service()`) rather than using FastAPI
`Depends()`. Only `files.py` uses `Depends(get_session)` for SQLAlchemy sessions. A `dependencies.py` module provides
`Depends()` wrappers for vector store / search service / document service for routes that prefer DI.

### Strategy Pattern

- **VectorStore**: Abstract base with `FAISSVectorStore` and `MilvusVectorStore` implementations. Selected via
  `VECTOR_STORE_TYPE` config.
- **Chunking**: Abstract `Chunker` base with `FixedSizeChunker`, `RecursiveChunker`, `SemanticChunker`, `AgentChunker`,
  `StructureChunker`. Default is recursive. Selected via config.

### Error Handling
- Global exception handler in `main.py` catches all unhandled exceptions → 500 JSON response.
- Service methods wrap operations in try/except, log errors, return dict with `status: "error"` or re-raise.
- Endpoint handlers use `HTTPException` for client errors (4xx), blanket catch for 500s.
- `ResponseHelper` utility provides static methods: `success()`, `error()`, `not_found()`, `bad_request()`, `internal_error()`, `created()`, `accepted()`, `paginated()`.

### Configuration

- `Settings` class in `config.py` extends `pydantic_settings.BaseSettings` with 80+ knobs.
- Reads from `.env` file. Case-sensitive env vars.
- Storage directories (`./storage/`) created eagerly at import time.

### Response Format
Two model families exist:
- `models/requests.py` — slim input models used in endpoint signatures.
- `models/responses.py` — full output schemas (`BaseResponse<T>`, `ErrorResponse`, `SearchResponse`, etc.) with
  `Generic[T]` type parameter.

`ResponseHelper` wraps all responses in a uniform `{"success": bool, "data": ..., "message": str, "request_id": str | None}` envelope.

## Important Files

| File                                              | Purpose                                                               |
|---------------------------------------------------|-----------------------------------------------------------------------|
| `src/py_vector/main.py`                           | FastAPI app creation, CORS, middleware, lifespan, entry point         |
| `src/py_vector/config.py`                         | `Settings` — 80+ env-based configuration knobs                        |
| `src/py_vector/dependencies.py`                   | FastAPI `Depends` wrappers for services                               |
| `src/py_vector/api/v1/api_routers.py`             | Top-level router aggregating all 5 endpoint sub-routers               |
| `src/py_vector/api/v1/endpoints/health.py`        | Health check endpoints + `HealthChecker` singleton + K8s probes       |
| `src/py_vector/api/v1/endpoints/search.py`        | Search endpoints (basic, advanced, suggestions, statistics, cache)    |
| `src/py_vector/api/v1/endpoints/documents.py`     | Document CRUD endpoints                                               |
| `src/py_vector/api/v1/endpoints/files.py`         | File storage endpoints (S3 + PG metadata)                             |
| `src/py_vector/api/v1/endpoints/rag.py`           | RAG Q&A endpoint using pydantic-ai Agent                              |
| `src/py_vector/core/embedding.py`                 | `EmbeddingService` — async OpenAI-compatible embedding generation     |
| `src/py_vector/core/document_processor.py`        | `DocumentProcessor` — multi-format text extraction                    |
| `src/py_vector/core/chunking/`                    | Chunking strategy implementations (fixed, recursive, semantic, etc.)  |
| `src/py_vector/core/reranker.py`                  | Two-stage reranking (model-based + heuristic)                         |
| `src/py_vector/core/database.py`                  | Async SQLAlchemy session lifecycle                                    |
| `src/py_vector/core/s3.py`                        | S3 client factory and bucket operations                               |
| `src/py_vector/core/file_store.py`                | S3 upload + PG metadata orchestration                                 |
| `src/py_vector/vector_dbs/vector_store.py`        | Abstract `VectorStore` base + factory                                 |
| `src/py_vector/vector_dbs/faiss_vector_store.py`  | FAISS backend implementation                                          |
| `src/py_vector/vector_dbs/milvus_vector_store.py` | Milvus backend implementation                                         |
| `src/py_vector/services/document_service.py`      | `DocumentService` — upload → extract → embed → index pipeline         |
| `src/py_vector/services/search_service.py`        | `SearchService` — multi-strategy search, reranking, cache             |
| `src/py_vector/models/responses.py`               | Generic `BaseResponse[T]` + domain-specific response schemas          |
| `src/py_vector/utils/response_helper.py`          | `ResponseHelper` — uniform JSON response builder                      |
| `pyproject.toml`                                  | Project metadata, dependencies, build config (hatchling), ruff config |

## Runtime/Tooling Preferences

- **Runtime**: CPython 3.13.
- **Package manager**: `uv` (not pip, not poetry). All commands use `uv run`.
- **Build system**: `hatchling` (configured in `pyproject.toml`).
- **Package index**: Tsinghua mirror (`https://pypi.tuna.tsinghua.edu.cn/simple`) as default.
- **Formatter/linter**: `ruff` only (no black, no isort, no flake8).
- **Embedding service**: OpenAI-compatible API at `EmbeddingService.base_url` (default `http://localhost:11434/v1`) with model configurable via `EMBEDDING_MODEL` (default `bge-m3`, 1024-dim). Works with Ollama, OpenAI, Azure, etc.
- **Vector stores**: FAISS (default, `faiss-cpu`) or Milvus (`pymilvus`), selected via `VECTOR_STORE_TYPE`.
- **Storage**: Local filesystem under `./storage/` (indexes, documents, temp — gitignored) or S3-compatible (MinIO).
- **Database**: PostgreSQL via `asyncpg`/`SQLAlchemy` for file metadata (optional, gated by config).
- **Container**: Dockerfile builds from `python:3.12-slim` (note: 3.12 != runtime 3.13 — needs update).

## Testing & QA

- **Framework**: `pytest` 9.x + `pytest-asyncio`.
- **Dev dependencies**: `pytest`, `pytest-asyncio`, `httpx`, `anyio`.
- **Current coverage**: Very low. `test_document_processor.py` (540 lines) is the primary test file but is *
  *significantly out of sync** with the current source — references deleted private methods (`_chunk_text`,
  `_process_pdf_file`, etc.) and mocks PyPDF2 instead of the current `pymupdf`/`fitz`. Several tests are commented out.
  `test_document_processor_extract.py` (~130 lines) is a newer version that better matches the current API but shares
  the same class name causing pytest collection conflicts. `test_embedding.py` is a stub (always fails).
- **Zero test coverage** for: `EmbeddingService`, `Chunker` base + strategies, PDF/DOCX/Excel/JSON/XML extraction, S3
  integration, `SearchService`, `Reranker`, `VectorStore` backends, RAG agent, full pipeline.
- **No `conftest.py`** — fixtures are duplicated inline across test files.
- **Pattern**: Tests use `pytest.fixture`, `unittest.mock.patch`, `MagicMock`, `tempfile.TemporaryDirectory`. Async
  tests use `@pytest.mark.asyncio`.
- **Test command**: `uv run pytest`.
- **No coverage threshold configured**; no CI config present yet.
