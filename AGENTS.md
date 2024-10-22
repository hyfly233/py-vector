# Repository Guidelines

## Project Overview

**py-vector** is a FastAPI-based vector similarity search API service. It ingests documents (PDF, DOCX, TXT, Excel, CSV, JSON, XML), chunks them, generates embeddings via Ollama (bge-m3), stores vectors in FAISS indexes, and serves search queries over the embedded content. Designed for RAG, multimodal search, and recommendation workloads.

## Architecture & Data Flow

```
HTTP Request → Router → Endpoint Handler → Service Layer → Core Layer → FAISS / Ollama
```

**Layered architecture** (all under `src/py_vector/`):

| Layer | Directory | Responsibility |
|---|---|---|
| **API** | `api/v1/endpoints/` | Route handlers, request validation, HTTP status codes |
| **Services** | `services/` | Business logic: document lifecycle, search orchestration, caching |
| **Core** | `core/` | Domain primitives: embedding, vector store, document parsing, FAISS persistence |
| **Models** | `models/` | Pydantic schemas for request/response shape |
| **Utils** | `utils/` | Cross-cutting helpers (response formatting) |

**Data flow**:
1. Document upload → `DocumentService` saves file → `DocumentProcessor` extracts text & chunks
2. Chunks → `EmbeddingService` (Ollama HTTP API) → float vectors
3. Vectors → `VectorStore` (FAISS index, persisted to disk)
4. Search query → same embedding → FAISS similarity search → ranked results

**Lifespan** (`main.py:AppLifespan`): Startup initializes embedding service, vector store, document service, search service; shutdown cleans them in reverse order. All services follow init/cleanup lifecycle.

## Key Directories

| Path | Contents |
|---|---|
| `src/py_vector/` | Package root |
| `src/py_vector/api/v1/endpoints/` | `search.py`, `documents.py`, `health.py` — FastAPI route handlers |
| `src/py_vector/core/` | `embedding.py`, `vector_store.py`, `search_engine.py`, `document_processor.py`, `faiss_persistence.py` |
| `src/py_vector/services/` | `document_service.py` (18.9KB), `search_service.py` (30.1KB) |
| `src/py_vector/models/` | `requests.py`, `responses.py` — Pydantic schemas |
| `src/py_vector/utils/` | `response_helper.py` — `ResponseHelper` singleton for uniform JSON responses |
| `tests/` | `test_document_processor.py`, `test_document_processor_extract.py`, `test_embedding.py` |
| `deploy/` | `Dockerfile`, `k8s-deployment.yaml` |
| `scripts/` | `start.sh` — checks Ollama, pulls bge-m3, starts uvicorn |
| `examples/` | `improved_endpoints.py` — example of standard response patterns |
| `depends/` | Placeholder for docker/k8s dependency configs |

**Scaffold-only directories** (empty `__init__.py`): `rag/`, `multimodal_search/`, `recommendation_system/`, `vector_dbs/`, `models/__init__.py`.

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
- File I/O via `aiofiles`; HTTP calls via `aiohttp`.
- `ThreadPoolExecutor` used inside `EmbeddingService` for CPU-bound embedding requests (Ollama sync client fallback).

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

Followed by: `EmbeddingService`, `VectorStore`, `DocumentService`, `SearchService`. Services obtain peer services via these factories at init time.

### Dependency Injection
FastAPI routes use `Depends(get_search_engine)` for the core `SearchEngine`. Service-layer dependencies are resolved internally via `get_*()` calls (not wired through FastAPI DI).

### Error Handling
- Global exception handler in `main.py` catches all unhandled exceptions → 500 JSON response.
- Service methods wrap operations in try/except, log errors, return dict with `status: "error"` or re-raise.
- Endpoint handlers use `HTTPException` for client errors (4xx), blanket catch for 500s.
- `ResponseHelper` utility provides static methods: `success()`, `error()`, `not_found()`, `bad_request()`, `internal_error()`, `created()`, `accepted()`, `paginated()`.

### Configuration
- `Settings` class in `config.py` extends `pydantic_settings.BaseSettings`.
- Reads from `.env` file (see `.env.example` for keys).
- Case-sensitive env vars. Storage paths created at import time.

### Response Format
Two model families exist:
- `models/requests.py` — slim input models used in endpoint signatures.
- `models/responses.py` — full output schemas (`BaseResponse<T>`, `ErrorResponse`, `SearchResponse`, etc.) used with `ResponseHelper`.

`ResponseHelper` wraps all responses in a uniform `{"success": bool, "data": ..., "message": str, "request_id": str | None}` envelope.

## Important Files

| File | Purpose |
|---|---|
| `src/py_vector/main.py` | FastAPI app creation, CORS, middleware, lifespan, entry point |
| `src/py_vector/config.py` | `Settings` — all env-based configuration |
| `src/py_vector/dependencies.py` | FastAPI `Depends` providers (e.g., `get_search_engine`) |
| `src/py_vector/api/v1/api.py` | Top-level router aggregating all endpoint sub-routers |
| `src/py_vector/api/v1/endpoints/health.py` | Health check endpoints (basic, detailed, K8s probes) |
| `src/py_vector/core/embedding.py` | `EmbeddingService` — Ollama embedding client |
| `src/py_vector/core/vector_store.py` | `VectorStore` — FAISS index wrapper with CRUD |
| `src/py_vector/core/document_processor.py` | `DocumentProcessor` — multi-format text extraction + chunking |
| `src/py_vector/core/search_engine.py` | `SearchEngine` — legacy search wrapper composing embedding + FAISS |
| `src/py_vector/core/faiss_persistence.py` | `FAISSPersistence`, `IncrementalFAISS`, `ShardedFAISS` — disk persistence utils |
| `src/py_vector/services/document_service.py` | `DocumentService` — upload, process, list, delete, backup, rebuild |
| `src/py_vector/services/search_service.py` | `SearchService` — advanced search with filters, reranking, feedback, suggestions |
| `pyproject.toml` | Project metadata, dependencies, build config (hatchling) |

## Runtime/Tooling Preferences

- **Runtime**: CPython 3.13.
- **Package manager**: `uv` (not pip, not poetry). All commands use `uv run`.
- **Build system**: `hatchling` (configured in `pyproject.toml`).
- **Package index**: Tsinghua mirror (`https://pypi.tuna.tsinghua.edu.cn/simple`) as default.
- **Formatter/linter**: `ruff` only (no black, no isort, no flake8).
- **Embedding service**: Ollama running at `http://localhost:11434` with model `bge-m3` (1024-dim).
- **Storage**: Local filesystem under `./storage/` (indexes, documents, temp — gitignored).
- **Container**: Dockerfile builds from `python:3.12-slim` (note: 3.12 != runtime 3.13 — likely needs update).

## Testing & QA

- **Framework**: `pytest` + `pytest-asyncio`.
- **Dev dependencies**: `pytest`, `pytest-asyncio`, `httpx`, `anyio`.
- **Current coverage**: Low — `test_document_processor.py` covers chunking, file support, and mocks document processing. `test_embedding.py` is a stub (failing test). `test_document_processor_extract.py` covers text extraction helpers.
- **Pattern**: Tests use `pytest.fixture`, `unittest.mock.patch`, `tempfile.TemporaryDirectory`. Async tests use `@pytest.mark.asyncio`.
- **Test command**: `uv run pytest`.
- **No coverage threshold configured**; no CI config present yet.
