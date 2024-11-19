# py-vector 项目概览

## 项目定位

**py-vector** 是一个基于 **FastAPI** 的向量相似度搜索 API 服务。核心能力是把文档转化为向量，存入 FAISS 索引，然后提供语义搜索接口。专为 RAG（检索增强生成）、以图搜文、个性化推荐等需要语义匹配的场景设计。

## 一句话架构

> 上传文档 → 提取文本 → 切片 → 生成嵌入向量 → 存入 FAISS 索引 → 接受查询 → 向量搜索 → 返回排序结果

## 技术栈

| 组件 | 选型 |
|---|---|
| API 框架 | FastAPI（async-first） |
| 向量引擎 | FAISS（支持 IndexFlatIP / IndexFlatL2 / IndexIVFFlat / IndexHNSW） |
| 嵌入模型 | OpenAI 兼容 API（Ollama / OpenAI / Azure 等） |
| 嵌入维度 | 1024（bge-m3）或按模型自动检测 |
| OpenAPI 客户端 | `openai` Python 库 |
| 文档解析 | PyMuPDF（PDF）、python-docx（DOCX）、openpyxl / pandas（Excel/CSV）、chardet（编码检测） |
| 运行环境 | Python 3.13，uv 包管理 |
| 构建系统 | Hatchling |

## 数据流

```
HTTP 请求 → 路由 → 端点处理器 → 服务层 → 核心层 → FAISS / Ollama
```

### 文档处理链路

1. **上传** → `POST /api/v1/documents/upload`
2. **异步处理** → `DocumentService` 保存临时文件，启动后台任务
3. **文本提取** → `DocumentProcessor` 按文件类型调用提取器（PDF/DOCX/TXT/Excel/CSV/JSON/XML）
4. **切片** → 按配置的 `chunk_size`（512）和 `chunk_overlap`（50）分割，可选智能分割（按句子边界）
5. **向量化** → `EmbeddingService` 通过 Ollama HTTP API 生成 bge-m3 嵌入向量
6. **存储** → `VectorStore` 将向量加入 FAISS 索引并持久化到磁盘

### 搜索链路

1. **查询** → `POST /api/v1/search/` 或 `POST /api/v1/search/advanced`
2. **向量化** → 用同一模型将查询文本转为向量
3. **检索** → FAISS 内积相似度搜索（余弦相似度）
4. **后处理** → 可选：重排序、关键词高亮、摘要生成、块合并、多样性过滤
5. **返回** → 排序后的搜索结果（含文档 ID、文件名、匹配文本、相似度分数）

## 核心模块

```
src/py_vector/
├── api/v1/endpoints/     # 路由处理器（search.py, documents.py, health.py）
├── vector_dbs/            # 向量存储实现
│   ├── vector_store.py    # VectorStore 抽象接口 + 工厂函数
│   ├── faiss_vector_store.py  # FAISS 后端
│   ├── faiss_persistence.py   # FAISSPersistence / IncrementalFAISS / ShardedFAISS
│   └── milvus_vector_store.py # Milvus 后端
├── core/                 # 领域核心
│   ├── embedding.py      # EmbeddingService — OpenAI 兼容嵌入客户端
│   ├── document_processor.py  # DocumentProcessor — 多格式文档提取和切片
│   └── search_engine.py  # SearchEngine — 旧版搜索封装
├── services/             # 业务逻辑
│   ├── document_service.py    # 文档上传、异步处理、查询、删除、备份、重建索引
│   └── search_service.py      # 高级搜索（向量/混合/关键词）、重排序、缓存、历史
├── models/               # Pydantic 请求/响应模型
├── utils/                # ResponseHelper — 统一 JSON 响应格式
├── config.py             # 环境配置（Settings）
└── main.py               # FastAPI 应用入口 + 生命周期管理
```

### core/embedding.py

`EmbeddingService` 通过 **OpenAI 兼容 API**（`/v1/embeddings`）生成嵌入向量，支持任何兼容的服务（Ollama、OpenAI、Azure 等）。提供：

- 单文本和批量文本的异步嵌入生成（使用 `openai` Python 客户端库）
- 指数退避重试
- 批量输入原生支持（一次请求发送多段文本）
- 维度自动检测和适配
- 启动时验证连接和模型可用性
- 同步接口 `get_embedding_sync` 用于非异步环境

### core/vector_store.py

`VectorStore` 是 FAISS 索引的完整封装：

- 支持多种索引类型（FlatIP / FlatL2 / IVF / HNSW）
- 文档 CRUD：添加、搜索、删除（标记删除）、重建索引
- 线程安全的读写锁
- 持久化到磁盘：FAISS 索引（`.bin`）+ 元数据（`.pkl`）+ 配置（`.json`）
- 统计信息追踪
- 文档 ID 到索引位置的映射管理

### core/faiss_persistence.py

FAISS 持久化工具集：

- `FAISSPersistence` — 基础保存/加载
- `IncrementalFAISS` — 增量保存 + 自动备份（保留最近 5 份）
- `ShardedFAISS` — 分片索引，适合大规模向量库

### services/search_service.py

`SearchService` 提供高级搜索能力：

- 三种搜索模式：**向量搜索**（语义）、**关键词搜索**（词频匹配）、**混合搜索**（两者加权合并）
- 搜索过滤器：文档 ID、文件名、文件类型、日期范围、元数据字段
- 结果后处理：块合并（同一文档的多块聚合）、重排序、查询词高亮、摘要
- 多样性过滤：避免结果过于相似
- 结果缓存（5 分钟 TTL）
- 搜索历史记录和热度统计

## API 端点一览

| 方法 | 路径 | 说明 |
|---|---|---|
| **文档管理** |||
| GET | `/api/v1/documents/` | 列出文档（分页） |
| POST | `/api/v1/documents/upload` | 上传文档 |
| GET | `/api/v1/documents/{doc_id}` | 获取文档详情 |
| GET | `/api/v1/documents/{doc_id}/status` | 获取处理状态 |
| DELETE | `/api/v1/documents/{doc_id}` | 删除文档 |
| GET | `/api/v1/documents/stats/overview` | 获取统计信息 |
| POST | `/api/v1/documents/admin/rebuild-index` | 重建索引 |
| POST | `/api/v1/documents/admin/backup` | 备份数据 |
| **搜索** |||
| POST | `/api/v1/search/` | 基础向量搜索 |
| GET | `/api/v1/search/documents` | GET 方式搜索 |
| POST | `/api/v1/search/documents` | 文档内容搜索 |
| POST | `/api/v1/search/advanced` | 高级搜索（混合/重排序/高亮等） |
| GET | `/api/v1/search/suggestions` | 搜索建议 |
| GET | `/api/v1/search/stats` | 搜索引擎统计 |
| DELETE | `/api/v1/search/cache/clear` | 清理缓存 |
| **健康检查** |||
| GET | `/api/v1/health/` | 基础健康检查 |
| GET | `/api/v1/health/detailed` | 详细健康检查（含组件和系统指标） |
| GET | `/api/v1/health/ready` | K8s 就绪探针 |
| GET | `/api/v1/health/live` | K8s 存活探针 |

## 启动方式

```bash
# 确保 Ollama 运行中，且已拉取 bge-m3 模型
ollama pull bge-m3

# 开发模式
uv run uvicorn py_vector.main:app --host 0.0.0.0 --port 8000 --reload

# 或使用启动脚本（自动检查 Ollama）
bash scripts/start.sh
```

## 配置项

通过 `.env` 文件或环境变量配置（参见 `config.py` 的 `Settings` 类）：

| 变量 | 默认值 | 说明 |
|---|---|---|
| `EMBEDDING_BASE_URL` | `http://localhost:11434/v1` | Embedding 服务地址（OpenAI 兼容） |
| `EMBEDDING_MODEL` | `bge-m3` | 嵌入模型名 |
| `EMBEDDING_DIMENSION` | `1024` | 嵌入向量维度 |
| `EMBEDDING_API_KEY` | `ollama` | API 密钥（Ollama 随便填，真实服务用真实 key） |
| **向量存储配置** | | |
| `VECTOR_STORE_TYPE` | `faiss` | 向量存储后端：`faiss` 或 `milvus` |
| `MILVUS_URI` | `""` | Milvus 连接地址。空时启用本地 LanceDB 模式 |
| **模型组配置（主备切换）** | | |
| `MODEL_GROUPS` | `{}` | JSON 格式，每种模型类型可定义多个端点，按顺序尝试。为空时退回到单字段配置 |
| `CHUNK_SIZE` | `512` | 文本切片大小（字符） |
| `CHUNK_OVERLAP` | `50` | 切片重叠量（字符） |
| `MAX_FILE_SIZE` | `50MB` | 上传文件大小限制 |
| `MAX_SEARCH_RESULTS` | `20` | 搜索返回上限 |
| **LLM 配置** | | |
| `LLM_BASE_URL` | `http://localhost:11434/v1` | LLM 服务地址（OpenAI 兼容） |
| `LLM_MODEL` | `qwen2.5` | 生成模型名 |
| `LLM_API_KEY` | `ollama` | API 密钥 |
| `LLM_TEMPERATURE` | `0.7` | 生成温度 |
| `LLM_MAX_TOKENS` | `2048` | 最大生成长度 |
| `LLM_CONTEXT_LENGTH` | `8192` | 上下文窗口大小 |
| **Reranker 配置** | | |
| `RERANKER_BASE_URL` | `""` | Reranker 服务地址（空则使用 Embedding 服务） |
| `RERANKER_MODEL` | `bge-reranker-v2-m3` | Reranker 模型名 |
| `RERANKER_API_KEY` | `ollama` | API 密钥 |
| `RERANKER_ENABLED` | `False` | 是否启用模型重排序 |
| `RERANKER_TOP_K` | `10` | 重排序返回数量 |
| **多模态配置（预留）** | | |
| `MULTIMODAL_ENABLED` | `False` | 是否启用多模态 |
| `MULTIMODAL_BASE_URL` | `""` | 多模态服务地址 |
| `MULTIMODAL_MODEL` | `""` | 多模态模型名 |
| `MULTIMODAL_EMBEDDING_MODEL` | `""` | 多模态嵌入模型名 |

## 设计决策

- **Async-first**：所有 I/O（HTTP 调用、文件操作、FAISS 操作）都是异步的，避免阻塞事件循环
- **FAISS 而非向量数据库**：默认使用本地 FAISS，减少外部依赖，适合中小规模。
- **也可选 Milvus**：通过 `VECTOR_STORE_TYPE=milvus` 切换到 Milvus（本地 LanceDB 模式或远程服务），
  二者共享同一 `VectorStore` 抽象接口，上层代码无需改动。
- **OpenAI 兼容协议**：Embedding / LLM / Reranker 全部使用 OpenAI 协议，切换服务商只需改配置
- **模型组主备切换**：支持每种模型类型配置多个端点，按顺序尝试，失败自动切换。例如本地 Ollama 做主，云端 OpenAI 做备。空配置时退回到单字段，完全向后兼容
- **标记删除**：删除文档时仅标记，重建索引时真正清理——避免频繁重建 FAISS 的不变性约束
- **异步文档处理**：上传后立即返回，后台任务处理文档，通过状态端点轮询进度

## 目录结构

```
py-vector/
├── src/py_vector/          # 主包
├── tests/                  # 测试（覆盖文档处理、嵌入）
├── scripts/start.sh        # 启动脚本
├── deploy/                 # Dockerfile + K8s 部署配置
├── frontend/               # Vue 前端（开发中）
└── examples/               # 示例代码
```

## 已知限制

- `tests/test_embedding.py` 是桩测试，尚未实现
- 目前缺少 CI 配置
- Dockerfile 使用 `python:3.12-slim`，与运行时要求（3.13）不一致
- `faiss_persistence.py` 中的硬编码维度（128）与项目配置（1024）不一致
