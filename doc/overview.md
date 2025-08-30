# py-vector 项目概览

## 项目定位

**py-vector** 是一个基于 **FastAPI** 的 RAG（检索增强生成）API 服务。核心能力是把文档转化为向量，存入向量索引，通过 LLM Agent
进行语义搜索和问答。专为文档问答、语义搜索、个性化推荐等需要语义匹配的场景设计。

## 一句话架构

> 上传文档 → 存储到 S3 → 提取文本 → 切片 → 生成嵌入向量 → 存入向量库 → 接收问题 → Agent 检索 → LLM 生成回答

## 技术栈

| 组件       | 选型                                                                        |
|----------|---------------------------------------------------------------------------|
| API 框架   | FastAPI（async-first）                                                      |
| 向量引擎     | FAISS（默认） / Milvus（可选）                                                    |
| 嵌入模型     | OpenAI 兼容 API（Ollama / OpenAI / Azure 等）                                  |
| 文档解析     | PyMuPDF（PDF）、python-docx（DOCX）、openpyxl / pandas（Excel/CSV）、chardet（编码检测） |
| Agent 框架 | pydantic-ai                                                               |
| 对象存储     | MinIO / S3 兼容（boto3）                                                      |
| 元数据库     | PostgreSQL 16（asyncpg + SQLAlchemy async）                                 |
| 运行环境     | Python 3.13，uv 包管理                                                        |
| 构建系统     | Hatchling                                                                 |

## 数据流

```
HTTP 请求 → 路由 → 端点处理器 → 服务层 → 核心层 → 向量库 / S3 / PG
```

### 文档处理链路（index=true）

```
POST /api/v1/documents/upload
├── 1. 保存临时文件
├── 2. 上传到 S3（media_id → 两级散列路径）
│        例: TtGr1gj1Usaf → media/Tt/Gr/1gj1Usaf
├── 3. 写入 PostgreSQL files 表（media_id、SHA256、文件名等）
├── 4. DocumentProcessor 提取文本
│    ├── PDF → get_text("html"）保留表格布局
│    ├── DOCX / TXT / MD / Excel / CSV / JSON / XML
│    └── 【预留】图片 OCR / 多模态描述
├── 5. 文本分块
│    └── CHUNKING_STRATEGY 指定策略
│         ├── recursive（默认）— 段落→行→句子→字符
│         ├── fixed_size       — 句子边界优先
│         ├── semantic         — 按语义切换点（预留）
│         ├── structure        — 按标题层级（预留）
│         └── agent            — LLM 判定（预留）
├── 6. EmbeddingService 生成向量
│    └── OpenAI 兼容 /v1/embeddings
├── 7. 向量存入 VectorStore
│    ├── FAISS（默认，文件持久化）
│    └── Milvus（可选，VECTOR_STORE_TYPE）
└── 8. 返回 doc_id + media_id，GET /documents/{id}/status 轮询
```

### 搜索链路

```
POST /api/v1/search/
├── 1. EmbeddingService 生成查询向量
├── 2. VectorStore 相似度检索（FAISS / Milvus）
├── 3. 可选后处理
│    ├── RERANKER_ENABLED=true → 模型重排序（/v1/rerank）
│    ├── 启发式重排序（词覆盖率回退）
│    ├── 关键词高亮 / 摘要生成
│    ├── 块合并 / 多样性过滤
│    └── 缓存（5 分钟 TTL）
└── 4. 返回排序结果
```

### RAG 问答链路

```
POST /api/v1/rag/ask
├── 1. 获取全局 RAG Agent（pydantic-ai，单例）
├── 2. Agent.run(user_prompt)
│    ├── LLM 决定调用 search_docs 工具
│    │    └── SearchService.search() → 检索 → 返回文档片段
│    ├── LLM 阅读检索结果
│    └── 生成 AnswerWithCitations
│         ├── answer: 最终回答
│         ├── sources: 引用来源列表
│         └── confidence: 可信度
├── 3. 返回 RAGResponse
└── 降级：Agent 异常时返回纯检索结果
```

## 核心模块

```
src/py_vector/
├── api/v1/endpoints/     # 路由处理器
│   ├── documents.py      # 文档管理（CRUD + 上传 + 备份）
│   ├── search.py         # 搜索（基础 / 高级 / 建议 / 统计）
│   ├── files.py          # 文件上传/下载（S3 + PG）
│   ├── rag.py            # RAG 问答
│   └── health.py         # 健康检查
├── vector_dbs/           # 向量存储实现
│   ├── vector_store.py   # VectorStore 抽象接口 + 工厂函数
│   ├── faiss_vector_store.py  # FAISS 后端
│   ├── faiss_persistence.py   # FAISSPersistence / IncrementalFAISS / ShardedFAISS
│   └── milvus_vector_store.py # Milvus 后端
├── agent/                # 基于 pydantic-ai 的 Agent
│   ├── models/rag.py     # 结构化输出模型
│   ├── tools/search.py   # search_docs 工具
│   └── rag.py            # RAG Agent 定义 + 工厂函数
├── core/                 # 领域核心
│   ├── embedding.py      # EmbeddingService — OpenAI 兼容嵌入
│   ├── document_processor.py  # DocumentProcessor — 多格式文档提取 + 切片
│   ├── database.py       # 异步数据库引擎（asyncpg + SQLAlchemy）
│   ├── s3.py             # MinIO / S3 桶检查 + 文件下载
│   ├── file_store.py     # 文件存储（S3 上传 + PG 元数据 CRUD）
│   ├── reranker.py       # 重排序（模型 / 启发式）
│   └── chunking/         # 分块策略包
│       ├── base.py       # Chunker 抽象基类
│       ├── recursive.py  # 递归分块（默认）
│       ├── fixed_size.py # 固定长度分块
│       ├── semantic.py   # 语义分块（预留）
│       ├── structure.py  # 文档结构分块（预留）
│       └── agent.py      # Agent/LLM 分块（预留）
├── services/             # 业务逻辑
│   ├── document_service.py  # 文档上传、处理、查询、索引重建
│   └── search_service.py    # 搜索、重排序、缓存、历史
├── models/               # Pydantic 数据模型
│   ├── requests.py       # 请求模型
│   ├── responses.py      # 响应模型
│   └── file.py           # SQLAlchemy 表模型 + Pydantic 响应
├── utils/                # 工具
│   └── response_helper.py # 统一 JSON 响应格式
├── config.py             # 环境配置（pydantic-settings）
├── dependencies.py       # FastAPI 依赖注入
└── main.py               # 应用入口 + 生命周期管理
```

### core/embedding.py

`EmbeddingService` 通过 **OpenAI 兼容 API**（`/v1/embeddings`）生成嵌入向量，支持任何兼容的服务（Ollama、OpenAI、Azure 等）。提供：

- 单文本和批量文本的异步嵌入生成（使用 `openai` Python 客户端库）
- 指数退避重试
- 批量输入原生支持（一次请求发送多段文本）
- 维度自动检测和适配
- 启动时验证连接和模型可用性
- 同步接口 `get_embedding_sync` 用于非异步环境

### vector_dbs/vector_store.py

`VectorStore` 是向量存储的抽象接口，FAISS 和 Milvus 均实现此接口：

- 文档 CRUD：添加、搜索、删除（标记删除）、重建索引
- 线程安全的读写锁
- 持久化到磁盘或由服务端托管
- 统计信息追踪

### core/chunking/

分块策略包，统一 `Chunker` 基类，通过 `create_chunker()` 工厂选择：

| 策略              | 实现   | 分割优先级                 |
|-----------------|------|-----------------------|
| `recursive`（默认） | ✅    | 段落 → 行 → 句子 → 字符      |
| `fixed_size`    | ✅    | 句子边界优先，超长句子退回字符       |
| `semantic`      | ⏸ 预留 | 按 embedding 相似度检测主题切换 |
| `structure`     | ⏸ 预留 | 按标题 / 章节层级分割          |
| `agent`         | ⏸ 预留 | LLM 判定语义边界            |

### core/file_store.py

整合 S3 + PG 的文件存储服务：

- `store_file_record()` — 上传文件到 S3 + 写入 `files` 表
- `get_file_record()` / `delete_file_record()` — 查询 / 删除
- SHA256 哈希用于内容去重
- media_id 两级散列路径：`media/{aa}/{bb}/{rest}`

### core/reranker.py

重排序服务，支持多级降级：

```
rerank(query, texts, initial_scores)
├── RERANKER_ENABLED=true → _model_rerank（POST /v1/rerank）
└── _heuristic_rerank（词覆盖率，回退方案）
     └── 综合加权：model × 0.5 + initial × 0.3 + heuristic × 0.2
```

### services/search_service.py

`SearchService` 提供高级搜索能力：

- 三种搜索模式：**向量搜索**（语义）、**关键词搜索**（词频匹配）、**混合搜索**（两者加权合并）
- 搜索过滤器：文档 ID、文件名、文件类型、日期范围、元数据字段
- 重排序：模型重排序（`/v1/rerank`）或启发式回退
- 结果后处理：块合并、查询词高亮、摘要
- 多样性过滤、结果缓存（5 分钟 TTL）

## API 端点一览

| 方法                | 路径                                      | 说明                   |
|-------------------|-----------------------------------------|----------------------|
| **文档管理**          |                                         |                      |
| GET               | `/api/v1/documents/`                    | 列出文档（分页）             |
| POST              | `/api/v1/documents/upload`              | 上传文档（index 参数控制是否索引） |
| GET               | `/api/v1/documents/{doc_id}`            | 获取文档详情               |
| GET               | `/api/v1/documents/{doc_id}/status`     | 获取处理状态               |
| DELETE            | `/api/v1/documents/{doc_id}`            | 删除文档                 |
| GET               | `/api/v1/documents/stats/overview`      | 获取统计信息               |
| POST              | `/api/v1/documents/admin/rebuild-index` | 重建索引                 |
| POST              | `/api/v1/documents/admin/backup`        | 备份数据                 |
| **搜索**            |                                         |                      |
| POST              | `/api/v1/search/`                       | 基础向量搜索               |
| GET               | `/api/v1/search/documents`              | GET 方式搜索             |
| POST              | `/api/v1/search/documents`              | 文档内容搜索               |
| POST              | `/api/v1/search/advanced`               | 高级搜索（混合/重排序/高亮等）     |
| GET               | `/api/v1/search/suggestions`            | 搜索建议                 |
| GET               | `/api/v1/search/stats`                  | 向量存储统计               |
| DELETE            | `/api/v1/search/cache/clear`            | 清理缓存                 |
| **RAG 问答**        |                                         |                      |
| POST              | `/api/v1/rag/ask`                       | RAG 问答（检索+生成，带引用）    |
| **文件管理（S3 + PG）** |                                         |                      |
| POST              | `/api/v1/files/upload`                  | 上传文件到 S3，返回 media_id |
| GET               | `/api/v1/files/{media_id}`              | 获取文件内容               |
| DELETE            | `/api/v1/files/{media_id}`              | 删除文件（S3 + PG）        |
| GET               | `/api/v1/files/{media_id}/meta`         | 获取文件元数据              |
| **健康检查**          |                                         |                      |
| GET               | `/api/v1/health/`                       | 基础健康检查               |
| GET               | `/api/v1/health/detailed`               | 详细健康检查（含组件和系统指标）     |
| GET               | `/api/v1/health/ready`                  | K8s 就绪探针             |
| GET               | `/api/v1/health/live`                   | K8s 存活探针             |

## 启动方式

```bash
# 开发模式
uv run uvicorn py_vector.main:app --host 0.0.0.0 --port 8000 --reload

# 或使用启动脚本
bash scripts/start.sh
```

## 配置项

通过 `.env` 文件或环境变量配置（参见 `config.py` 的 `Settings` 类）：

| 变量                                | 默认值                         | 说明                       |
|-----------------------------------|-----------------------------|--------------------------|
| **Embedding**                     |                             |                          |
| `EMBEDDING_BASE_URL`              | `http://localhost:11434/v1` | 嵌入服务地址                   |
| `EMBEDDING_MODEL`                 | `bge-m3`                    | 嵌入模型名                    |
| `EMBEDDING_DIMENSION`             | `1024`                      | 嵌入向量维度                   |
| `EMBEDDING_API_KEY`               | `ollama`                    | API 密钥                   |
| **LLM（RAG 生成）**                   |                             |                          |
| `LLM_BASE_URL`                    | `http://localhost:11434/v1` | LLM 服务地址                 |
| `LLM_MODEL`                       | `qwen2.5`                   | 生成模型名                    |
| `LLM_API_KEY`                     | `ollama`                    | API 密钥                   |
| `LLM_TEMPERATURE`                 | `0.7`                       | 生成温度                     |
| `LLM_MAX_TOKENS`                  | `2048`                      | 最大生成长度                   |
| **向量存储**                          |                             |                          |
| `VECTOR_STORE_TYPE`               | `faiss`                     | faiss / milvus           |
| `MILVUS_URI`                      | `""`                        | Milvus 连接地址，空=本地 LanceDB |
| **MinIO / S3 对象存储**               |                             |                          |
| `S3_ENABLED`                      | `false`                     | 开关                       |
| `S3_ENDPOINT`                     | `http://localhost:9000`     | MinIO 端点                 |
| `S3_ACCESS_KEY` / `S3_SECRET_KEY` | `minioadmin`                | 密钥                       |
| `S3_BUCKET`                       | `py-vector`                 | 存储桶                      |
| **PostgreSQL**                    |                             |                          |
| `PG_ENABLED`                      | `false`                     | 开关                       |
| `PG_HOST` / `PG_PORT`             | `localhost:5432`            | 连接地址                     |
| `PG_USER` / `PG_PASSWORD`         | `postgres` / `password`     | 凭据                       |
| `PG_DATABASE`                     | `mydb`                      | 数据库名                     |
| **文档处理**                          |                             |                          |
| `CHUNKING_STRATEGY`               | `recursive`                 | 分块策略                     |
| `CHUNK_SIZE`                      | `512`                       | 块大小（字符）                  |
| `CHUNK_OVERLAP`                   | `50`                        | 块重叠（字符）                  |
| `MAX_FILE_SIZE`                   | `50MB`                      | 上传限制                     |
| **Reranker**                      |                             |                          |
| `RERANKER_ENABLED`                | `false`                     | 是否启用模型重排序                |
| `RERANKER_MODEL`                  | `bge-reranker-v2-m3`        | 重排序模型                    |
| **模型组（主备切换）**                     |                             |                          |
| `MODEL_GROUPS`                    | `{}`                        | JSON 格式，空时退回到单字段         |
| **多模态（预留）**                       |                             |                          |
| `MULTIMODAL_ENABLED`              | `false`                     | 开关                       |
| **日志**                            |                             |                          |
| `LOG_LEVEL`                       | `INFO`                      | 日志级别                     |
| `LOG_FILE`                        | `./logs/app.log`            | 日志文件（按天滚动，保留 30 天）       |

## 设计决策

- **Async-first**：所有 I/O（HTTP 调用、S3 操作、向量库）都是异步的，避免阻塞事件循环
- **向量存储抽象**：`VectorStore` 抽象接口 + `FAISSVectorStore` / `MilvusVectorStore`，上层服务无感知
- **OpenAI 兼容协议**：Embedding / LLM / Reranker 全部使用 OpenAI 协议，切换服务商只需改配置
- **模型组主备切换**：每种模型可配置多个端点，按顺序尝试，失败自动切换
- **文件存储参考 Synapse**：media_id → S3 两级散列路径 + PG 元数据表 + SHA256 去重
- **文件流式返回**：`GET /files/{media_id}` 通过 `StreamingResponse` 从 S3 流式返回，不占用本地磁盘
- **标记删除**：FAISS 中删除文档时仅标记，重建索引时真正清理
- **异步文档处理**：上传后立即返回，后台异步处理，通过状态端点轮询
- **Agent 降级**：RAG Agent 调用失败时降级返回纯检索结果，保证可用性

## 目录结构

```
py-vector/
├── src/py_vector/          # 主包
├── tests/                  # 测试
├── scripts/start.sh        # 启动脚本
├── deploy/                 # Dockerfile + K8s 部署配置
├── frontend/               # Vue 3 + Vite 前端
└── doc/                    # 文档
```

## 已知限制

- `tests/test_embedding.py` 是桩测试，尚未实现
- 目前缺少 CI 配置
- Dockerfile 使用 `python:3.12-slim`，与运行时要求（3.13）不一致
- `tests/test_document_processor.py` 中 15 个测试因方法重命名和编码差异为失败状态，需后续修复
