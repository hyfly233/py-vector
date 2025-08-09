import hashlib
import logging
import re
from collections import defaultdict
from datetime import datetime
from typing import Any

from py_vector.core.embedding import get_embedding_service
from py_vector.services.document_service import get_document_service
from py_vector.vector_dbs.vector_store import SearchResult, get_vector_store

logger = logging.getLogger(__name__)


class SearchFilter:
    """搜索过滤器"""

    def __init__(
        self,
        doc_ids: list[str] | None = None,
        file_names: list[str] | None = None,
        file_types: list[str] | None = None,
        date_range: tuple[str, str] | None = None,
        min_score: float = 0.0,
        metadata_filters: dict[str, Any] | None = None,
    ):
        """初始化搜索过滤器

        Args:
            doc_ids: 要过滤的文档ID列表，仅返回指定文档的结果
            file_names: 要过滤的文件名列表，支持子串匹配
            file_types: 要过滤的文件类型列表（如 ["pdf", "docx"]）
            date_range: 日期范围过滤，ISO 时间字符串元组
            min_score: 最小相似度分数
            metadata_filters: 元数据过滤条件字典，键为元数据字段名，值为期望值
        """
        self.doc_ids = doc_ids
        self.file_names = file_names
        self.file_types = file_types
        self.date_range = date_range
        self.min_score = min_score
        self.metadata_filters = metadata_filters or {}


class SearchOptions:
    """搜索选项"""

    def __init__(
        self,
        search_type: str = "vector",  # vector, hybrid, keyword
        top_k: int = 10,
        enable_rerank: bool = False,
        enable_highlight: bool = True,
        enable_summary: bool = False,
        chunk_merge: bool = True,
        diversity_threshold: float = 0.7,
    ):
        """初始化搜索选项

        Args:
            search_type: 搜索类型，可选 "vector"（向量搜索）、"hybrid"（混合搜索）、
                "keyword"（关键词搜索）
            top_k: 返回的最大结果数量
            enable_rerank: 是否启用重排序
            enable_highlight: 是否启用关键词高亮
            enable_summary: 是否启用摘要生成
            chunk_merge: 是否合并同一文档的多个分块
            diversity_threshold: 多样性过滤阈值（0.0-1.0），值越低结果越多样化
        """
        self.search_type = search_type
        self.top_k = top_k
        self.enable_rerank = enable_rerank
        self.enable_highlight = enable_highlight
        self.enable_summary = enable_summary
        self.chunk_merge = chunk_merge
        self.diversity_threshold = diversity_threshold


class EnhancedSearchResult:
    """增强的搜索结果"""

    def __init__(
        self,
        doc_id: str,
        file_name: str,
        file_path: str,
        chunks: list[dict[str, Any]],
        max_score: float,
        avg_score: float,
        rank: int,
        highlighted_text: str = "",
        summary: str = "",
        metadata: dict[str, Any] | None = None,
    ):
        """初始化增强搜索结果

        Args:
            doc_id: 文档ID
            file_name: 文件名
            file_path: 文件路径
            chunks: 匹配的文本块列表，每项包含 chunk_index（块索引）、text（文本内容）、
                score（得分）、text_length（文本长度）
            max_score: 最大匹配分数
            avg_score: 平均匹配分数
            rank: 排序位置
            highlighted_text: 高亮文本
            summary: 摘要文本
            metadata: 文档元数据
        """
        self.doc_id = doc_id
        self.file_name = file_name
        self.file_path = file_path
        self.chunks = chunks
        self.max_score = max_score
        self.avg_score = avg_score
        self.rank = rank
        self.highlighted_text = highlighted_text
        self.summary = summary
        self.metadata = metadata or {}

    def to_dict(self) -> dict[str, Any]:
        """转换为字典

        Args:
            无

        Returns:
            返回包含所有字段的字典，关键字段：
            - doc_id: 文档ID
            - file_name: 文件名
            - file_path: 文件路径
            - chunks: 文本块列表
            - max_score: 最大匹配分数
            - avg_score: 平均匹配分数
            - rank: 排序位置
            - highlighted_text: 高亮文本
            - summary: 摘要文本
            - metadata: 文档元数据
            - chunk_count: 文本块数量
        """
        return {
            "doc_id": self.doc_id,
            "file_name": self.file_name,
            "file_path": self.file_path,
            "chunks": self.chunks,
            "max_score": float(self.max_score),
            "avg_score": float(self.avg_score),
            "rank": self.rank,
            "highlighted_text": self.highlighted_text,
            "summary": self.summary,
            "metadata": self.metadata,
            "chunk_count": len(self.chunks),
        }


class SearchService:
    """搜索服务 - 提供高级搜索功能"""

    def __init__(self):
        """初始化搜索服务

        Args:
            无

        Returns:
            None
        """
        self.embedding_service = None
        self.vector_store = None
        self.document_service = None

        # 搜索历史和缓存
        self.search_history: list[dict[str, Any]] = []
        self.search_cache: dict[str, dict[str, Any]] = {}
        self.cache_ttl = 300  # 5分钟缓存

        # 搜索统计
        self.search_stats = {
            "total_searches": 0,
            "avg_search_time": 0.0,
            "popular_queries": defaultdict(int),
            "search_types": defaultdict(int),
        }

    async def initialize(self):
        """初始化搜索服务

        初始化嵌入向量服务、向量存储和文档服务实例。

        Args:
            无

        Returns:
            None
        """
        try:
            self.embedding_service = await get_embedding_service()
            self.vector_store = await get_vector_store()
            self.document_service = await get_document_service()
            logger.info("搜索服务初始化完成")
        except Exception as e:
            logger.error(f"搜索服务初始化失败: {e}")
            raise

    async def search(
        self,
        query: str,
        options: SearchOptions = None,
        filters: SearchFilter = None,
        user_id: str | None = None,
    ) -> dict[str, Any]:
        """
        统一搜索入口

        根据搜索选项执行向量搜索、混合搜索或关键词搜索，并进行后处理
        （合并块、重排序、高亮、摘要、多样性过滤）。

        Args:
            query: 搜索查询文本
            options: 搜索选项（SearchOptions），包括搜索类型、返回数量等配置
            filters: 搜索过滤器（SearchFilter），包括文档ID、文件名、文件类型等过滤条件
            user_id: 用户ID，用于记录搜索历史

        Returns:
            返回包含搜索结果的字典，关键字段：
            - query: 原始查询文本
            - results: EnhancedSearchResult 的字典列表，每项包含 doc_id、file_name、
              chunks、max_score、avg_score、rank、highlighted_text、summary、metadata 等
            - total_results: 结果总数
            - search_time: 搜索耗时（秒）
            - search_type: 搜索类型
            - error: 错误信息（搜索失败时返回）
            - options: 实际使用的搜索选项
            - timestamp: 时间戳
        """
        if not query or not query.strip():
            return self._empty_search_result(query, "查询不能为空")

        options = options or SearchOptions()
        filters = filters or SearchFilter()

        start_time = datetime.now()

        try:
            # 检查缓存
            cache_key = self._generate_cache_key(query, options, filters)
            cached_result = self._get_from_cache(cache_key)
            if cached_result:
                logger.info(f"返回缓存结果: {query}")
                return cached_result

            # 根据搜索类型调用不同的搜索方法
            if options.search_type == "vector":
                results = await self._vector_search(query, options, filters)
            elif options.search_type == "hybrid":
                results = await self._hybrid_search(query, options, filters)
            elif options.search_type == "keyword":
                results = await self._keyword_search(query, options, filters)
            else:
                results = await self._vector_search(query, options, filters)

            # 后处理
            if options.chunk_merge:
                results = await self._merge_chunks(results)

            if options.enable_rerank:
                results = await self._rerank_results(results, query)

            if options.enable_highlight:
                results = await self._add_highlights(results, query)

            if options.enable_summary:
                results = await self._add_summaries(results, query)

            # 多样性过滤
            results = await self._apply_diversity_filter(
                results, options.diversity_threshold
            )

            # 计算搜索时间
            search_time = (datetime.now() - start_time).total_seconds()

            # 构建最终结果
            final_result = {
                "query": query,
                "results": [result.to_dict() for result in results[: options.top_k]],
                "total_results": len(results),
                "search_time": search_time,
                "search_type": options.search_type,
                "timestamp": datetime.now().isoformat(),
                "options": {
                    "top_k": options.top_k,
                    "search_type": options.search_type,
                    "enable_rerank": options.enable_rerank,
                    "enable_highlight": options.enable_highlight,
                    "chunk_merge": options.chunk_merge,
                },
            }

            # 缓存结果
            self._save_to_cache(cache_key, final_result)

            # 记录搜索历史和统计
            await self._record_search(
                query, options, search_time, len(results), user_id
            )

            return final_result

        except Exception as e:
            logger.error(f"搜索失败: {e}")
            return self._empty_search_result(query, f"搜索失败: {str(e)}")

    async def _vector_search(
        self, query: str, options: SearchOptions, filters: SearchFilter
    ) -> list[EnhancedSearchResult]:
        """向量搜索

        使用嵌入向量在向量存储中执行相似度搜索。

        Args:
            query: 搜索查询文本
            options: 搜索选项
            filters: 搜索过滤器

        Returns:
            EnhancedSearchResult 列表，按分数降序排列；搜索失败时返回空列表
        """
        try:
            # 生成查询向量
            query_embedding = await self.embedding_service.get_embedding(query)

            # 执行向量搜索
            search_results = await self.vector_store.search(
                query_embedding=query_embedding,
                top_k=options.top_k * 3,  # 搜索更多结果用于后处理
                filter_doc_ids=filters.doc_ids,
                min_score=filters.min_score,
            )

            # 应用其他过滤器
            filtered_results = await self._apply_filters(search_results, filters)

            # 转换为增强结果
            enhanced_results = await self._convert_to_enhanced_results(filtered_results)

            return enhanced_results

        except Exception as e:
            logger.error(f"向量搜索失败: {e}")
            return []

    async def _hybrid_search(
        self, query: str, options: SearchOptions, filters: SearchFilter
    ) -> list[EnhancedSearchResult]:
        """混合搜索（向量 + 关键词）

        同时执行向量搜索和关键词搜索，并将结果合并排序。

        Args:
            query: 搜索查询文本
            options: 搜索选项
            filters: 搜索过滤器

        Returns:
            合并排序后的 EnhancedSearchResult 列表；搜索失败时返回空列表
        """
        try:
            # 向量搜索
            vector_results = await self._vector_search(query, options, filters)

            # 关键词搜索
            keyword_results = await self._keyword_search(query, options, filters)

            # 合并和重新排序
            combined_results = await self._combine_search_results(
                vector_results, keyword_results, query
            )

            return combined_results

        except Exception as e:
            logger.error(f"混合搜索失败: {e}")
            return []

    async def _keyword_search(
        self, query: str, options: SearchOptions, filters: SearchFilter
    ) -> list[EnhancedSearchResult]:
        """关键词搜索

        通过关键词匹配在文档文本中进行搜索。

        Args:
            query: 搜索查询文本
            options: 搜索选项
            filters: 搜索过滤器

        Returns:
            EnhancedSearchResult 列表，按分数降序排列；搜索失败时返回空列表
        """
        try:
            # 获取所有文档
            all_documents = await self.vector_store.list_documents()

            # 提取关键词
            keywords = self._extract_keywords(query)

            # 搜索匹配的文档
            matching_results = []

            for doc_info in all_documents:
                doc_id = doc_info["doc_id"]

                # 应用过滤器
                if filters.doc_ids and doc_id not in filters.doc_ids:
                    continue

                # 获取文档详情
                doc_details = await self.document_service.get_document_details(doc_id)
                if not doc_details:
                    continue

                # 计算关键词匹配分数
                doc_score = self._calculate_keyword_score(doc_details, keywords)

                if doc_score > filters.min_score:
                    # 找到最佳匹配的块
                    best_chunks = self._find_best_matching_chunks(
                        doc_details["chunks"], keywords, options.top_k
                    )

                    if best_chunks:
                        result = EnhancedSearchResult(
                            doc_id=doc_id,
                            file_name=doc_details["file_name"],
                            file_path=doc_details["file_path"],
                            chunks=best_chunks,
                            max_score=max(chunk["score"] for chunk in best_chunks),
                            avg_score=sum(chunk["score"] for chunk in best_chunks)
                            / len(best_chunks),
                            rank=len(matching_results),
                            metadata=doc_details["metadata"],
                        )
                        matching_results.append(result)

            # 按分数排序
            matching_results.sort(key=lambda x: x.max_score, reverse=True)

            return matching_results

        except Exception as e:
            logger.error(f"关键词搜索失败: {e}")
            return []

    async def _apply_filters(
        self, search_results: list[SearchResult], filters: SearchFilter
    ) -> list[SearchResult]:
        """应用搜索过滤器

        根据文件名、文件类型、日期范围和元数据条件过滤搜索结果。

        Args:
            search_results: 原始搜索结果列表（SearchResult）
            filters: 搜索过滤器（SearchFilter）

        Returns:
            过滤后的 SearchResult 列表
        """
        filtered_results = []

        for result in search_results:
            document = result.document

            # 文件名过滤
            if filters.file_names:
                if not any(
                    name.lower() in document.file_name.lower()
                    for name in filters.file_names
                ):
                    continue

            # 文件类型过滤
            if filters.file_types:
                file_ext = document.file_name.split(".")[-1].lower()
                if file_ext not in [ft.lower() for ft in filters.file_types]:
                    continue

            # 日期范围过滤
            if filters.date_range:
                doc_date = datetime.fromisoformat(
                    document.created_at.replace("Z", "+00:00")
                )
                start_date = datetime.fromisoformat(filters.date_range[0])
                end_date = datetime.fromisoformat(filters.date_range[1])
                if not (start_date <= doc_date <= end_date):
                    continue

            # 元数据过滤
            if filters.metadata_filters:
                match = True
                for key, value in filters.metadata_filters.items():
                    if key not in document.metadata or document.metadata[key] != value:
                        match = False
                        break
                if not match:
                    continue

            filtered_results.append(result)

        return filtered_results

    async def _convert_to_enhanced_results(
        self, search_results: list[SearchResult]
    ) -> list[EnhancedSearchResult]:
        """转换为增强搜索结果

        将原始 SearchResult 列表转换为 EnhancedSearchResult 列表。

        Args:
            search_results: 原始搜索结果列表（SearchResult）

        Returns:
            EnhancedSearchResult 列表
        """
        enhanced_results = []

        for result in search_results:
            document = result.document

            chunk_info = {
                "chunk_index": document.chunk_index,
                "text": document.text,
                "score": result.score,
                "text_length": len(document.text),
            }

            enhanced_result = EnhancedSearchResult(
                doc_id=document.doc_id,
                file_name=document.file_name,
                file_path=document.file_path,
                chunks=[chunk_info],
                max_score=result.score,
                avg_score=result.score,
                rank=result.rank,
                metadata=document.metadata,
            )

            enhanced_results.append(enhanced_result)

        return enhanced_results

    async def _merge_chunks(
        self, results: list[EnhancedSearchResult]
    ) -> list[EnhancedSearchResult]:
        """合并同一文档的多个块

        将同一文档的多个 EnhancedSearchResult 合并为一个，合并分块列表并更新分数。

        Args:
            results: 增强搜索结果列表（EnhancedSearchResult）

        Returns:
            合并后的 EnhancedSearchResult 列表，按最大分数降序排列，
            更新排序位置，每个文档最多保留 5 个最佳块
        """
        doc_groups = defaultdict(list)

        # 按文档ID分组
        for result in results:
            doc_groups[result.doc_id].append(result)

        merged_results = []

        for doc_id, doc_results in doc_groups.items():
            if len(doc_results) == 1:
                merged_results.append(doc_results[0])
            else:
                # 合并多个块
                first_result = doc_results[0]
                all_chunks = []
                scores = []

                for result in doc_results:
                    all_chunks.extend(result.chunks)
                    scores.append(result.max_score)

                # 按分数排序块
                all_chunks.sort(key=lambda x: x["score"], reverse=True)

                merged_result = EnhancedSearchResult(
                    doc_id=doc_id,
                    file_name=first_result.file_name,
                    file_path=first_result.file_path,
                    chunks=all_chunks[:5],  # 最多保留5个最佳块
                    max_score=max(scores),
                    avg_score=sum(scores) / len(scores),
                    rank=min(result.rank for result in doc_results),
                    metadata=first_result.metadata,
                )

                merged_results.append(merged_result)

        # 重新排序
        merged_results.sort(key=lambda x: x.max_score, reverse=True)
        for i, result in enumerate(merged_results):
            result.rank = i

        return merged_results

    async def _rerank_results(
        self, results: list[EnhancedSearchResult], query: str
    ) -> list[EnhancedSearchResult]:
        """重新排序结果

        根据配置启用不同的重排序策略：
        - RERANKER_ENABLED=true → 调用模型重排序（/v1/rerank）
        - 否则 → 基于查询词覆盖率的启发式重排序

        Args:
            results: 增强搜索结果列表（EnhancedSearchResult）
            query: 搜索查询文本

        Returns:
            重排序后的 EnhancedSearchResult 列表，更新 max_score、avg_score 和 rank
        """
        try:
            if not results:
                return results

            # 提取文本列表和原始分数
            texts = []
            initial_scores = []
            for r in results:
                chunk_texts = [c.get("text", "") for c in r.chunks if c.get("text")]
                texts.append(" ".join(chunk_texts))
                initial_scores.append(r.max_score)

            from py_vector.core.reranker import rerank as rerank_scores

            ranked = await rerank_scores(
                query=query,
                texts=texts,
                initial_scores=initial_scores,
                top_k=len(results),
            )

            # 按重排序结果重新排列
            reranked = []
            for orig_idx, new_score in ranked:
                if orig_idx < len(results):
                    r = results[orig_idx]
                    r.max_score = new_score
                    r.avg_score = new_score
                    r.rank = len(reranked)
                    reranked.append(r)

            return reranked

        except Exception as e:
            logger.error(f"重排序失败: {e}")
            return results

    async def _add_highlights(
        self, results: list[EnhancedSearchResult], query: str
    ) -> list[EnhancedSearchResult]:
        """添加高亮

        为每个文本块添加关键词高亮标记。

        Args:
            results: 增强搜索结果列表（EnhancedSearchResult）
            query: 搜索查询文本

        Returns:
            更新了 highlighted_text 字段的 EnhancedSearchResult 列表，
            每个块的 chunks 中包含 highlighted_text 字段
        """
        query_words = query.lower().split()

        for result in results:
            highlighted_chunks = []

            for chunk in result.chunks:
                highlighted_text = self._highlight_text(chunk["text"], query_words)
                highlighted_chunks.append(
                    {**chunk, "highlighted_text": highlighted_text}
                )

            result.chunks = highlighted_chunks

            # 生成整体高亮摘要
            if highlighted_chunks:
                result.highlighted_text = highlighted_chunks[0]["highlighted_text"]

        return results

    def _highlight_text(
        self, text: str, query_words: list[str], max_length: int = 300
    ) -> str:
        """高亮文本中的关键词

        在文本中搜索关键词并添加 ** 高亮标记，返回包含匹配上下文的摘要片段。

        Args:
            text: 原始文本
            query_words: 查询词列表
            max_length: 返回摘要的最大长度

        Returns:
            包含高亮标记（**关键词**）的文本摘要片段，前后可能带有省略号
        """
        try:
            text_lower = text.lower()

            # 找到第一个匹配位置
            first_match_pos = len(text)
            for word in query_words:
                if len(word) > 1:
                    pos = text_lower.find(word.lower())
                    if pos != -1:
                        first_match_pos = min(first_match_pos, pos)

            # 如果没有匹配，返回开头部分
            if first_match_pos == len(text):
                return text[:max_length] + ("..." if len(text) > max_length else "")

            # 计算摘要范围
            start = max(0, first_match_pos - max_length // 3)
            end = min(len(text), start + max_length)
            excerpt = text[start:end]

            # 高亮关键词
            for word in query_words:
                if len(word) > 1:
                    pattern = re.compile(re.escape(word), re.IGNORECASE)
                    excerpt = pattern.sub(f"**{word}**", excerpt)

            # 添加省略号
            prefix = "..." if start > 0 else ""
            suffix = "..." if end < len(text) else ""

            return prefix + excerpt + suffix

        except Exception:
            return text[:max_length] + ("..." if len(text) > max_length else "")

    async def _add_summaries(
        self, results: list[EnhancedSearchResult], query: str
    ) -> list[EnhancedSearchResult]:
        """添加摘要（可以集成大语言模型）

        为每个搜索结果生成文本摘要，当前策略是取最高分块的开头部分。

        Args:
            results: 增强搜索结果列表（EnhancedSearchResult）
            query: 搜索查询文本（保留以支持未来 LLM 摘要集成）

        Returns:
            更新了 summary 字段的 EnhancedSearchResult 列表
        """
        for result in results:
            # 简单的摘要：取最相关的块的开头
            if result.chunks:
                best_chunk = max(result.chunks, key=lambda x: x["score"])
                summary = best_chunk["text"][:200] + (
                    "..." if len(best_chunk["text"]) > 200 else ""
                )
                result.summary = summary

        return results

    async def _apply_diversity_filter(
        self, results: list[EnhancedSearchResult], threshold: float
    ) -> list[EnhancedSearchResult]:
        """应用多样性过滤，避免过于相似的结果

        基于文本相似度过滤相似度高于阈值的重复结果，确保返回结果多样化。

        Args:
            results: 增强搜索结果列表（EnhancedSearchResult）
            threshold: 相似度阈值（0.0-1.0），高于此值视为相似结果

        Returns:
            多样性过滤后的 EnhancedSearchResult 列表，始终保留第一个结果
        """
        if not results or threshold >= 1.0:
            return results

        diverse_results = [results[0]]  # 总是保留第一个结果

        for result in results[1:]:
            is_diverse = True

            for existing_result in diverse_results:
                # 简单的文本相似度检查
                similarity = self._calculate_text_similarity(
                    result.chunks[0]["text"] if result.chunks else "",
                    existing_result.chunks[0]["text"] if existing_result.chunks else "",
                )

                if similarity > threshold:
                    is_diverse = False
                    break

            if is_diverse:
                diverse_results.append(result)

        return diverse_results

    def _calculate_text_similarity(self, text1: str, text2: str) -> float:
        """计算文本相似度

        基于 Jaccard 相似度（词集交集/并集）计算两段文本的相似度。

        Args:
            text1: 第一段文本
            text2: 第二段文本

        Returns:
            相似度分数（0.0-1.0），0 表示完全不相似，1 表示完全相同的词集
        """
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())

        if not words1 or not words2:
            return 0.0

        intersection = len(words1.intersection(words2))
        union = len(words1.union(words2))

        return intersection / union if union > 0 else 0.0

    async def _combine_search_results(
        self,
        vector_results: list[EnhancedSearchResult],
        keyword_results: list[EnhancedSearchResult],
        query: str,
    ) -> list[EnhancedSearchResult]:
        """合并向量搜索和关键词搜索结果

        以向量搜索结果为主（权重 0.7），关键词搜索为辅（权重 0.3），
        按文档ID合并并去重。

        Args:
            vector_results: 向量搜索结果列表
            keyword_results: 关键词搜索结果列表
            query: 搜索查询文本（保留以支持未来扩展）

        Returns:
            合并排序后的 EnhancedSearchResult 列表，按分数降序排列
        """
        combined_map = {}

        # 添加向量搜索结果
        for result in vector_results:
            result.max_score *= 0.7  # 向量搜索权重
            combined_map[result.doc_id] = result

        # 添加关键词搜索结果
        for result in keyword_results:
            result.max_score *= 0.3  # 关键词搜索权重

            if result.doc_id in combined_map:
                # 合并分数
                existing = combined_map[result.doc_id]
                existing.max_score += result.max_score
                existing.avg_score = (existing.avg_score + result.avg_score) / 2

                # 合并块，去重
                existing_chunk_indices = {
                    chunk["chunk_index"] for chunk in existing.chunks
                }
                for chunk in result.chunks:
                    if chunk["chunk_index"] not in existing_chunk_indices:
                        existing.chunks.append(chunk)
            else:
                combined_map[result.doc_id] = result

        # 转换为列表并排序
        combined_results = list(combined_map.values())
        combined_results.sort(key=lambda x: x.max_score, reverse=True)

        # 更新排名
        for i, result in enumerate(combined_results):
            result.rank = i

        return combined_results

    def _extract_keywords(self, query: str) -> list[str]:
        """提取关键词

        从查询文本中提取有意义的搜索关键词，过滤停用词和单字符词。

        Args:
            query: 搜索查询文本

        Returns:
            提取出的关键词列表，已过滤停用词
        """
        # 简单的关键词提取
        words = re.findall(r"\b\w+\b", query.lower())
        # 过滤停用词
        stop_words = {
            "的",
            "是",
            "在",
            "有",
            "和",
            "或",
            "了",
            "与",
            "the",
            "a",
            "an",
            "and",
            "or",
            "but",
            "in",
            "on",
            "at",
            "to",
            "for",
            "of",
            "with",
            "by",
        }
        keywords = [word for word in words if word not in stop_words and len(word) > 1]
        return keywords

    def _calculate_keyword_score(
        self, doc_details: dict[str, Any], keywords: list[str]
    ) -> float:
        """计算关键词匹配分数

        基于关键词在文档文本块中的出现频率计算匹配分数。

        Args:
            doc_details: 文档详情字典，包含 chunks 列表（每项有 text 字段）
            keywords: 搜索关键词列表

        Returns:
            匹配分数（0.0-1.0），表示关键词匹配密度
        """
        if not keywords:
            return 0.0

        total_matches = 0
        total_words = 0

        for chunk in doc_details["chunks"]:
            text_words = chunk["text"].lower().split()
            total_words += len(text_words)

            for keyword in keywords:
                total_matches += text_words.count(keyword)

        return total_matches / max(total_words, 1) if total_words > 0 else 0.0

    def _find_best_matching_chunks(
        self, chunks: list[dict[str, Any]], keywords: list[str], top_k: int
    ) -> list[dict[str, Any]]:
        """找到最佳匹配的文本块

        在文档的分块中找出与关键词匹配度最高的 top_k 个块，并计算标准化分数。

        Args:
            chunks: 文档块列表，每项包含 text 字段
            keywords: 搜索关键词列表
            top_k: 返回的最大块数量

        Returns:
            匹配分数最高的 top_k 个块列表，每项包含原字段和 score（标准化匹配分数）
        """
        scored_chunks = []

        for chunk in chunks:
            text = chunk["text"].lower()
            score = 0

            for keyword in keywords:
                score += text.count(keyword)

            if score > 0:
                scored_chunks.append(
                    {
                        **chunk,
                        "score": score / len(chunk["text"].split()),  # 标准化分数
                    }
                )

        # 按分数排序
        scored_chunks.sort(key=lambda x: x["score"], reverse=True)

        return scored_chunks[:top_k]

    def _generate_cache_key(
        self, query: str, options: SearchOptions, filters: SearchFilter
    ) -> str:
        """生成缓存键

        基于查询文本、搜索选项和过滤器生成 MD5 缓存键。

        Args:
            query: 搜索查询文本
            options: 搜索选项
            filters: 搜索过滤器

        Returns:
            MD5 哈希字符串作为缓存键
        """
        key_parts = [
            query,
            options.search_type,
            str(options.top_k),
            str(options.enable_rerank),
            str(filters.doc_ids),
            str(filters.min_score),
        ]
        key_string = "|".join(str(part) for part in key_parts)
        return hashlib.md5(key_string.encode()).hexdigest()

    def _get_from_cache(self, cache_key: str) -> dict[str, Any] | None:
        """从缓存获取结果

        根据缓存键获取搜索结果，如果缓存已过期则删除并返回 None。

        Args:
            cache_key: 缓存键（MD5 哈希字符串）

        Returns:
            缓存的结果字典（若存在且未过期），否则返回 None
        """
        if cache_key in self.search_cache:
            cache_entry = self.search_cache[cache_key]
            cache_time = datetime.fromisoformat(cache_entry["cached_at"])

            if (datetime.now() - cache_time).total_seconds() < self.cache_ttl:
                return cache_entry["result"]
            else:
                # 缓存过期，删除
                del self.search_cache[cache_key]

        return None

    def _save_to_cache(self, cache_key: str, result: dict[str, Any]):
        """保存到缓存

        将搜索结果保存到缓存中，并自动清理过期缓存（缓存超过 100 项时）。

        Args:
            cache_key: 缓存键（MD5 哈希字符串）
            result: 要缓存的结果字典

        Returns:
            None
        """
        self.search_cache[cache_key] = {
            "result": result,
            "cached_at": datetime.now().isoformat(),
        }

        # 清理过期缓存
        if len(self.search_cache) > 100:  # 限制缓存大小
            self._cleanup_cache()

    def _cleanup_cache(self):
        """清理过期缓存

        遍历缓存并删除所有已过期的条目。

        Args:
            无

        Returns:
            None
        """
        current_time = datetime.now()
        expired_keys = []

        for key, cache_entry in self.search_cache.items():
            cache_time = datetime.fromisoformat(cache_entry["cached_at"])
            if (current_time - cache_time).total_seconds() >= self.cache_ttl:
                expired_keys.append(key)

        for key in expired_keys:
            del self.search_cache[key]

    async def _record_search(
        self,
        query: str,
        options: SearchOptions,
        search_time: float,
        result_count: int,
        user_id: str | None = None,
    ):
        """记录搜索历史和统计

        保存搜索记录到历史列表，并更新搜索统计数据（总搜索次数、平均搜索时间、
        热门查询、搜索类型分布）。

        Args:
            query: 搜索查询文本
            options: 搜索选项
            search_time: 搜索耗时（秒）
            result_count: 结果数量
            user_id: 用户ID

        Returns:
            None
        """
        try:
            # 更新统计
            self.search_stats["total_searches"] += 1
            self.search_stats["avg_search_time"] = (
                self.search_stats["avg_search_time"]
                * (self.search_stats["total_searches"] - 1)
                + search_time
            ) / self.search_stats["total_searches"]
            self.search_stats["popular_queries"][query] += 1
            self.search_stats["search_types"][options.search_type] += 1

            # 记录搜索历史
            search_record = {
                "query": query,
                "user_id": user_id,
                "search_type": options.search_type,
                "result_count": result_count,
                "search_time": search_time,
                "timestamp": datetime.now().isoformat(),
            }

            self.search_history.append(search_record)

            # 限制历史记录数量
            if len(self.search_history) > 1000:
                self.search_history = self.search_history[-500:]  # 保留最近500条

        except Exception as e:
            logger.error(f"记录搜索失败: {e}")

    def _empty_search_result(self, query: str, message: str) -> dict[str, Any]:
        """返回空搜索结果

        构建并返回一个表示搜索失败或无效查询的空结果字典。

        Args:
            query: 原始查询文本
            message: 错误或提示信息

        Returns:
            空搜索结果字典，关键字段：
            - query: 原始查询文本
            - results: 空列表
            - total_results: 0
            - search_time: 0.0
            - error: 错误信息
            - timestamp: 时间戳
        """
        return {
            "query": query,
            "results": [],
            "total_results": 0,
            "search_time": 0.0,
            "error": message,
            "timestamp": datetime.now().isoformat(),
        }

    async def get_search_suggestions(
        self, partial_query: str, limit: int = 5
    ) -> list[str]:
        """获取搜索建议

        基于历史热门查询为部分输入提供搜索建议。

        Args:
            partial_query: 部分查询文本
            limit: 返回建议的最大数量

        Returns:
            搜索建议列表（字符串），按流行度降序排列；获取失败时返回空列表
        """
        try:
            # 基于搜索历史提供建议
            suggestions = []
            partial_lower = partial_query.lower()

            for query, count in self.search_stats["popular_queries"].items():
                if partial_lower in query.lower() and query != partial_query:
                    suggestions.append((query, count))

            # 按流行度排序
            suggestions.sort(key=lambda x: x[1], reverse=True)

            return [suggestion[0] for suggestion in suggestions[:limit]]

        except Exception as e:
            logger.error(f"获取搜索建议失败: {e}")
            return []

    async def get_search_statistics(self) -> dict[str, Any]:
        """获取搜索统计

        获取搜索服务的统计信息，包括总搜索次数、平均搜索时间、热门查询等。

        Args:
            无

        Returns:
            搜索统计信息字典，关键字段：
            - total_searches: 总搜索次数
            - avg_search_time: 平均搜索耗时（秒）
            - popular_queries: 热门查询列表，每项包含 query（查询文本）和 count（次数）
            - search_types: 搜索类型分布字典
            - cache_size: 缓存条目数
            - history_size: 历史记录数
            获取失败时返回空字典
        """
        try:
            # 获取最受欢迎的查询
            popular_queries = sorted(
                self.search_stats["popular_queries"].items(),
                key=lambda x: x[1],
                reverse=True,
            )[:10]

            return {
                "total_searches": self.search_stats["total_searches"],
                "avg_search_time": round(self.search_stats["avg_search_time"], 3),
                "popular_queries": [
                    {"query": q, "count": c} for q, c in popular_queries
                ],
                "search_types": dict(self.search_stats["search_types"]),
                "cache_size": len(self.search_cache),
                "history_size": len(self.search_history),
            }

        except Exception as e:
            logger.error(f"获取搜索统计失败: {e}")
            return {}

    async def clear_cache(self):
        """清理缓存

        清空搜索缓存中的所有条目。

        Args:
            无

        Returns:
            None
        """
        self.search_cache.clear()
        logger.info("搜索缓存已清理")

    async def cleanup(self):
        """清理搜索服务资源

        清空搜索缓存和历史记录，释放服务资源。

        Args:
            无

        Returns:
            None
        """
        try:
            self.search_cache.clear()
            self.search_history.clear()
            logger.info("搜索服务清理完成")
        except Exception as e:
            logger.error(f"搜索服务清理失败: {e}")


# 全局实例
_search_service: SearchService | None = None


async def get_search_service() -> SearchService:
    """获取全局搜索服务实例

    获取或创建全局唯一的 SearchService 单例实例。

    Args:
        无

    Returns:
        SearchService: 已初始化的搜索服务实例
    """
    global _search_service

    if _search_service is None:
        _search_service = SearchService()
        await _search_service.initialize()

    return _search_service


async def cleanup_search_service():
    """清理全局搜索服务

    清理并销毁全局 SearchService 实例，释放资源。

    Args:
        无

    Returns:
        None
    """
    global _search_service

    if _search_service:
        await _search_service.cleanup()
        _search_service = None
