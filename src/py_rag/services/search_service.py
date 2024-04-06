import hashlib
import logging
import re
from collections import defaultdict
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple

from py_faiss.core.embedding import get_embedding_service
from py_faiss.core.vector_store import get_vector_store, SearchResult
from py_faiss.services.document_service import get_document_service

logger = logging.getLogger(__name__)


class SearchFilter:
    """搜索过滤器"""

    def __init__(
            self,
            doc_ids: Optional[List[str]] = None,
            file_names: Optional[List[str]] = None,
            file_types: Optional[List[str]] = None,
            date_range: Optional[Tuple[str, str]] = None,
            min_score: float = 0.0,
            metadata_filters: Optional[Dict[str, Any]] = None
    ):
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
            diversity_threshold: float = 0.7
    ):
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
            chunks: List[Dict[str, Any]],
            max_score: float,
            avg_score: float,
            rank: int,
            highlighted_text: str = "",
            summary: str = "",
            metadata: Optional[Dict[str, Any]] = None
    ):
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

    def to_dict(self) -> Dict[str, Any]:
        return {
            'doc_id': self.doc_id,
            'file_name': self.file_name,
            'file_path': self.file_path,
            'chunks': self.chunks,
            'max_score': float(self.max_score),
            'avg_score': float(self.avg_score),
            'rank': self.rank,
            'highlighted_text': self.highlighted_text,
            'summary': self.summary,
            'metadata': self.metadata,
            'chunk_count': len(self.chunks)
        }


class SearchService:
    """搜索服务 - 提供高级搜索功能"""

    def __init__(self):
        self.embedding_service = None
        self.vector_store = None
        self.document_service = None

        # 搜索历史和缓存
        self.search_history: List[Dict[str, Any]] = []
        self.search_cache: Dict[str, Dict[str, Any]] = {}
        self.cache_ttl = 300  # 5分钟缓存

        # 搜索统计
        self.search_stats = {
            'total_searches': 0,
            'avg_search_time': 0.0,
            'popular_queries': defaultdict(int),
            'search_types': defaultdict(int)
        }

    async def initialize(self):
        """初始化搜索服务"""
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
            user_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        统一搜索入口

        Args:
            query: 搜索查询
            options: 搜索选项
            filters: 搜索过滤器
            user_id: 用户ID

        Returns:
            搜索结果
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
            results = await self._apply_diversity_filter(results, options.diversity_threshold)

            # 计算搜索时间
            search_time = (datetime.now() - start_time).total_seconds()

            # 构建最终结果
            final_result = {
                'query': query,
                'results': [result.to_dict() for result in results[:options.top_k]],
                'total_results': len(results),
                'search_time': search_time,
                'search_type': options.search_type,
                'timestamp': datetime.now().isoformat(),
                'options': {
                    'top_k': options.top_k,
                    'search_type': options.search_type,
                    'enable_rerank': options.enable_rerank,
                    'enable_highlight': options.enable_highlight,
                    'chunk_merge': options.chunk_merge
                }
            }

            # 缓存结果
            self._save_to_cache(cache_key, final_result)

            # 记录搜索历史和统计
            await self._record_search(query, options, search_time, len(results), user_id)

            return final_result

        except Exception as e:
            logger.error(f"搜索失败: {e}")
            return self._empty_search_result(query, f"搜索失败: {str(e)}")

    async def _vector_search(
            self,
            query: str,
            options: SearchOptions,
            filters: SearchFilter
    ) -> List[EnhancedSearchResult]:
        """向量搜索"""
        try:
            # 生成查询向量
            query_embedding = await self.embedding_service.get_embedding(query)

            # 执行向量搜索
            search_results = await self.vector_store.search(
                query_embedding=query_embedding,
                top_k=options.top_k * 3,  # 搜索更多结果用于后处理
                filter_doc_ids=filters.doc_ids,
                min_score=filters.min_score
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
            self,
            query: str,
            options: SearchOptions,
            filters: SearchFilter
    ) -> List[EnhancedSearchResult]:
        """混合搜索（向量 + 关键词）"""
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
            self,
            query: str,
            options: SearchOptions,
            filters: SearchFilter
    ) -> List[EnhancedSearchResult]:
        """关键词搜索"""
        try:
            # 获取所有文档
            all_documents = await self.vector_store.list_documents()

            # 提取关键词
            keywords = self._extract_keywords(query)

            # 搜索匹配的文档
            matching_results = []

            for doc_info in all_documents:
                doc_id = doc_info['doc_id']

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
                        doc_details['chunks'], keywords, options.top_k
                    )

                    if best_chunks:
                        result = EnhancedSearchResult(
                            doc_id=doc_id,
                            file_name=doc_details['file_name'],
                            file_path=doc_details['file_path'],
                            chunks=best_chunks,
                            max_score=max(chunk['score'] for chunk in best_chunks),
                            avg_score=sum(chunk['score'] for chunk in best_chunks) / len(best_chunks),
                            rank=len(matching_results),
                            metadata=doc_details['metadata']
                        )
                        matching_results.append(result)

            # 按分数排序
            matching_results.sort(key=lambda x: x.max_score, reverse=True)

            return matching_results

        except Exception as e:
            logger.error(f"关键词搜索失败: {e}")
            return []

    async def _apply_filters(
            self,
            search_results: List[SearchResult],
            filters: SearchFilter
    ) -> List[SearchResult]:
        """应用搜索过滤器"""
        filtered_results = []

        for result in search_results:
            document = result.document

            # 文件名过滤
            if filters.file_names:
                if not any(name.lower() in document.file_name.lower() for name in filters.file_names):
                    continue

            # 文件类型过滤
            if filters.file_types:
                file_ext = document.file_name.split('.')[-1].lower()
                if file_ext not in [ft.lower() for ft in filters.file_types]:
                    continue

            # 日期范围过滤
            if filters.date_range:
                doc_date = datetime.fromisoformat(document.created_at.replace('Z', '+00:00'))
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
            self,
            search_results: List[SearchResult]
    ) -> List[EnhancedSearchResult]:
        """转换为增强搜索结果"""
        enhanced_results = []

        for result in search_results:
            document = result.document

            chunk_info = {
                'chunk_index': document.chunk_index,
                'text': document.text,
                'score': result.score,
                'text_length': len(document.text)
            }

            enhanced_result = EnhancedSearchResult(
                doc_id=document.doc_id,
                file_name=document.file_name,
                file_path=document.file_path,
                chunks=[chunk_info],
                max_score=result.score,
                avg_score=result.score,
                rank=result.rank,
                metadata=document.metadata
            )

            enhanced_results.append(enhanced_result)

        return enhanced_results

    async def _merge_chunks(
            self,
            results: List[EnhancedSearchResult]
    ) -> List[EnhancedSearchResult]:
        """合并同一文档的多个块"""
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
                all_chunks.sort(key=lambda x: x['score'], reverse=True)

                merged_result = EnhancedSearchResult(
                    doc_id=doc_id,
                    file_name=first_result.file_name,
                    file_path=first_result.file_path,
                    chunks=all_chunks[:5],  # 最多保留5个最佳块
                    max_score=max(scores),
                    avg_score=sum(scores) / len(scores),
                    rank=min(result.rank for result in doc_results),
                    metadata=first_result.metadata
                )

                merged_results.append(merged_result)

        # 重新排序
        merged_results.sort(key=lambda x: x.max_score, reverse=True)
        for i, result in enumerate(merged_results):
            result.rank = i

        return merged_results

    async def _rerank_results(
            self,
            results: List[EnhancedSearchResult],
            query: str
    ) -> List[EnhancedSearchResult]:
        """重新排序结果"""
        try:
            # 这里可以集成更复杂的重排序模型
            # 目前使用简单的文本相似度重排序

            query_words = set(query.lower().split())

            for result in results:
                # 计算查询词覆盖率
                text_words = set()
                for chunk in result.chunks:
                    text_words.update(chunk['text'].lower().split())

                coverage = len(query_words.intersection(text_words)) / len(query_words) if query_words else 0

                # 结合原始分数和覆盖率
                result.max_score = result.max_score * 0.7 + coverage * 0.3
                result.avg_score = result.avg_score * 0.7 + coverage * 0.3

            # 重新排序
            results.sort(key=lambda x: x.max_score, reverse=True)
            for i, result in enumerate(results):
                result.rank = i

            return results

        except Exception as e:
            logger.error(f"重排序失败: {e}")
            return results

    async def _add_highlights(
            self,
            results: List[EnhancedSearchResult],
            query: str
    ) -> List[EnhancedSearchResult]:
        """添加高亮"""
        query_words = query.lower().split()

        for result in results:
            highlighted_chunks = []

            for chunk in result.chunks:
                highlighted_text = self._highlight_text(chunk['text'], query_words)
                highlighted_chunks.append({
                    **chunk,
                    'highlighted_text': highlighted_text
                })

            result.chunks = highlighted_chunks

            # 生成整体高亮摘要
            if highlighted_chunks:
                result.highlighted_text = highlighted_chunks[0]['highlighted_text']

        return results

    def _highlight_text(self, text: str, query_words: List[str], max_length: int = 300) -> str:
        """高亮文本中的关键词"""
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
            self,
            results: List[EnhancedSearchResult],
            query: str
    ) -> List[EnhancedSearchResult]:
        """添加摘要（可以集成大语言模型）"""
        for result in results:
            # 简单的摘要：取最相关的块的开头
            if result.chunks:
                best_chunk = max(result.chunks, key=lambda x: x['score'])
                summary = best_chunk['text'][:200] + ("..." if len(best_chunk['text']) > 200 else "")
                result.summary = summary

        return results

    async def _apply_diversity_filter(
            self,
            results: List[EnhancedSearchResult],
            threshold: float
    ) -> List[EnhancedSearchResult]:
        """应用多样性过滤，避免过于相似的结果"""
        if not results or threshold >= 1.0:
            return results

        diverse_results = [results[0]]  # 总是保留第一个结果

        for result in results[1:]:
            is_diverse = True

            for existing_result in diverse_results:
                # 简单的文本相似度检查
                similarity = self._calculate_text_similarity(
                    result.chunks[0]['text'] if result.chunks else "",
                    existing_result.chunks[0]['text'] if existing_result.chunks else ""
                )

                if similarity > threshold:
                    is_diverse = False
                    break

            if is_diverse:
                diverse_results.append(result)

        return diverse_results

    def _calculate_text_similarity(self, text1: str, text2: str) -> float:
        """计算文本相似度"""
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())

        if not words1 or not words2:
            return 0.0

        intersection = len(words1.intersection(words2))
        union = len(words1.union(words2))

        return intersection / union if union > 0 else 0.0

    async def _combine_search_results(
            self,
            vector_results: List[EnhancedSearchResult],
            keyword_results: List[EnhancedSearchResult],
            query: str
    ) -> List[EnhancedSearchResult]:
        """合并向量搜索和关键词搜索结果"""
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
                existing_chunk_indices = {chunk['chunk_index'] for chunk in existing.chunks}
                for chunk in result.chunks:
                    if chunk['chunk_index'] not in existing_chunk_indices:
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

    def _extract_keywords(self, query: str) -> List[str]:
        """提取关键词"""
        # 简单的关键词提取
        words = re.findall(r'\b\w+\b', query.lower())
        # 过滤停用词
        stop_words = {'的', '是', '在', '有', '和', '或', '了', '与', 'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on',
                      'at', 'to', 'for', 'of', 'with', 'by'}
        keywords = [word for word in words if word not in stop_words and len(word) > 1]
        return keywords

    def _calculate_keyword_score(self, doc_details: Dict[str, Any], keywords: List[str]) -> float:
        """计算关键词匹配分数"""
        if not keywords:
            return 0.0

        total_matches = 0
        total_words = 0

        for chunk in doc_details['chunks']:
            text_words = chunk['text'].lower().split()
            total_words += len(text_words)

            for keyword in keywords:
                total_matches += text_words.count(keyword)

        return total_matches / max(total_words, 1) if total_words > 0 else 0.0

    def _find_best_matching_chunks(
            self,
            chunks: List[Dict[str, Any]],
            keywords: List[str],
            top_k: int
    ) -> List[Dict[str, Any]]:
        """找到最佳匹配的文本块"""
        scored_chunks = []

        for chunk in chunks:
            text = chunk['text'].lower()
            score = 0

            for keyword in keywords:
                score += text.count(keyword)

            if score > 0:
                scored_chunks.append({
                    **chunk,
                    'score': score / len(chunk['text'].split())  # 标准化分数
                })

        # 按分数排序
        scored_chunks.sort(key=lambda x: x['score'], reverse=True)

        return scored_chunks[:top_k]

    def _generate_cache_key(self, query: str, options: SearchOptions, filters: SearchFilter) -> str:
        """生成缓存键"""
        key_parts = [
            query,
            options.search_type,
            str(options.top_k),
            str(options.enable_rerank),
            str(filters.doc_ids),
            str(filters.min_score)
        ]
        key_string = "|".join(str(part) for part in key_parts)
        return hashlib.md5(key_string.encode()).hexdigest()

    def _get_from_cache(self, cache_key: str) -> Optional[Dict[str, Any]]:
        """从缓存获取结果"""
        if cache_key in self.search_cache:
            cache_entry = self.search_cache[cache_key]
            cache_time = datetime.fromisoformat(cache_entry['cached_at'])

            if (datetime.now() - cache_time).total_seconds() < self.cache_ttl:
                return cache_entry['result']
            else:
                # 缓存过期，删除
                del self.search_cache[cache_key]

        return None

    def _save_to_cache(self, cache_key: str, result: Dict[str, Any]):
        """保存到缓存"""
        self.search_cache[cache_key] = {
            'result': result,
            'cached_at': datetime.now().isoformat()
        }

        # 清理过期缓存
        if len(self.search_cache) > 100:  # 限制缓存大小
            self._cleanup_cache()

    def _cleanup_cache(self):
        """清理过期缓存"""
        current_time = datetime.now()
        expired_keys = []

        for key, cache_entry in self.search_cache.items():
            cache_time = datetime.fromisoformat(cache_entry['cached_at'])
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
            user_id: Optional[str] = None
    ):
        """记录搜索历史和统计"""
        try:
            # 更新统计
            self.search_stats['total_searches'] += 1
            self.search_stats['avg_search_time'] = (
                    (self.search_stats['avg_search_time'] * (self.search_stats['total_searches'] - 1) + search_time) /
                    self.search_stats['total_searches']
            )
            self.search_stats['popular_queries'][query] += 1
            self.search_stats['search_types'][options.search_type] += 1

            # 记录搜索历史
            search_record = {
                'query': query,
                'user_id': user_id,
                'search_type': options.search_type,
                'result_count': result_count,
                'search_time': search_time,
                'timestamp': datetime.now().isoformat()
            }

            self.search_history.append(search_record)

            # 限制历史记录数量
            if len(self.search_history) > 1000:
                self.search_history = self.search_history[-500:]  # 保留最近500条

        except Exception as e:
            logger.error(f"记录搜索失败: {e}")

    def _empty_search_result(self, query: str, message: str) -> Dict[str, Any]:
        """返回空搜索结果"""
        return {
            'query': query,
            'results': [],
            'total_results': 0,
            'search_time': 0.0,
            'error': message,
            'timestamp': datetime.now().isoformat()
        }

    async def get_search_suggestions(self, partial_query: str, limit: int = 5) -> List[str]:
        """获取搜索建议"""
        try:
            # 基于搜索历史提供建议
            suggestions = []
            partial_lower = partial_query.lower()

            for query, count in self.search_stats['popular_queries'].items():
                if partial_lower in query.lower() and query != partial_query:
                    suggestions.append((query, count))

            # 按流行度排序
            suggestions.sort(key=lambda x: x[1], reverse=True)

            return [suggestion[0] for suggestion in suggestions[:limit]]

        except Exception as e:
            logger.error(f"获取搜索建议失败: {e}")
            return []

    async def get_search_statistics(self) -> Dict[str, Any]:
        """获取搜索统计"""
        try:
            # 获取最受欢迎的查询
            popular_queries = sorted(
                self.search_stats['popular_queries'].items(),
                key=lambda x: x[1],
                reverse=True
            )[:10]

            return {
                'total_searches': self.search_stats['total_searches'],
                'avg_search_time': round(self.search_stats['avg_search_time'], 3),
                'popular_queries': [{'query': q, 'count': c} for q, c in popular_queries],
                'search_types': dict(self.search_stats['search_types']),
                'cache_size': len(self.search_cache),
                'history_size': len(self.search_history)
            }

        except Exception as e:
            logger.error(f"获取搜索统计失败: {e}")
            return {}

    async def clear_cache(self):
        """清理缓存"""
        self.search_cache.clear()
        logger.info("搜索缓存已清理")

    async def cleanup(self):
        """清理搜索服务资源"""
        try:
            self.search_cache.clear()
            self.search_history.clear()
            logger.info("搜索服务清理完成")
        except Exception as e:
            logger.error(f"搜索服务清理失败: {e}")


# 全局实例
_search_service: Optional[SearchService] = None


async def get_search_service() -> SearchService:
    """获取全局搜索服务实例"""
    global _search_service

    if _search_service is None:
        _search_service = SearchService()
        await _search_service.initialize()

    return _search_service


async def cleanup_search_service():
    """清理全局搜索服务"""
    global _search_service

    if _search_service:
        await _search_service.cleanup()
        _search_service = None
