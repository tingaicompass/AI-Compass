"""
多路召回模块

结合稠密向量和 BM25 稀疏向量进行混合检索，
使用 RRF 融合排序

使用方式:
    from retriever import HybridRetriever

    retriever = HybridRetriever(vector_store, embeddings)
    results = retriever.retrieve("什么是深度学习？", top_k=50)
"""

import logging
from typing import List, Dict, Any, Optional

try:
    from .embeddings import OpenAIEmbeddings
    from .vector_store import MilvusVectorStore
    from .query_rewriter import QueryRewriter
    from .config import TOP_K_RETRIEVE
except ImportError:
    from embeddings import OpenAIEmbeddings
    from vector_store import MilvusVectorStore
    from query_rewriter import QueryRewriter
    from config import TOP_K_RETRIEVE


class HybridRetriever:
    """
    混合检索器

    功能:
        - 稠密向量召回（语义相似度）
        - BM25 稀疏向量召回（关键词匹配）
        - RRF 融合排序
        - 可选的 Query 改写
    """

    def __init__(
        self,
        vector_store: MilvusVectorStore,
        embeddings: OpenAIEmbeddings,
        query_rewriter: Optional[QueryRewriter] = None
    ):
        """
        初始化混合检索器

        Args:
            vector_store: Milvus 向量库实例
            embeddings: OpenAI embedding 实例
            query_rewriter: Query 改写器实例（可选）
        """
        self.vector_store = vector_store
        self.embeddings = embeddings
        self.query_rewriter = query_rewriter

        logging.info("HybridRetriever 初始化完成")

    def retrieve(
        self,
        query: str,
        top_k: int = TOP_K_RETRIEVE,
        use_rewrite: bool = False
    ) -> List[Dict[str, Any]]:
        """
        执行混合检索

        流程:
            1. (可选) Query 改写
            2. 生成 query embedding
            3. 执行混合搜索（向量 + BM25）
            4. RRF 融合排序

        Args:
            query: 用户查询
            top_k: 返回的结果数量
            use_rewrite: 是否使用 Query 改写

        Returns:
            List[Dict]: 检索结果列表，每个结果包含:
                - id: 文档 ID
                - text: 原文内容
                - distance: 相似度分数

        Example:
            >>> retriever = HybridRetriever(store, embeddings)
            >>> results = retriever.retrieve("什么是 RAG？", top_k=50)
            >>> for r in results[:5]:
            ...     print(f"ID: {r['id']}, Score: {r['distance']:.4f}")
        """
        if not query or not query.strip():
            return []

        query = query.strip()

        # 可选：Query 改写
        queries_to_search = [query]
        if use_rewrite and self.query_rewriter:
            try:
                queries_to_search = self.query_rewriter.rewrite(query)
                logging.info(f"使用 {len(queries_to_search)} 个查询变体进行检索")
            except Exception as e:
                logging.warning(f"Query 改写失败: {e}，使用原始查询")

        # 收集所有检索结果
        all_results = {}  # 使用 dict 去重，key 为文档 ID

        for q in queries_to_search:
            # 生成 query embedding
            query_vector = self.embeddings.embed_query(q)

            # 执行混合搜索
            results = self.vector_store.hybrid_search(
                query_vector=query_vector,
                query_text=q,
                top_k=top_k
            )

            # 合并结果（保留最高分数）
            for r in results:
                doc_id = r["id"]
                if doc_id not in all_results:
                    all_results[doc_id] = r
                else:
                    # 保留分数更高的
                    if r["distance"] > all_results[doc_id]["distance"]:
                        all_results[doc_id] = r

        # 按分数排序，取 top_k
        sorted_results = sorted(
            all_results.values(),
            key=lambda x: x["distance"],
            reverse=True
        )[:top_k]

        logging.info(f"混合检索返回 {len(sorted_results)} 条结果")
        return sorted_results

    def dense_only_retrieve(
        self,
        query: str,
        top_k: int = TOP_K_RETRIEVE
    ) -> List[Dict[str, Any]]:
        """
        仅使用稠密向量检索

        Args:
            query: 用户查询
            top_k: 返回的结果数量

        Returns:
            List[Dict]: 检索结果列表
        """
        query_vector = self.embeddings.embed_query(query)
        results = self.vector_store.dense_search(
            query_vector=query_vector,
            top_k=top_k
        )
        return results

    def retrieve_with_metadata(
        self,
        query: str,
        top_k: int = TOP_K_RETRIEVE,
        use_rewrite: bool = False
    ) -> Dict[str, Any]:
        """
        检索并返回详细元数据

        Args:
            query: 用户查询
            top_k: 返回的结果数量
            use_rewrite: 是否使用 Query 改写

        Returns:
            Dict: 包含检索结果和元数据:
                - query: 原始查询
                - queries_used: 实际使用的查询列表
                - results: 检索结果列表
                - total: 结果总数
        """
        queries_used = [query]
        if use_rewrite and self.query_rewriter:
            try:
                queries_used = self.query_rewriter.rewrite(query)
            except Exception:
                pass

        results = self.retrieve(query, top_k, use_rewrite)

        return {
            "query": query,
            "queries_used": queries_used,
            "results": results,
            "total": len(results)
        }
