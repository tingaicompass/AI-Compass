"""
Cohere 精排模块

使用 Cohere rerank-multilingual-v3.0 模型对召回结果进行精排，
提高最终结果的相关性

使用方式:
    from reranker import CohereReranker

    reranker = CohereReranker(api_key="your_cohere_api_key")
    ranked_results = reranker.rerank(query, documents, top_k=10)
"""

import logging
from typing import List, Dict, Any, Optional

import cohere

try:
    from .config import COHERE_API_KEY, RERANK_MODEL, TOP_K_RERANK
except ImportError:
    from config import COHERE_API_KEY, RERANK_MODEL, TOP_K_RERANK


class CohereReranker:
    """
    Cohere 精排类

    使用 Cohere 的 rerank API 对召回结果进行重新排序，
    提高搜索结果的相关性
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = RERANK_MODEL
    ):
        """
        初始化 Cohere 精排器

        Args:
            api_key: Cohere API 密钥，默认从 config 读取
            model: 使用的 rerank 模型，默认 rerank-multilingual-v3.0
        """
        self.api_key = api_key or COHERE_API_KEY
        self.model = model

        # 初始化 Cohere 客户端
        # 注意: 不同版本的 cohere SDK 参数不同，使用最简单的初始化方式
        self.client = cohere.Client(api_key=self.api_key)

        logging.info(f"CohereReranker 初始化完成，模型: {self.model}")

    def rerank(
        self,
        query: str,
        documents: List[str],
        top_k: int = TOP_K_RERANK,
        return_documents: bool = True
    ) -> List[Dict[str, Any]]:
        """
        对文档列表进行精排

        Args:
            query: 用户查询
            documents: 待排序的文档列表
            top_k: 返回的结果数量
            return_documents: 是否在结果中包含文档内容

        Returns:
            List[Dict]: 精排结果列表，每个结果包含:
                - index: 原始索引
                - document: 文档内容（如果 return_documents=True）
                - relevance_score: 相关性分数

        Example:
            >>> reranker = CohereReranker(api_key="xxx")
            >>> docs = ["机器学习是...", "深度学习是...", "AI 是..."]
            >>> results = reranker.rerank("什么是深度学习", docs, top_k=2)
            >>> for r in results:
            ...     print(f"Index: {r['index']}, Score: {r['relevance_score']:.4f}")
        """
        if not query or not documents:
            return []

        # 限制文档数量，避免 API 限制
        if len(documents) > 1000:
            logging.warning(f"文档数量 {len(documents)} 超过 1000，截断处理")
            documents = documents[:1000]

        try:
            # 调用 Cohere rerank API
            response = self.client.rerank(
                query=query,
                documents=documents,
                model=self.model,
                top_n=min(top_k, len(documents)),
                return_documents=return_documents
            )

            # 解析结果
            results = []
            for item in response.results:
                result = {
                    "index": item.index,
                    "relevance_score": item.relevance_score
                }
                if return_documents and hasattr(item, "document"):
                    result["document"] = item.document.text
                results.append(result)

            logging.info(f"精排完成，返回 {len(results)} 条结果")
            return results

        except cohere.errors.TooManyRequestsError:
            logging.error("Cohere API 请求过于频繁，稍后重试")
            raise

        except Exception as e:
            logging.error(f"Cohere 精排失败: {e}")
            raise

    def rerank_with_ids(
        self,
        query: str,
        documents: List[Dict[str, Any]],
        top_k: int = TOP_K_RERANK
    ) -> List[Dict[str, Any]]:
        """
        对带有 ID 的文档进行精排

        Args:
            query: 用户查询
            documents: 待排序的文档列表，每个文档包含:
                - id: 文档 ID
                - text: 文档内容
            top_k: 返回的结果数量

        Returns:
            List[Dict]: 精排结果列表，每个结果包含:
                - id: 文档 ID
                - text: 文档内容
                - relevance_score: 相关性分数

        Example:
            >>> docs = [
            ...     {"id": 1, "text": "机器学习是..."},
            ...     {"id": 2, "text": "深度学习是..."}
            ... ]
            >>> results = reranker.rerank_with_ids("什么是深度学习", docs, top_k=1)
        """
        if not query or not documents:
            return []

        # 提取文本列表
        texts = [doc.get("text", "") for doc in documents]

        # 执行精排
        rerank_results = self.rerank(
            query=query,
            documents=texts,
            top_k=top_k,
            return_documents=False
        )

        # 合并结果
        final_results = []
        for r in rerank_results:
            original_doc = documents[r["index"]]
            final_results.append({
                "id": original_doc.get("id"),
                "text": original_doc.get("text", ""),
                "relevance_score": r["relevance_score"]
            })

        return final_results
