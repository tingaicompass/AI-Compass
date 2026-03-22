"""
Milvus/Zilliz Cloud 向量库操作模块

支持功能:
    - 创建 Collection（稠密向量 + BM25 稀疏向量）
    - 插入文档
    - 混合搜索（向量 + 关键词）
    - RRF 融合排序

使用方式:
    from vector_store import MilvusVectorStore

    store = MilvusVectorStore(
        uri="https://xxx.zillizcloud.com",
        token="your_api_key",
        collection_name="ai_knowledge_base"
    )
    store.create_collection()
    store.insert_documents([{"text": "...", "dense_vector": [...]}])
    results = store.hybrid_search(query_vector, query_text, top_k=50)
"""

import logging
from typing import List, Dict, Any, Optional

from pymilvus import (
    MilvusClient,
    DataType,
    AnnSearchRequest,
    RRFRanker,
    Function,
    FunctionType,
)

try:
    from .config import (
        ZILLIZ_ENDPOINT,
        ZILLIZ_API_KEY,
        COLLECTION_NAME,
        EMBEDDING_DIM,
        MAX_TEXT_LENGTH,
    )
except ImportError:
    from config import (
        ZILLIZ_ENDPOINT,
        ZILLIZ_API_KEY,
        COLLECTION_NAME,
        EMBEDDING_DIM,
        MAX_TEXT_LENGTH,
    )


class MilvusVectorStore:
    """
    Milvus 向量库管理类

    功能:
        - 连接 Zilliz Cloud
        - 创建支持混合搜索的 Collection
        - 插入带有稠密向量的文档
        - 执行混合搜索（稠密向量 + BM25 稀疏向量）
    """

    def __init__(
        self,
        uri: Optional[str] = None,
        token: Optional[str] = None,
        collection_name: str = COLLECTION_NAME
    ):
        """
        初始化 Milvus 客户端

        Args:
            uri: Zilliz Cloud endpoint，默认从 config 读取
            token: API token，默认从 config 读取
            collection_name: Collection 名称
        """
        self.uri = uri or ZILLIZ_ENDPOINT
        self.token = token or ZILLIZ_API_KEY
        self.collection_name = collection_name

        # 初始化 Milvus 客户端
        self.client = MilvusClient(
            uri=self.uri,
            token=self.token
        )

        logging.info(f"MilvusVectorStore 初始化完成，连接: {self.uri}")

    def create_collection(self, drop_if_exists: bool = False) -> None:
        """
        创建 Collection

        Schema 包含:
            - id: INT64 主键，自动生成
            - text: VARCHAR 原文内容
            - dense_vector: FLOAT_VECTOR(1536) 稠密向量
            - sparse_vector: SPARSE_FLOAT_VECTOR BM25 稀疏向量（自动生成）

        Args:
            drop_if_exists: 如果 Collection 已存在是否删除重建
        """
        # 检查是否已存在
        if self.client.has_collection(self.collection_name):
            if drop_if_exists:
                logging.info(f"删除已存在的 Collection: {self.collection_name}")
                self.client.drop_collection(self.collection_name)
            else:
                logging.info(f"Collection 已存在: {self.collection_name}")
                return

        # 创建 Schema
        schema = self.client.create_schema(auto_id=True, enable_dynamic_field=False)

        # 添加字段
        # 主键
        schema.add_field(
            field_name="id",
            datatype=DataType.INT64,
            is_primary=True,
            auto_id=True
        )

        # 原文内容（启用分析器，用于 BM25）
        schema.add_field(
            field_name="text",
            datatype=DataType.VARCHAR,
            max_length=MAX_TEXT_LENGTH,
            enable_analyzer=True,           # 启用分析器
            analyzer_params={"type": "chinese"},  # 使用中文分析器
            enable_match=True               # 启用全文匹配
        )

        # 稠密向量（OpenAI ada-002 生成）
        schema.add_field(
            field_name="dense_vector",
            datatype=DataType.FLOAT_VECTOR,
            dim=EMBEDDING_DIM
        )

        # 稀疏向量（BM25 自动生成）
        schema.add_field(
            field_name="sparse_vector",
            datatype=DataType.SPARSE_FLOAT_VECTOR
        )

        # 添加 BM25 函数：自动从 text 生成 sparse_vector
        bm25_function = Function(
            name="bm25_function",
            function_type=FunctionType.BM25,
            input_field_names=["text"],
            output_field_names=["sparse_vector"]
        )
        schema.add_function(bm25_function)

        # 创建索引参数
        index_params = self.client.prepare_index_params()

        # 稠密向量索引（AUTOINDEX 自动选择最优索引类型）
        # Zilliz Cloud 会根据数据规模和硬件自动选择 HNSW/IVF_FLAT 等
        index_params.add_index(
            field_name="dense_vector",
            index_type="AUTOINDEX",
            metric_type="COSINE"
        )

        # 稀疏向量索引（AUTOINDEX 自动优化 BM25 索引）
        index_params.add_index(
            field_name="sparse_vector",
            index_type="AUTOINDEX",
            metric_type="BM25"
        )

        # 创建 Collection
        self.client.create_collection(
            collection_name=self.collection_name,
            schema=schema,
            index_params=index_params
        )

        logging.info(f"Collection 创建成功: {self.collection_name}")

    def insert_documents(self, documents: List[Dict[str, Any]]) -> List[int]:
        """
        插入文档到 Collection

        Args:
            documents: 文档列表，每个文档包含:
                - text: str 原文内容
                - dense_vector: List[float] 稠密向量

        Returns:
            List[int]: 插入的文档 ID 列表

        Example:
            >>> store.insert_documents([
            ...     {"text": "什么是机器学习", "dense_vector": [0.1, 0.2, ...]},
            ...     {"text": "深度学习基础", "dense_vector": [0.3, 0.4, ...]}
            ... ])
        """
        if not documents:
            return []

        # 准备插入数据
        data = []
        for doc in documents:
            data.append({
                "text": doc["text"],
                "dense_vector": doc["dense_vector"]
                # sparse_vector 由 BM25 函数自动生成，无需手动提供
            })

        # 插入数据
        result = self.client.insert(
            collection_name=self.collection_name,
            data=data
        )

        logging.info(f"插入 {len(documents)} 条文档")
        return result.get("ids", [])

    def hybrid_search(
        self,
        query_vector: List[float],
        query_text: str,
        top_k: int = 50
    ) -> List[Dict[str, Any]]:
        """
        执行混合搜索（稠密向量 + BM25 稀疏向量）

        使用 RRF (Reciprocal Rank Fusion) 融合两路召回结果

        Args:
            query_vector: 查询的稠密向量
            query_text: 查询的原始文本（用于 BM25）
            top_k: 返回的结果数量

        Returns:
            List[Dict]: 搜索结果列表，每个结果包含:
                - id: 文档 ID
                - text: 原文内容
                - distance: 相似度分数
        """
        # 稠密向量搜索请求
        dense_search_params = {
            "metric_type": "COSINE",
            "params": {"nprobe": 10}
        }
        dense_req = AnnSearchRequest(
            data=[query_vector],
            anns_field="dense_vector",
            param=dense_search_params,
            limit=top_k
        )

        # BM25 稀疏向量搜索请求
        sparse_search_params = {
            "metric_type": "BM25",
            "params": {}
        }
        sparse_req = AnnSearchRequest(
            data=[query_text],  # BM25 直接使用文本
            anns_field="sparse_vector",
            param=sparse_search_params,
            limit=top_k
        )

        # 执行混合搜索，使用 RRF 融合
        results = self.client.hybrid_search(
            collection_name=self.collection_name,
            reqs=[dense_req, sparse_req],
            ranker=RRFRanker(),
            limit=top_k,
            output_fields=["text"]
        )

        # 解析结果
        search_results = []
        if results and len(results) > 0:
            for hit in results[0]:
                search_results.append({
                    "id": hit.get("id"),
                    "text": hit.get("entity", {}).get("text", ""),
                    "distance": hit.get("distance", 0.0)
                })

        logging.info(f"混合搜索返回 {len(search_results)} 条结果")
        return search_results

    def dense_search(
        self,
        query_vector: List[float],
        top_k: int = 50
    ) -> List[Dict[str, Any]]:
        """
        仅执行稠密向量搜索

        Args:
            query_vector: 查询的稠密向量
            top_k: 返回的结果数量

        Returns:
            List[Dict]: 搜索结果列表
        """
        results = self.client.search(
            collection_name=self.collection_name,
            data=[query_vector],
            anns_field="dense_vector",
            search_params={"metric_type": "COSINE", "params": {"nprobe": 10}},
            limit=top_k,
            output_fields=["text"]
        )

        search_results = []
        if results and len(results) > 0:
            for hit in results[0]:
                search_results.append({
                    "id": hit.get("id"),
                    "text": hit.get("entity", {}).get("text", ""),
                    "distance": hit.get("distance", 0.0)
                })

        return search_results

    def drop_collection(self) -> None:
        """删除 Collection"""
        if self.client.has_collection(self.collection_name):
            self.client.drop_collection(self.collection_name)
            logging.info(f"Collection 已删除: {self.collection_name}")

    def get_collection_stats(self) -> Dict[str, Any]:
        """获取 Collection 统计信息"""
        if not self.client.has_collection(self.collection_name):
            return {"exists": False}

        stats = self.client.get_collection_stats(self.collection_name)
        return {
            "exists": True,
            "row_count": stats.get("row_count", 0),
            "collection_name": self.collection_name
        }
