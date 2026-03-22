"""
Milvus RAG 问答系统模块

基于 Milvus/Zilliz Cloud 实现的完整 RAG 系统，包含:
- 向量索引构建（稠密向量 + BM25 稀疏向量）
- Query 改写（Gemini）
- 多路召回 + RRF 融合
- Cohere 精排
- LLM 问答

使用方式:
    from mind.internal_tools.general.milvus import RAGPipeline

    # 初始化（需要先配置 API keys）
    pipeline = RAGPipeline(config={
        "zilliz_endpoint": "https://xxx.zillizcloud.com",
        "zilliz_api_key": "your_api_key",
        "gemini_api_key": "your_gemini_key"  # 可选
    })

    # 构建索引
    pipeline.build_index([
        {"text": "文档内容1"},
        {"text": "文档内容2"}
    ])

    # 问答
    result = pipeline.query("你的问题")
    print(result["answer"])
"""

# 配置模块
from .config import (
    get_config,
    update_config,
    ZILLIZ_ENDPOINT,
    ZILLIZ_API_KEY,
    OPENAI_API_KEY,
    GEMINI_API_KEY,
    COHERE_API_KEY,
)

# 核心组件
from .embeddings import OpenAIEmbeddings
from .vector_store import MilvusVectorStore
from .query_rewriter import QueryRewriter
from .retriever import HybridRetriever
from .reranker import CohereReranker
from .qa_chain import QAChain

# 完整 Pipeline
from .rag_pipeline import RAGPipeline

__all__ = [
    # 配置
    "get_config",
    "update_config",
    "ZILLIZ_ENDPOINT",
    "ZILLIZ_API_KEY",
    "OPENAI_API_KEY",
    "GEMINI_API_KEY",
    "COHERE_API_KEY",
    # 组件
    "OpenAIEmbeddings",
    "MilvusVectorStore",
    "QueryRewriter",
    "HybridRetriever",
    "CohereReranker",
    "QAChain",
    # Pipeline
    "RAGPipeline",
]
