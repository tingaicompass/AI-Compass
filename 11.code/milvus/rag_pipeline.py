"""
完整 RAG Pipeline 模块

整合所有组件，实现完整的 RAG 问答流程:
1. 构建向量索引
2. Query 改写（可选）
3. 多路召回（向量 + BM25）
4. RRF 融合
5. Cohere 精排
6. LLM 问答

使用方式:
    from rag_pipeline import RAGPipeline

    # 初始化
    pipeline = RAGPipeline(config={
        "zilliz_endpoint": "...",
        "zilliz_api_key": "...",
        ...
    })

    # 构建索引
    pipeline.build_index(documents)

    # 问答
    result = pipeline.query("什么是机器学习？")
"""

import logging
from typing import List, Dict, Any, Optional

try:
    from .config import get_config, update_config
    from .embeddings import OpenAIEmbeddings
    from .vector_store import MilvusVectorStore
    from .query_rewriter import QueryRewriter
    from .retriever import HybridRetriever
    from .reranker import CohereReranker
    from .qa_chain import QAChain
except ImportError:
    from config import get_config, update_config
    from embeddings import OpenAIEmbeddings
    from vector_store import MilvusVectorStore
    from query_rewriter import QueryRewriter
    from retriever import HybridRetriever
    from reranker import CohereReranker
    from qa_chain import QAChain


class RAGPipeline:
    """
    完整的 RAG 问答系统

    整合向量索引、检索、精排、问答等所有组件
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        初始化 RAG Pipeline

        Args:
            config: 配置字典，可选。包含:
                - zilliz_endpoint: Zilliz Cloud endpoint
                - zilliz_api_key: Zilliz API key
                - openai_api_key: OpenAI API key
                - gemini_api_key: Gemini API key
                - cohere_api_key: Cohere API key
                - collection_name: Milvus collection 名称
                - top_k_retrieve: 召回数量
                - top_k_rerank: 精排后返回数量

        Example:
            >>> pipeline = RAGPipeline(config={
            ...     "zilliz_endpoint": "https://xxx.zillizcloud.com",
            ...     "zilliz_api_key": "your_api_key"
            ... })
        """
        # 更新配置
        if config:
            update_config(**config)

        # 获取当前配置
        self.config = get_config()

        # 初始化各组件
        self._init_components()

        logging.info("RAGPipeline 初始化完成")

    def _init_components(self):
        """初始化所有组件"""

        # 1. Embedding 模块
        self.embeddings = OpenAIEmbeddings(
            api_key=self.config["openai_api_key"],
            base_url=self.config["openai_base_url"]
        )

        # 2. 向量库
        self.vector_store = MilvusVectorStore(
            uri=self.config["zilliz_endpoint"],
            token=self.config["zilliz_api_key"],
            collection_name=self.config["collection_name"]
        )

        # 3. Query 改写器（可能没有配置 Gemini API Key）
        self.query_rewriter = None
        if self.config["gemini_api_key"] and self.config["gemini_api_key"] != "YOUR_GEMINI_API_KEY":
            try:
                self.query_rewriter = QueryRewriter(
                    api_key=self.config["gemini_api_key"]
                )
            except Exception as e:
                logging.warning(f"Query 改写器初始化失败: {e}")

        # 4. 混合检索器
        self.retriever = HybridRetriever(
            vector_store=self.vector_store,
            embeddings=self.embeddings,
            query_rewriter=self.query_rewriter
        )

        # 5. 精排器
        self.reranker = CohereReranker(
            api_key=self.config["cohere_api_key"]
        )

        # 6. 问答链
        self.qa_chain = QAChain(
            api_key=self.config["openai_api_key"],
            base_url=self.config["openai_base_url"]
        )

    def build_index(
        self,
        documents: List[Dict[str, str]],
        drop_if_exists: bool = False
    ) -> Dict[str, Any]:
        """
        构建向量索引

        Args:
            documents: 文档列表，每个文档包含:
                - text: str 文档内容
            drop_if_exists: 如果 collection 已存在是否删除重建

        Returns:
            Dict: 构建结果，包含:
                - success: 是否成功
                - count: 插入的文档数量
                - collection_name: collection 名称

        Example:
            >>> docs = [
            ...     {"text": "机器学习是人工智能的一个分支..."},
            ...     {"text": "深度学习使用神经网络..."}
            ... ]
            >>> result = pipeline.build_index(docs)
        """
        logging.info(f"开始构建索引，文档数量: {len(documents)}")

        try:
            # 1. 创建 Collection
            self.vector_store.create_collection(drop_if_exists=drop_if_exists)

            # 2. 生成 embeddings
            texts = [doc["text"] for doc in documents]
            embeddings = self.embeddings.embed_documents_batch(texts)

            # 3. 准备插入数据
            insert_data = []
            for i, doc in enumerate(documents):
                insert_data.append({
                    "text": doc["text"],
                    "dense_vector": embeddings[i]
                })

            # 4. 插入数据
            ids = self.vector_store.insert_documents(insert_data)

            logging.info(f"索引构建完成，插入 {len(ids)} 条文档")

            return {
                "success": True,
                "count": len(ids),
                "collection_name": self.config["collection_name"]
            }

        except Exception as e:
            logging.error(f"索引构建失败: {e}")
            return {
                "success": False,
                "error": str(e)
            }

    def query(
        self,
        question: str,
        top_k_retrieve: Optional[int] = None,
        top_k_rerank: Optional[int] = None,
        use_rewrite: bool = True
    ) -> Dict[str, Any]:
        """
        执行完整的 RAG 问答流程

        流程:
            1. Query 改写（可选）
            2. 多路召回（向量 + BM25）
            3. RRF 融合取 top_k_retrieve
            4. Cohere 精排取 top_k_rerank
            5. LLM 生成回答

        Args:
            question: 用户问题
            top_k_retrieve: 召回数量，默认使用配置值
            top_k_rerank: 精排后返回数量，默认使用配置值
            use_rewrite: 是否使用 Query 改写

        Returns:
            Dict: 问答结果，包含:
                - answer: 回答内容
                - question: 原始问题
                - contexts: 使用的上下文列表
                - retrieval_count: 召回数量
                - rerank_count: 精排后数量

        Example:
            >>> result = pipeline.query("什么是深度学习？")
            >>> print(result["answer"])
        """
        top_k_retrieve = top_k_retrieve or self.config["top_k_retrieve"]
        top_k_rerank = top_k_rerank or self.config["top_k_rerank"]

        logging.info(f"开始 RAG 问答，问题: {question}")

        try:
            # 1. 混合检索
            # 如果没有配置 Query 改写器，use_rewrite 自动设为 False
            effective_use_rewrite = use_rewrite and self.query_rewriter is not None

            retrieval_results = self.retriever.retrieve(
                query=question,
                top_k=top_k_retrieve,
                use_rewrite=effective_use_rewrite
            )

            if not retrieval_results:
                return {
                    "answer": "抱歉，没有找到与您问题相关的信息。",
                    "question": question,
                    "contexts": [],
                    "retrieval_count": 0,
                    "rerank_count": 0
                }

            logging.info(f"召回 {len(retrieval_results)} 条结果")

            # 2. Cohere 精排
            rerank_results = self.reranker.rerank_with_ids(
                query=question,
                documents=retrieval_results,
                top_k=top_k_rerank
            )

            logging.info(f"精排后 {len(rerank_results)} 条结果")

            # 3. 提取上下文
            contexts = [r["text"] for r in rerank_results]

            # 4. LLM 生成回答
            answer = self.qa_chain.answer(
                question=question,
                contexts=contexts
            )

            return {
                "answer": answer,
                "question": question,
                "contexts": contexts,
                "retrieval_count": len(retrieval_results),
                "rerank_count": len(rerank_results),
                "rerank_scores": [r["relevance_score"] for r in rerank_results]
            }

        except Exception as e:
            logging.error(f"RAG 问答失败: {e}")
            return {
                "answer": f"处理问题时发生错误: {e}",
                "question": question,
                "contexts": [],
                "error": str(e)
            }

    def query_with_details(
        self,
        question: str,
        top_k_retrieve: Optional[int] = None,
        top_k_rerank: Optional[int] = None,
        use_rewrite: bool = True
    ) -> Dict[str, Any]:
        """
        执行 RAG 问答并返回详细的中间结果

        Args:
            question: 用户问题
            top_k_retrieve: 召回数量
            top_k_rerank: 精排后返回数量
            use_rewrite: 是否使用 Query 改写

        Returns:
            Dict: 包含所有中间结果的详细信息
        """
        top_k_retrieve = top_k_retrieve or self.config["top_k_retrieve"]
        top_k_rerank = top_k_rerank or self.config["top_k_rerank"]

        result = {
            "question": question,
            "config": {
                "top_k_retrieve": top_k_retrieve,
                "top_k_rerank": top_k_rerank,
                "use_rewrite": use_rewrite
            }
        }

        # 1. Query 改写
        if use_rewrite and self.query_rewriter:
            try:
                rewritten_queries = self.query_rewriter.rewrite(question)
                result["rewritten_queries"] = rewritten_queries
            except Exception as e:
                result["rewrite_error"] = str(e)
                result["rewritten_queries"] = [question]
        else:
            result["rewritten_queries"] = [question]

        # 2. 检索
        retrieval_results = self.retriever.retrieve(
            query=question,
            top_k=top_k_retrieve,
            use_rewrite=use_rewrite and self.query_rewriter is not None
        )
        result["retrieval_results"] = retrieval_results

        if not retrieval_results:
            result["answer"] = "抱歉，没有找到与您问题相关的信息。"
            result["rerank_results"] = []
            result["contexts"] = []
            return result

        # 3. 精排
        rerank_results = self.reranker.rerank_with_ids(
            query=question,
            documents=retrieval_results,
            top_k=top_k_rerank
        )
        result["rerank_results"] = rerank_results

        # 4. 提取上下文
        contexts = [r["text"] for r in rerank_results]
        result["contexts"] = contexts

        # 5. 生成回答
        answer = self.qa_chain.answer(question=question, contexts=contexts)
        result["answer"] = answer

        return result

    def get_stats(self) -> Dict[str, Any]:
        """获取系统统计信息"""
        collection_stats = self.vector_store.get_collection_stats()

        return {
            "collection": collection_stats,
            "config": {
                "embedding_model": self.config["embedding_model"],
                "rerank_model": self.config["rerank_model"],
                "qa_model": self.config["qa_model"],
                "top_k_retrieve": self.config["top_k_retrieve"],
                "top_k_rerank": self.config["top_k_rerank"]
            },
            "components": {
                "query_rewriter_enabled": self.query_rewriter is not None
            }
        }
