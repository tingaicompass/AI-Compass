"""
OpenAI Embedding 封装模块

使用 OpenAI text-embedding-ada-002 模型生成文本向量
直接使用 openai SDK，简洁清晰

使用方式:
    from embeddings import OpenAIEmbeddings

    embedder = OpenAIEmbeddings(api_key="sk-xxx")
    vector = embedder.embed_query("什么是机器学习？")
    vectors = embedder.embed_documents(["文本1", "文本2"])
"""

import logging
from typing import List, Optional

import openai

try:
    from .config import OPENAI_API_KEY, OPENAI_BASE_URL, EMBEDDING_MODEL
except ImportError:
    from config import OPENAI_API_KEY, OPENAI_BASE_URL, EMBEDDING_MODEL


class OpenAIEmbeddings:
    """
    OpenAI text-embedding-ada-002 封装类

    功能:
        - 单条文本 embedding
        - 批量文本 embedding
        - 自动处理 API 调用和错误
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model: str = EMBEDDING_MODEL
    ):
        """
        初始化 OpenAI Embeddings 客户端

        Args:
            api_key: OpenAI API 密钥，默认从 config 读取
            base_url: API base URL，默认从 config 读取
            model: embedding 模型名称，默认 text-embedding-ada-002
        """
        self.api_key = api_key or OPENAI_API_KEY
        self.base_url = base_url or OPENAI_BASE_URL
        self.model = model

        # 初始化 OpenAI 客户端
        self.client = openai.OpenAI(
            api_key=self.api_key,
            base_url=self.base_url
        )

        logging.info(f"OpenAIEmbeddings 初始化完成，模型: {self.model}")

    def embed_query(self, text: str) -> List[float]:
        """
        为单条查询文本生成 embedding 向量

        Args:
            text: 查询文本

        Returns:
            List[float]: 1536 维向量 (ada-002)

        Example:
            >>> embedder = OpenAIEmbeddings(api_key="sk-xxx")
            >>> vector = embedder.embed_query("什么是深度学习？")
            >>> len(vector)
            1536
        """
        if not text or not text.strip():
            raise ValueError("输入文本不能为空")

        try:
            response = self.client.embeddings.create(
                model=self.model,
                input=text.strip()
            )
            return response.data[0].embedding

        except openai.APIError as e:
            logging.error(f"OpenAI API 调用失败: {e}")
            raise

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        为多条文档文本生成 embedding 向量

        Args:
            texts: 文档文本列表

        Returns:
            List[List[float]]: embedding 向量列表，每个向量 1536 维

        Example:
            >>> embedder = OpenAIEmbeddings(api_key="sk-xxx")
            >>> vectors = embedder.embed_documents(["文本1", "文本2"])
            >>> len(vectors)
            2
            >>> len(vectors[0])
            1536
        """
        if not texts:
            return []

        # 过滤空文本
        cleaned_texts = [t.strip() for t in texts if t and t.strip()]
        if not cleaned_texts:
            return []

        try:
            response = self.client.embeddings.create(
                model=self.model,
                input=cleaned_texts
            )
            # 按原始顺序返回 embedding
            return [item.embedding for item in response.data]

        except openai.APIError as e:
            logging.error(f"OpenAI API 批量调用失败: {e}")
            raise

    def embed_documents_batch(
        self,
        texts: List[str],
        batch_size: int = 100
    ) -> List[List[float]]:
        """
        分批处理大量文档的 embedding

        当文档数量较大时，分批调用 API 避免超时

        Args:
            texts: 文档文本列表
            batch_size: 每批处理的文档数量，默认 100

        Returns:
            List[List[float]]: 所有文档的 embedding 向量列表
        """
        if not texts:
            return []

        all_embeddings = []
        total = len(texts)

        for i in range(0, total, batch_size):
            batch = texts[i:i + batch_size]
            logging.info(f"处理 embedding 批次: {i + 1}-{min(i + batch_size, total)}/{total}")

            batch_embeddings = self.embed_documents(batch)
            all_embeddings.extend(batch_embeddings)

        return all_embeddings
