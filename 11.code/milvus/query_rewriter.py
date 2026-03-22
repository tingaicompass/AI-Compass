"""
Query 改写模块

使用 Gemini-2.0-flash 模型对用户查询进行改写和扩展，
以提高召回率和搜索效果

使用方式:
    from query_rewriter import QueryRewriter

    rewriter = QueryRewriter(api_key="your_gemini_api_key")
    queries = rewriter.rewrite("什么是机器学习？")
    # 返回: ["什么是机器学习？", "机器学习的定义和基本概念", ...]
"""

import json
import logging
from typing import List, Optional

import google.generativeai as genai

try:
    from .config import GEMINI_API_KEY, REWRITE_MODEL
except ImportError:
    from config import GEMINI_API_KEY, REWRITE_MODEL


# Query 改写的 Prompt 模板
REWRITE_PROMPT = """你是一个查询改写专家。请将用户的查询进行改写和扩展，生成多个语义相似但表达不同的查询变体。

要求：
1. 保持原始查询的核心语义
2. 使用不同的表达方式
3. 可以添加相关的同义词或近义词
4. 生成 2-3 个改写变体

用户查询: {query}

请以 JSON 格式返回，格式如下：
{{"queries": ["原始查询", "改写1", "改写2"]}}

只返回 JSON，不要有其他内容。"""


class QueryRewriter:
    """
    Query 改写类

    使用 Gemini-2.0-flash 对查询进行改写，
    生成多个语义相似的查询变体，提高召回率
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = REWRITE_MODEL
    ):
        """
        初始化 Query 改写器

        Args:
            api_key: Gemini API 密钥，默认从 config 读取
            model: 使用的模型，默认 gemini-2.0-flash
        """
        self.api_key = api_key or GEMINI_API_KEY
        self.model_name = model

        # 配置 Gemini
        genai.configure(api_key=self.api_key)

        # 初始化模型
        self.model = genai.GenerativeModel(
            model_name=self.model_name,
            generation_config={
                "temperature": 0.3,
                "top_p": 0.8,
                "max_output_tokens": 500,
            }
        )

        logging.info(f"QueryRewriter 初始化完成，模型: {self.model_name}")

    def rewrite(self, query: str, include_original: bool = True) -> List[str]:
        """
        改写查询，生成多个查询变体

        Args:
            query: 原始查询
            include_original: 是否在结果中包含原始查询

        Returns:
            List[str]: 查询变体列表（包含原始查询）

        Example:
            >>> rewriter = QueryRewriter(api_key="xxx")
            >>> queries = rewriter.rewrite("什么是机器学习？")
            >>> print(queries)
            ['什么是机器学习？', '机器学习的定义是什么', '如何理解机器学习这个概念']
        """
        if not query or not query.strip():
            return [query] if include_original else []

        query = query.strip()

        try:
            # 构建 prompt
            prompt = REWRITE_PROMPT.format(query=query)

            # 调用 Gemini 生成改写
            response = self.model.generate_content(prompt)
            response_text = response.text.strip()

            # 解析 JSON 响应
            # 处理可能的 markdown 代码块
            if response_text.startswith("```"):
                # 移除 markdown 代码块标记
                lines = response_text.split("\n")
                response_text = "\n".join(lines[1:-1])

            result = json.loads(response_text)
            queries = result.get("queries", [])

            # 确保原始查询在第一位
            if include_original:
                if query not in queries:
                    queries.insert(0, query)
                elif queries[0] != query:
                    queries.remove(query)
                    queries.insert(0, query)

            logging.info(f"Query 改写完成，生成 {len(queries)} 个变体")
            return queries

        except json.JSONDecodeError as e:
            logging.warning(f"解析改写结果失败: {e}，返回原始查询")
            return [query] if include_original else []

        except Exception as e:
            logging.error(f"Query 改写失败: {e}，返回原始查询")
            return [query] if include_original else []

    def rewrite_batch(self, queries: List[str]) -> List[List[str]]:
        """
        批量改写多个查询

        Args:
            queries: 原始查询列表

        Returns:
            List[List[str]]: 每个查询的改写变体列表
        """
        results = []
        for query in queries:
            rewritten = self.rewrite(query)
            results.append(rewritten)
        return results
