"""
问答链模块

使用 GPT-5.1-chat-latest 模型根据检索到的上下文回答问题

使用方式:
    from qa_chain import QAChain

    qa = QAChain(api_key="your_openai_api_key")
    answer = qa.answer("什么是机器学习？", contexts=["机器学习是...", "..."])
"""

import logging
from typing import List, Dict, Any, Optional

import openai

try:
    from .config import OPENAI_API_KEY, OPENAI_BASE_URL, QA_MODEL
except ImportError:
    from config import OPENAI_API_KEY, OPENAI_BASE_URL, QA_MODEL


# 问答 Prompt 模板
QA_PROMPT_TEMPLATE = """你是一个专业的 AI 问答助手。请根据以下提供的参考资料回答用户的问题。

要求：
1. 基于参考资料回答，不要编造信息
2. 如果参考资料无法回答问题，请明确说明
3. 回答要简洁清晰，重点突出
4. 可以适当整合多个参考资料的信息

参考资料：
{contexts}

用户问题：{question}

请回答："""


class QAChain:
    """
    问答链类

    使用 LLM 根据检索到的上下文回答用户问题
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model: str = QA_MODEL
    ):
        """
        初始化问答链

        Args:
            api_key: OpenAI API 密钥，默认从 config 读取
            base_url: API base URL，默认从 config 读取
            model: 使用的模型，默认 gpt-5.1-chat-latest
        """
        self.api_key = api_key or OPENAI_API_KEY
        self.base_url = base_url or OPENAI_BASE_URL
        self.model = model

        # 初始化 OpenAI 客户端
        self.client = openai.OpenAI(
            api_key=self.api_key,
            base_url=self.base_url
        )

        logging.info(f"QAChain 初始化完成，模型: {self.model}")

    def answer(
        self,
        question: str,
        contexts: List[str],
        max_tokens: int = 1024,
        temperature: float = 0.7
    ) -> str:
        """
        根据上下文回答问题

        Args:
            question: 用户问题
            contexts: 参考上下文列表
            max_tokens: 最大输出 token 数
            temperature: 生成温度

        Returns:
            str: 回答内容

        Example:
            >>> qa = QAChain(api_key="xxx")
            >>> answer = qa.answer(
            ...     "什么是深度学习？",
            ...     contexts=["深度学习是机器学习的一个分支..."]
            ... )
            >>> print(answer)
        """
        if not question:
            return "请提供问题内容"

        if not contexts:
            return "抱歉，没有找到相关的参考资料来回答您的问题。"

        # 格式化上下文
        formatted_contexts = self._format_contexts(contexts)

        # 构建 prompt
        prompt = QA_PROMPT_TEMPLATE.format(
            contexts=formatted_contexts,
            question=question
        )

        try:
            # 调用 LLM
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "你是一个专业的 AI 问答助手。"},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=max_tokens,
                temperature=temperature
            )

            answer = response.choices[0].message.content.strip()
            logging.info(f"问答生成完成，回答长度: {len(answer)}")
            return answer

        except openai.APIError as e:
            logging.error(f"OpenAI API 调用失败: {e}")
            raise

    def answer_with_metadata(
        self,
        question: str,
        contexts: List[str],
        max_tokens: int = 1024,
        temperature: float = 0.7
    ) -> Dict[str, Any]:
        """
        回答问题并返回详细元数据

        Args:
            question: 用户问题
            contexts: 参考上下文列表
            max_tokens: 最大输出 token 数
            temperature: 生成温度

        Returns:
            Dict: 包含回答和元数据:
                - answer: 回答内容
                - question: 原始问题
                - contexts_count: 使用的上下文数量
                - model: 使用的模型
        """
        answer = self.answer(question, contexts, max_tokens, temperature)

        return {
            "answer": answer,
            "question": question,
            "contexts_count": len(contexts),
            "model": self.model
        }

    def _format_contexts(self, contexts: List[str], max_length: int = 8000) -> str:
        """
        格式化上下文列表

        Args:
            contexts: 上下文列表
            max_length: 最大总长度（避免超出 token 限制）

        Returns:
            str: 格式化后的上下文字符串
        """
        formatted = []
        total_length = 0

        for i, ctx in enumerate(contexts, 1):
            ctx_text = f"[{i}] {ctx.strip()}"

            # 检查长度限制
            if total_length + len(ctx_text) > max_length:
                logging.warning(f"上下文超出长度限制，截断到 {i-1} 条")
                break

            formatted.append(ctx_text)
            total_length += len(ctx_text)

        return "\n\n".join(formatted)

    def stream_answer(
        self,
        question: str,
        contexts: List[str],
        max_tokens: int = 1024,
        temperature: float = 0.7
    ):
        """
        流式回答问题

        Args:
            question: 用户问题
            contexts: 参考上下文列表
            max_tokens: 最大输出 token 数
            temperature: 生成温度

        Yields:
            str: 回答内容片段
        """
        if not question:
            yield "请提供问题内容"
            return

        if not contexts:
            yield "抱歉，没有找到相关的参考资料来回答您的问题。"
            return

        # 格式化上下文
        formatted_contexts = self._format_contexts(contexts)

        # 构建 prompt
        prompt = QA_PROMPT_TEMPLATE.format(
            contexts=formatted_contexts,
            question=question
        )

        try:
            # 调用 LLM（流式）
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "你是一个专业的 AI 问答助手。"},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=max_tokens,
                temperature=temperature,
                stream=True
            )

            for chunk in response:
                if chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content

        except openai.APIError as e:
            logging.error(f"OpenAI API 调用失败: {e}")
            yield f"生成回答时发生错误: {e}"
