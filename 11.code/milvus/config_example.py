"""
配置管理模块
存放各种 API 密钥和 endpoint 配置

使用方式:
    from config import ZILLIZ_ENDPOINT, OPENAI_API_KEY, ...

注意: 这是示例配置文件，请复制为 config.py 并填入真实密钥
"""

import os

# =============================================================================
# Zilliz Cloud 配置 (Milvus 托管服务)
# =============================================================================
ZILLIZ_ENDPOINT = os.getenv("ZILLIZ_ENDPOINT", "YOUR_ZILLIZ_ENDPOINT")
ZILLIZ_API_KEY = os.getenv("ZILLIZ_API_KEY", "YOUR_ZILLIZ_API_KEY")

# =============================================================================
# OpenAI 配置 (用于 embedding 和问答)
# 使用项目内 litellm 代理服务
# =============================================================================
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "YOUR_OPENAI_API_KEY")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "YOUR_OPENAI_BASE_URL")

# =============================================================================
# Gemini 配置 (用于 query 改写)
# =============================================================================
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "YOUR_GEMINI_API_KEY")

# =============================================================================
# Cohere 配置 (用于精排)
# 使用项目内配置的 API Key
# =============================================================================
COHERE_API_KEY = os.getenv("COHERE_API_KEY", "YOUR_COHERE_API_KEY")

# =============================================================================
# 模型配置
# =============================================================================
EMBEDDING_MODEL = "text-embedding-ada-002"      # OpenAI embedding 模型
EMBEDDING_DIM = 1536                            # ada-002 向量维度

REWRITE_MODEL = "gemini-2.0-flash"              # Gemini query 改写模型

RERANK_MODEL = "rerank-multilingual-v3.0"       # Cohere 精排模型

QA_MODEL = "gpt-5-chat-latest"                   # OpenAI 问答模型 (litellm 格式)

# =============================================================================
# 检索配置
# =============================================================================
TOP_K_RETRIEVE = 50     # 多路召回后 RRF 融合取 top 50
TOP_K_RERANK = 10       # Cohere 精排后取 top 10

# =============================================================================
# Milvus Collection 配置
# =============================================================================
COLLECTION_NAME = "ai_knowledge_base"           # 默认 collection 名称
MAX_TEXT_LENGTH = 65535                         # VARCHAR 最大长度


def get_config() -> dict:
    """
    获取所有配置的字典形式

    Returns:
        dict: 包含所有配置项的字典
    """
    return {
        # Zilliz Cloud
        "zilliz_endpoint": ZILLIZ_ENDPOINT,
        "zilliz_api_key": ZILLIZ_API_KEY,

        # OpenAI
        "openai_api_key": OPENAI_API_KEY,
        "openai_base_url": OPENAI_BASE_URL,

        # Gemini
        "gemini_api_key": GEMINI_API_KEY,

        # Cohere
        "cohere_api_key": COHERE_API_KEY,

        # 模型
        "embedding_model": EMBEDDING_MODEL,
        "embedding_dim": EMBEDDING_DIM,
        "rewrite_model": REWRITE_MODEL,
        "rerank_model": RERANK_MODEL,
        "qa_model": QA_MODEL,

        # 检索
        "top_k_retrieve": TOP_K_RETRIEVE,
        "top_k_rerank": TOP_K_RERANK,

        # Milvus
        "collection_name": COLLECTION_NAME,
        "max_text_length": MAX_TEXT_LENGTH,
    }


def update_config(**kwargs) -> None:
    """
    动态更新配置项

    Args:
        **kwargs: 要更新的配置项

    Example:
        update_config(
            zilliz_endpoint="https://xxx.zillizcloud.com",
            openai_api_key="sk-xxx"
        )
    """
    global ZILLIZ_ENDPOINT, ZILLIZ_API_KEY
    global OPENAI_API_KEY, OPENAI_BASE_URL
    global GEMINI_API_KEY, COHERE_API_KEY
    global EMBEDDING_MODEL, REWRITE_MODEL, RERANK_MODEL, QA_MODEL
    global TOP_K_RETRIEVE, TOP_K_RERANK
    global COLLECTION_NAME

    config_map = {
        "zilliz_endpoint": "ZILLIZ_ENDPOINT",
        "zilliz_api_key": "ZILLIZ_API_KEY",
        "openai_api_key": "OPENAI_API_KEY",
        "openai_base_url": "OPENAI_BASE_URL",
        "gemini_api_key": "GEMINI_API_KEY",
        "cohere_api_key": "COHERE_API_KEY",
        "embedding_model": "EMBEDDING_MODEL",
        "rewrite_model": "REWRITE_MODEL",
        "rerank_model": "RERANK_MODEL",
        "qa_model": "QA_MODEL",
        "top_k_retrieve": "TOP_K_RETRIEVE",
        "top_k_rerank": "TOP_K_RERANK",
        "collection_name": "COLLECTION_NAME",
    }

    for key, value in kwargs.items():
        if key in config_map:
            globals()[config_map[key]] = value
