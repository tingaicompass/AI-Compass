"""
RAG 系统测试脚本

测试所有模块功能，包括:
- 索引构建
- 混合检索
- 精排
- 问答

使用方式:
    # 测试全部功能
    python test.py

    # 只测试索引构建
    python test.py --test build
    
    # 只测试召回
    python test.py --test retrieval
    
    # 只测试召回
    python test.py --test rerank

    # 只测试问答
    python test.py --test query
    
    
    

    # 指定配置
    python test.py --zilliz-endpoint "https://xxx" --zilliz-key "xxx"
"""

import argparse
import json
import logging
import sys
from pathlib import Path

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# 添加当前目录到 path，支持直接运行
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))

# 直接导入本地模块，避免触发整个项目的导入链
from config import update_config, get_config
from rag_pipeline import RAGPipeline


def load_test_cases() -> dict:
    """加载测试用例"""
    test_file = current_dir / "test_cases.json"
    with open(test_file, "r", encoding="utf-8") as f:
        return json.load(f)


def test_build_index(pipeline: RAGPipeline, test_data: dict) -> bool:
    """
    测试索引构建

    Args:
        pipeline: RAG Pipeline 实例
        test_data: 测试数据

    Returns:
        bool: 测试是否通过
    """
    print("\n" + "="*60)
    print("测试索引构建")
    print("="*60)

    documents = test_data["documents"]
    print(f"文档数量: {len(documents)}")

    # 构建索引
    result = pipeline.build_index(
        documents=[{"text": doc["text"]} for doc in documents],
        drop_if_exists=True
    )

    if result["success"]:
        print(f"✅ 索引构建成功，插入 {result['count']} 条文档")
        return True
    else:
        print(f"❌ 索引构建失败: {result.get('error', 'Unknown error')}")
        return False


def test_retrieval(pipeline: RAGPipeline, test_data: dict) -> bool:
    """
    测试检索功能

    Args:
        pipeline: RAG Pipeline 实例
        test_data: 测试数据

    Returns:
        bool: 测试是否通过
    """
    print("\n" + "="*60)
    print("测试检索功能")
    print("="*60)

    test_questions = test_data.get("test_questions", [])[:3]  # 只测试前3个

    all_passed = True
    for q in test_questions:
        question = q["question"]
        print(f"\n问题: {question}")

        # 执行检索
        results = pipeline.retriever.retrieve(question, top_k=5, use_rewrite=False)

        if results:
            print(f"✅ 检索到 {len(results)} 条结果")
            for i, r in enumerate(results[:3], 1):
                print(f"  [{i}] 分数: {r['distance']:.4f} | 内容: {r['text'][:50]}...")
        else:
            print("❌ 未检索到结果")
            all_passed = False

    return all_passed


def test_rerank(pipeline: RAGPipeline, test_data: dict) -> bool:
    """
    测试精排功能

    Args:
        pipeline: RAG Pipeline 实例
        test_data: 测试数据

    Returns:
        bool: 测试是否通过
    """
    print("\n" + "="*60)
    print("测试精排功能")
    print("="*60)

    question = "什么是深度学习？"
    documents = [doc["text"] for doc in test_data["documents"][:10]]

    print(f"问题: {question}")
    print(f"待排序文档数: {len(documents)}")

    try:
        results = pipeline.reranker.rerank(
            query=question,
            documents=documents,
            top_k=3
        )

        if results:
            print(f"✅ 精排成功，返回 {len(results)} 条结果")
            for i, r in enumerate(results, 1):
                doc_preview = documents[r['index']][:50]
                print(f"  [{i}] 分数: {r['relevance_score']:.4f} | 内容: {doc_preview}...")
            return True
        else:
            print("❌ 精排返回空结果")
            return False

    except Exception as e:
        print(f"❌ 精排失败: {e}")
        return False


def test_query(pipeline: RAGPipeline, test_data: dict = None) -> bool:
    """
    测试完整问答流程

    Args:
        pipeline: RAG Pipeline 实例
        test_data: 测试数据（未使用，保持接口一致）

    Returns:
        bool: 测试是否通过
    """
    _ = test_data  # 保持接口一致
    print("\n" + "="*60)
    print("测试完整问答流程")
    print("="*60)

    test_questions = [
        "什么是 RAG？它有什么作用？",
        "如何解决模型过拟合问题？",
        "Transformer 架构的核心是什么？"
    ]

    all_passed = True
    for question in test_questions:
        print(f"\n问题: {question}")
        print("-"*40)

        result = pipeline.query(
            question=question,
            use_rewrite=False  # 如果没有配置 Gemini，不使用改写
        )

        if result.get("answer") and not result.get("error"):
            print(f"✅ 问答成功")
            print(f"召回数: {result['retrieval_count']}")
            print(f"精排后: {result['rerank_count']}")
            print(f"回答: {result['answer'][:200]}...")
        else:
            print(f"❌ 问答失败: {result.get('error', 'Unknown error')}")
            all_passed = False

    return all_passed


def test_full_pipeline(pipeline: RAGPipeline, test_data: dict = None) -> bool:
    """
    测试完整 Pipeline（包含详细中间结果）

    Args:
        pipeline: RAG Pipeline 实例
        test_data: 测试数据（未使用，保持接口一致）

    Returns:
        bool: 测试是否通过
    """
    _ = test_data  # 保持接口一致
    print("\n" + "="*60)
    print("测试完整 Pipeline（详细模式）")
    print("="*60)

    question = "大语言模型的幻觉问题如何解决？"
    print(f"问题: {question}")

    result = pipeline.query_with_details(
        question=question,
        use_rewrite=False
    )

    print(f"\n改写后的查询: {result.get('rewritten_queries', [question])}")
    print(f"检索结果数: {len(result.get('retrieval_results', []))}")
    print(f"精排结果数: {len(result.get('rerank_results', []))}")

    if result.get("rerank_results"):
        print("\n精排结果:")
        for i, r in enumerate(result["rerank_results"][:3], 1):
            print(f"  [{i}] 分数: {r['relevance_score']:.4f}")

    print(f"\n回答:\n{result.get('answer', 'No answer')}")

    return bool(result.get("answer"))


def run_tests(
    zilliz_endpoint: str = None,
    zilliz_api_key: str = None,
    gemini_api_key: str = None,
    test_type: str = "all"
):
    """
    运行测试

    Args:
        zilliz_endpoint: Zilliz Cloud endpoint
        zilliz_api_key: Zilliz API key
        gemini_api_key: Gemini API key (可选)
        test_type: 测试类型 (all, build, query, retrieval, rerank)
    """
    print("="*60)
    print("Milvus RAG 系统测试")
    print("="*60)

    # 检查必要配置
    config = get_config()

    if zilliz_endpoint:
        update_config(zilliz_endpoint=zilliz_endpoint)
    if zilliz_api_key:
        update_config(zilliz_api_key=zilliz_api_key)
    if gemini_api_key:
        update_config(gemini_api_key=gemini_api_key)

    config = get_config()

    # 验证配置
    if config["zilliz_endpoint"] == "YOUR_ZILLIZ_ENDPOINT":
        print("❌ 错误: 请提供 Zilliz Cloud endpoint")
        print("使用方式: python test.py --zilliz-endpoint 'https://xxx' --zilliz-key 'xxx'")
        return

    print(f"Zilliz Endpoint: {config['zilliz_endpoint']}")
    print(f"Collection: {config['collection_name']}")
    print(f"Query Rewrite: {'启用' if config['gemini_api_key'] != 'YOUR_GEMINI_API_KEY' else '禁用'}")

    # 加载测试数据
    test_data = load_test_cases()
    print(f"\n加载测试数据: {len(test_data['documents'])} 条文档")

    # 初始化 Pipeline
    print("\n初始化 RAG Pipeline...")
    try:
        pipeline = RAGPipeline()
        print("✅ Pipeline 初始化成功")
    except Exception as e:
        print(f"❌ Pipeline 初始化失败: {e}")
        return

    # 运行测试
    results = {}

    if test_type in ["all", "build"]:
        results["build"] = test_build_index(pipeline, test_data)

    if test_type in ["all", "retrieval"]:
        results["retrieval"] = test_retrieval(pipeline, test_data)

    if test_type in ["all", "rerank"]:
        results["rerank"] = test_rerank(pipeline, test_data)

    if test_type in ["all", "query"]:
        results["query"] = test_query(pipeline, test_data)

    if test_type == "all":
        results["full"] = test_full_pipeline(pipeline, test_data)

    # 输出测试结果汇总
    print("\n" + "="*60)
    print("测试结果汇总")
    print("="*60)

    for test_name, passed in results.items():
        status = "✅ 通过" if passed else "❌ 失败"
        print(f"  {test_name}: {status}")

    all_passed = all(results.values())
    print(f"\n总体结果: {'✅ 全部通过' if all_passed else '❌ 存在失败'}")


def main():
    parser = argparse.ArgumentParser(description="Milvus RAG 系统测试")
    parser.add_argument(
        "--zilliz-endpoint",
        type=str,
        help="Zilliz Cloud endpoint URL"
    )
    parser.add_argument(
        "--zilliz-key",
        type=str,
        help="Zilliz API key"
    )
    parser.add_argument(
        "--gemini-key",
        type=str,
        help="Gemini API key (用于 query 改写)"
    )
    parser.add_argument(
        "--test",
        type=str,
        default="all",
        choices=["all", "build", "retrieval", "rerank", "query"],
        help="要运行的测试类型"
    )

    args = parser.parse_args()

    run_tests(
        zilliz_endpoint=args.zilliz_endpoint,
        zilliz_api_key=args.zilliz_key,
        gemini_api_key=args.gemini_key,
        test_type=args.test
    )


if __name__ == "__main__":
    main()
