# Repository Guidelines（仓库贡献指南）

## 项目定位与维护原则

AI-Compass 是以 Markdown 为主体的 AI 知识库，重点沉淀 AI 学习路线、技术专题、每周精选、博客文章和少量可运行工程示例。贡献时优先保持“分类清晰、链接可追溯、内容可复用”的原则：新增内容要能被读者直接阅读，也要便于 Claude Code、Codex 等 AI 工具作为本地知识库检索和引用。

## 项目结构与模块组织

顶层编号目录按长期主题划分，例如 `1.1 Prompt工程/`、`2.2 LLM推理框架+部署/`、`3.2 Agent/`、`8.0 Python/`、`10.腾讯/`。新增专题页应放入最接近的既有分类；只有确实出现全新长期主题时，才新增顶层目录。

`weeklyHighlights/` 存放每周精选。新增一期时，从 `weeklyHighlights/TEMPLATE.md` 复制结构，创建下一个编号文件，并同步维护 `weeklyHighlights/INDEX.md`、`weeklyHighlights/latest.md` 和 `README.md` 中的周报入口。`11.blog/` 存放体系化文章和课程内容；`11.code/` 存放可运行示例，当前核心示例是 `11.code/milvus/` 下的 Milvus/Zilliz RAG Demo。仓库图片资源放在 `picture/main/` 或 `picture/minor/`，避免散落到正文目录。

## 内容编写与资源整理规范

正文优先使用简体中文，已有英文文件如 `README-EN.md` 保持英文。Markdown 使用清晰的 `##`、`###` 层级，段落尽量短，避免整页长段堆叠。新增工具、模型或框架时，建议包含简介、核心能力、适用场景、相关链接；如果内容来自周报，也要能独立回答“它是什么、解决什么问题、适合谁看”。

仓库内部链接使用相对路径，例如 `../3.2 Agent/3.Agent.md`。外部链接保留完整 URL，并尽量链接到官网、GitHub、论文或可信来源。图片优先使用仓库本地资源；如使用远程图片，应确认链接稳定且不含私密信息。

## 构建、测试与本地开发命令

本仓库没有统一构建系统，多数变更是文档维护。

- `rg --files`：快速查看仓库文件，确认新增内容应放在哪里。
- `rg -n "关键词" .`：添加资源前搜索既有条目，避免重复收录。
- `python3 -m py_compile 11.code/milvus/*.py`：检查 Python 示例语法。
- `cd 11.code/milvus && python3 test.py --test retrieval`：配置本地 `config.py` 后运行单一路径测试。

## 代码风格与命名约定

Markdown 文件名应延续当前目录风格：专题目录多使用中文标题，算法课程使用 `001-two-sum/lesson.md` 这类编号格式，周报使用连续编号文件。新增内容不要随意改动大量历史标题，避免破坏已有链接。

Python 代码遵循 PEP 8，使用 4 空格缩进；模块名保持小写加下划线，例如 `query_rewriter.py`；函数和变量名应表达业务含义。可读性优先于炫技，只有复杂流程才补充简短注释。不要提交 `__pycache__/`、`.pyc` 或本地临时配置。

## 测试与变更自查

文档类 PR 至少检查：新增链接能打开，目录索引已同步，README 入口未失效，图片路径能正常渲染。新增周报时，确认 front matter 中的 `id`、`issue`、`published_at`、`title`、`summary`、`tags`、`entities`、`related` 与正文一致，并更新 `latest.md` 指向最新一期。

`11.code/milvus` 的测试依赖外部服务和 API Key。将 `config_example.py` 复制为本地未提交的 `config.py`，优先用环境变量管理密钥，然后按需运行 `python3 test.py --test <build|retrieval|rerank|query|full>`。无法运行外部依赖测试时，请在 PR 中说明原因和已执行的替代检查。

## 提交与 Pull Request 规范

近期提交信息较短，例如 `docs: sync weekly highlight 46`、`Add weekly highlight 43`、`Sync weekly highlight 42`。文档同步建议使用 `docs:` 前缀，并在信息中说明期号、目录或主题，例如 `docs: update Agent resources`。

PR 描述应包含变更摘要、主要路径、验证方式和潜在影响。新增资源时附来源链接；修改图片或 README 首屏展示时附截图；涉及 `11.code/` 时说明运行命令和结果。避免在同一个 PR 中混合大量无关目录重排和内容新增。

## 安全与配置提示

不要提交真实 API Key、Endpoint、Token、Cookie 或私有服务凭据。`11.code/milvus/config.py` 应作为本地配置文件使用，不应进入版本控制。提交前用 `git diff` 检查是否误带密钥、个人路径或临时调试输出。
