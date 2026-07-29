# AI-Compass 仓库维护指南

## 1. 项目定位与原则

AI-Compass 是以 Markdown 为主体的中文 AI 知识库，包含长期专题、每周精选、学习文章和少量可运行示例，也可安装为 Codex、Claude Code 的本地知识库 Skill。每项修改同时服务读者与 Coding Agent 检索，应遵守：分类准确、来源可追溯、内容可独立阅读、历史内容增量维护。

- 新资源放入最细且已存在的主题模块；单个项目不得新建顶层目录。
- 优先保留官网、GitHub、论文、模型页等一手链接。
- 新增条目应说明“是什么、解决什么、适合谁”，而不是只堆链接。
- 区分项目方提交信息、已公开文档能力和维护者判断；不得臆造基准、集成、许可证、定价或可用性。
- 正文默认使用简体中文；`README-EN.md` 等既有英文文件保持英文。

## 2. 目录地图与信息优先级

| 区域 | 用途 | 关键入口 |
| --- | --- | --- |
| 根 README | 总导航、投稿格式、周报首屏 | `README.md`、`README-EN.md` |
| 本地知识库 Skill | 检索规则与安装 | `SKILL.md`、`Install.md`、`install.sh` |
| 基础知识 | 导航、Prompt、模型、多模态 | `0.AI导航工具集/`、`1.*/` |
| 技术框架 | Embedding、训练、推理、评估、RLHF | `2.*/` |
| 应用实践 | MCP+A2A、RAG、Agent、DeepSearch、GraphRAG、NL2SQL、机器人 | `3.*/` |
| 应用、产品与项目 | 用户应用、产品平台、开源项目推荐 | `4.AI 应用/`、`5.AI产品/`、`7.AI项目推荐/` |
| 学习资源 | 竞赛、课程、Python、数据库、ML、CV、RL、图谱 | `6.*/`、`8.*/` |
| 社区与企业生态 | 社区、学术工具、软件、企业开源 | `9.*/`、`10.*/` |
| 长文与代码 | 博客、课程、可运行示例 | `11.blog/`、`11.code/` |
| 周度增量与图片 | 周报、共享图片资源 | `weeklyHighlights/`、`picture/main/`、`picture/minor/` |

信息冲突时，优先级为：`README.md` 说明当前公开结构；`weeklyHighlights/latest.md` 说明最新期；`weeklyHighlights/INDEX.md` 说明历史时间线；编号专题页说明长期知识；`SKILL.md`、`Install.md` 与 `install.sh` 说明本地知识库行为。来源冲突时报告路径与日期，不得静默覆盖。

## 3. 新资源收录与放置

先在最可能的模块执行 `rg -n -i "名称|别名" <目标文件>` 排重；只有分类不清时才扩大搜索范围。

| 资源类型 | 默认模块 |
| --- | --- |
| Prompt、模型、多模态、评测、训练、推理、RLHF | 匹配的 `1.*`、`2.*` 专题页 |
| Agent、Coding Agent、MCP/A2A、RAG、DeepSearch、GraphRAG、NL2SQL | 匹配的 `3.*` 专题页 |
| 完整终端用户应用，语音/图像/视频/文档工作流，AI 编程产品 | `4.AI 应用/4.AI应用.md` |
| MaaS、企业应用、产品平台与产品分析 | `5.AI产品/5.AI产品.md` |
| 可复用开源项目、框架组件、学习型实现 | `7.AI项目推荐/7.AI项目推荐.md` |
| 课程、竞赛、社区、学术工具、面试资料 | 对应 `6.*` 或 `9.*` 文件 |

向大型模块页新增条目时：更新已有手工目录；插入共享页脚之前；遵循邻近标题层级和分隔线；按来源提供简介、核心功能、已公开的技术原理、应用场景、开源/收费说明及一手链接。项目方自荐且仍早期时，应保留这一上下文。除非改变公开入口，否则不必为每个项目改根 README。

推荐结构：

```markdown
## 项目名 – 一句话定位

#### 简介
#### 核心功能
#### 技术原理
#### 应用场景
#### 开源与使用说明
* 项目官网：
* GitHub 仓库：
```

内容粒度必须与已验证资料相称；未发布路线图不能写成当前能力。

## 4. 周报维护

`weeklyHighlights/` 是增量层，不替代长期专题。新增周报从 `weeklyHighlights/TEMPLATE.md` 创建下一个连续编号文件，并完成所有 front matter：`id`、`issue`、`published_at`、`title`、`summary`、`tags`、`entities`、`aliases`、`related`、`supersedes`、`last_updated`。

每期流程：

1. 每个正文条目标题前创建稳定的 `<a id="item-..."></a>` 锚点。
2. 更新 `weeklyHighlights/latest.md` 的期号、文件、标题和日期。
3. 将该期加入 `weeklyHighlights/INDEX.md`，用于历史检索。
4. 更新 `README.md` 的历史列表与最新期首屏；`README-EN.md` 保持同一最新期、结构和锚点。
5. 仅把具备长期价值的内容同步到最接近的专题页，并保留 `ai-compass-weekly-highlight-sync` 标记、原始期数和来源。

处理聚焦任务时，不要全量阅读周报：最新问题先读 `latest.md`，历史问题先读 `INDEX.md`，再打开相关期数。

## 5. README、英文版与本地 Skill

`README.md` 是信息架构基线，`README-EN.md` 是面向国际开发者的自然英文对应页，不做逐句硬译。

- 中文 README 的安装方式、投稿流程、模块结构、周报首屏、历史列表、公开链接、社区链接、入口改名或删除，必须同步英文版。
- 模块说明、学习路径和项目展示发生实质变化时同步英文版；中文专属推广文案可保留核心含义和目标链接，无需机械翻译。
- 英文标题、徽章和社交标签保持英文；项目名、模型名、URL、锚点和相对路径必须原样保留。
- `SKILL.md` 在知识层级、路由、回答契约或周报规则变化时同步更新。
- `Install.md` 必须与 `install.sh` 一致。安装脚本仅能创建或更新 `~/.codex/skills/ai-compass` 与 `~/.claude/skills/ai-compass` 两个软链接，并拒绝覆盖普通文件或目录；不得削弱这一安全边界、复制整个仓库或提交用户路径。

## 6. Markdown、链接与图片规范

- 标题清晰、层级稳定、段落简短；保留既有 emoji、分隔线、文件名和历史标题，避免无关格式化。
- 仓库链接使用相对路径，含空格目录沿用现有 URL 编码形式；外部链接使用完整 HTTPS URL。
- 有网络时验证新增官方链接。自动请求受反爬限制时保留官方 URL，并在交付中说明限制，不要替换成非官方镜像。
- 图片优先使用 `picture/main/` 或 `picture/minor/` 中已存在的稳定本地资源；不得为装饰加入未经验证的远程图。

## 7. 代码与配置

多数任务是文档维护。可运行示例当前位于 `11.code/milvus/`：Python 遵循 PEP 8、四空格缩进、`lower_snake_case` 命名。`11.code/milvus/config.py` 只能作为本地配置，不得提交 API Key、Endpoint、Token、Cookie 或私有路径；`config_example.py` 必须安全可发布，优先使用环境变量。不要提交 `__pycache__/`、`.pyc`、临时文件或编辑器状态。

## 8. 验证矩阵

按变更风险执行检查，不得声称未执行的验证。

| 变更 | 最低验证 |
| --- | --- |
| 单个 Markdown 资源 | `git diff --check`、排重、验证新增官方链接、检查目录与页脚位置 |
| 模块或 README 导航 | `git diff --check`、验证修改的相对链接、检查标题和锚点 |
| 英文 README 同步 | 对照变化的中英文区块，验证公开链接、周报文件和 `item-*` 锚点 |
| 新周报 | 核对 front matter 与正文，验证 `latest.md`、`INDEX.md`、两份 README、锚点及长期专题同步 |
| Python 示例 | `python3 -m py_compile 11.code/milvus/*.py`；具备本地配置时执行 `cd 11.code/milvus && python3 test.py --test <build|retrieval|rerank|query|full>` |
| Skill 或安装变更 | 联读 `SKILL.md`、`Install.md`、`install.sh`，验证命令和软链接安全约束 |

常用只读命令：

```bash
rg --files
rg -n -i "关键词" <可能的模块>
git diff --check
git diff -- <路径>
git status --short
```

## 9. Git 协作与交付

- 工作区可能已有用户改动；必须保留并与当前内容协作，不得回滚。
- 除非用户明确要求，不得使用 `git reset --hard`、`git checkout --` 等破坏性操作。
- 提交保持聚焦；文档常用 `docs:` 前缀，例如 `docs: update Agent resources`、`docs: sync weekly highlight 49`。
- PR 或交付说明应写清用户可见变化、主要路径、实际验证、来源链接，以及未解决的外部服务或链接限制。
- 代码评审优先检查错误事实、失效路径、元数据漂移、公开链接回归和遗漏的文档同步，而非纯文风偏好。

## 10. 交付前检查

- 资源或功能放在正确模块，局部目录已同步。
- 新说法有提交材料或一手来源支撑。
- 链接、图片和锚点有效，或已明确访问限制。
- 影响到周报、README、英文 README、Skill、安装行为时已完成对应同步。
- 未引入凭据、用户路径、临时文件和无关改动。
- 最终说明列出修改文件和实际执行的检查。
