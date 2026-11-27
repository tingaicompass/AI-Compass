# AI-Compass 专题新鲜度审计（2026-11-27）

## 分析结论

长期专题的低新鲜度来自三种问题：路由覆盖不足、语义候选不足与上游内容偏科。单纯提高冷门专题权重会把不相关材料放进去，不能当作解决方案。

改进前，周报路由只覆盖 21 个专题；本次模块清单共 49 个文件，其中原有 28 个不在路由表内。新增 Skills 路由后覆盖 22 个，仍有 27 个需要按模块用途单独评估。经典课程、名录、教程不必按周更新，不能据此认定这 27 个都属于分类缺陷。

最近 60 天（第 59–67 期）共有 275 次条目收录，不等于 275 个独立项目。实际公开路由中，多模态 70、项目推荐 64、产品 47、语言模型 34，合计 215/275（78.2%）。49 个模块里有 30 个的 Git 最近改动距今至少 60 天；这只是文件更新间隔，不等于 30 个外部项目已停更。

### 已复现的分类问题

- 标题关键词每个 +12、正文每个 +2，标题中“开源”和品牌词可压过正文中的具体技术领域。
- `app` 等英文词原先按子串匹配，重叠别名也可能重复计分；URL 路径中的技术词会误作主题线索。
- 原 taxonomy 缺少草稿模型、翻译模型、系统级仿真、Skills 等明确对象线索。
- 模型只能在规则 Top 3 内裁决。正确专题没进候选时，即便使用模型也无法选择它。
- 自然语言排除句原来只做字面匹配，不能表达完整语义否定；本次保留这一限制的明确说明，依然需要模型或人工复核。
- 第 67 期实际采用规则路由，不能将其称为完成了在线模型复核。

### 已实施的改进

1. taxonomy 中新增 `include.focus`，标题主旨命中时只加一次 +16；保留原来的匹配、范围、精选门槛和排除规则。更新时间不参与分类打分。
2. 英文关键词使用边界匹配并支持 Qwen4 等版本后缀；同一位置的嵌套别名只计一次；正文主题评分剥离来源 URL。
3. 补齐明确对象词，并接入现有 `11.Skills/11.skills.md`。同步器模型复核默认值改为此前用户指定的 `deepseek-v4.1-flash-1017`。
4. 每轮同步前运行新鲜度审计：默认 60 天复核、90 天优先补采，回看最近 60 天的源周报。
5. 分开记录 Git 改动日期、周报原始内容日期、主专题正文和关联目录。历史回填保留原发布日期，避免伪装新内容。

技能位于仓库外的两个既有技能目录中，已同步安装；该改进不要求调整根 README 或重写已发布周报。

### 验证与边界

第 60–67 期 216 条既有内容进行新旧规则回放，有 57 条主分类变化；这不是人工标注后的准确率指标，也不能把分布变化当成准确率提升。回放保留完整源条目，未写入正式专题。

已复核的代表性变化：

| 条目 | 原规则 | 新规则 |
| --- | --- | --- |
| DFly2 草稿模型 | 项目推荐 | 推理与部署 |
| Uni3Sim 系统级仿真 | 项目推荐 | 推理与部署 |
| Claude Ship | 语言模型 | Agent |
| Hy MT3 翻译模型 | 项目推荐 | 语言模型 |
| Sage-Interaction | 项目推荐 | 多模态 |
| cinecode 电影制作技能包 | 项目推荐 | Skills |
| GPT Images 2.6 | 语言模型 | 多模态 |
| PhysCapBench | 评估 | 评估，保持正确 |
| Tripo 3D 基座模型 | 多模态 | 多模态，保持正确 |

现有 17 项同步测试、新增 6 项回归测试（包含 11 个标题/对象分类用例）、技能结构校验均已通过。规则回放为离线验证，未调用外网模型，未核实外部项目当前发布状态。

本次只更新分类/审计技能并新增审计报告，已发布的第 59–67 期、专题正文、README、公开路由保持原样；未提交或推送本轮新报告。

### 后续维护顺序

- MCP、GraphRAG、Skills：先阅读下方候选的完整周报和一手来源，确认应为主归档还是关联目录；MCP 仅作为产品接入口时不应变成主专题。
- NL2SQL、DeepSearch、RLHF、AI 数据库、ML、CV：审计窗口内未发现足够可靠且未归档的候选，需定向补采。不能把大模型正文里的“DPO”“调研”“SQL 存储”直接当成新领域资源。
- 未覆盖的课程、排行榜、学术工具、企业生态：根据模块定位评估独立采集入口与更新周期，避免与产品发布页共用一套输入。

以下为机器审计快照；候选只供复核，未自动回填。

扫描 49 个模块文件；路由覆盖 22 个专题，27 个模块不在路由表中。最近 60 天周报含 275 条可回放内容。

Git 日期只代表文件改动；周报日期代表内容来源时间。手工收录、目录变动和旧条目回填均不能证明外部项目有新版本。

| 模块 | Git 最近改动 | 周报最近内容 | 近窗主/关联条数 | 建议 |
| --- | --- | --- | --- | --- |
| 0.AI导航工具集/0.AI导航工具集.md | 2026-11-27 | 无周报归档 | 0/0 | taxonomy_review |
| 1.1 Prompt工程/2.Prompt工程.md | 2026-11-27 | 2026-10-18 | 1/0 | monitor |
| 1.2 LLM测评榜/1.大模型测评榜.md | 2026-03-22 | 无周报归档 | 0/0 | taxonomy_review |
| 1.3 LLM合集-语言/1.LLM合集-语言.md | 2026-11-27 | 2026-11-27 | 34/14 | monitor |
| 1.4 LLM合集-多模态/多模态.md | 2026-11-27 | 2026-11-27 | 70/21 | monitor |
| 10.Datawhale/Datawhale.md | 2026-07-07 | 无周报归档 | 0/0 | taxonomy_review |
| 10.paddle/paddle1.md | 2026-07-07 | 无周报归档 | 0/0 | taxonomy_review |
| 10.paddle/paddle2.md | 2026-07-07 | 无周报归档 | 0/0 | taxonomy_review |
| 10.华为开源/华为.md | 2026-07-07 | 无周报归档 | 0/0 | taxonomy_review |
| 10.腾讯/README.md | 2026-08-16 | 无周报归档 | 0/0 | taxonomy_review |
| 10.腾讯/腾讯.md | 2026-07-07 | 无周报归档 | 0/0 | taxonomy_review |
| 10.阿里开源/阿里.md | 2026-07-07 | 无周报归档 | 0/0 | taxonomy_review |
| 11.Skills/11.skills.md | 2026-11-06 | 无周报归档 | 0/0 | review_existing_sources |
| 2.0 Embedding模型/2.Embedding模型.md | 2026-10-18 | 2026-10-18 | 2/0 | monitor |
| 2.1 LLM训练框架/2.大模型训练框架.md | 2026-11-16 | 2026-10-18 | 3/3 | monitor |
| 2.2 LLM推理框架+部署/2.LLM训练推理加速框架+部署.md | 2026-11-24 | 2026-11-24 | 3/3 | monitor |
| 2.3 LLM评估框架/2.LLM模型评估.md | 2026-11-27 | 2026-11-27 | 12/1 | monitor |
| 2.4 RLHF/2.RLHF.md | 2026-07-07 | 2026-05-06 | 0/0 | targeted_collection |
| 3.0 MCP+A2A/3.MCP+A2A.md | 2026-11-27 | 2026-08-28 | 0/0 | review_existing_sources |
| 3.1 RAG+workflow/3.RAG+workflow.md | 2026-11-27 | 2026-11-27 | 2/0 | monitor |
| 3.2 Agent/3.Agent.md | 2026-11-27 | 2026-11-27 | 20/25 | monitor |
| 3.3 DeepSearch/3.DeepSearch.md | 2026-07-28 | 2026-07-28 | 0/0 | targeted_collection |
| 3.4 GraphRAG/3.GraphRAG.md | 2026-07-07 | 2026-06-08 | 0/0 | review_existing_sources |
| 3.5 NLP2SQL/3.NL2SQL.md | 2026-08-11 | 2026-08-11 | 0/0 | targeted_collection |
| 3.6 AI Popular Framework/3.AI Popular Framework.md | 2026-07-07 | 无周报归档 | 0/0 | taxonomy_review |
| 3.7 AI Robot/3.AIRobot.md | 2026-11-16 | 2026-11-16 | 3/2 | monitor |
| 4.AI 应用/4.AI应用.md | 2026-11-27 | 2026-11-24 | 14/11 | monitor |
| 5.AI产品/5.AI产品.md | 2026-11-27 | 2026-11-27 | 47/0 | monitor |
| 6.AI-LLM比赛/6.AI-LLM比赛.md | 2026-07-07 | 无周报归档 | 0/0 | taxonomy_review |
| 6.AI课程/6.AI课程.md | 2026-07-07 | 无周报归档 | 0/0 | taxonomy_review |
| 7.AI项目推荐/7.AI项目推荐.md | 2026-11-27 | 2026-11-27 | 64/25 | monitor |
| 8.0 Python/PY.md | 2026-07-07 | 无周报归档 | 0/0 | taxonomy_review |
| 8.1 AI数据库/8.AI数据库.md | 2026-07-20 | 2026-07-20 | 0/0 | targeted_collection |
| 8.2 AI可视化/8.AI可视.md | 2026-07-07 | 无周报归档 | 0/0 | taxonomy_review |
| 8.3 ML/ML.md | 2026-08-24 | 2026-08-24 | 0/0 | targeted_collection |
| 8.4 CV/8.CV.md | 2026-07-07 | 无周报归档 | 0/0 | targeted_collection |
| 8.5 RecommenderSystem/推荐系统.md | 2026-07-07 | 无周报归档 | 0/0 | taxonomy_review |
| 8.6 RL/RL.md | 2026-07-07 | 无周报归档 | 0/0 | taxonomy_review |
| 8.7 KnowledgeGraph/图谱.md | 2026-07-07 | 无周报归档 | 0/0 | taxonomy_review |
| 8.7 KnowledgeGraph/图谱2.md | 2026-07-07 | 无周报归档 | 0/0 | taxonomy_review |
| 8.7 KnowledgeGraph/图谱3.md | 2026-07-07 | 无周报归档 | 0/0 | taxonomy_review |
| 9.学习平台/学习平台.md | 2026-11-27 | 无周报归档 | 0/0 | taxonomy_review |
| 9.学术工具/学术工具.md | 2026-07-07 | 无周报归档 | 0/0 | taxonomy_review |
| 9.文章集/文章集.md | 2026-11-06 | 无周报归档 | 0/0 | taxonomy_review |
| 9.社区论坛/Docker hub镜像加速.md | 2026-03-22 | 无周报归档 | 0/0 | taxonomy_review |
| 9.社区论坛/云算力.md | 2026-11-27 | 无周报归档 | 0/0 | taxonomy_review |
| 9.社区论坛/社区论坛.md | 2026-07-07 | 无周报归档 | 0/0 | taxonomy_review |
| 9.软件/软件.md | 2026-07-07 | 无周报归档 | 0/0 | taxonomy_review |
| 9.面试/面试.md | 2026-07-07 | 无周报归档 | 0/0 | taxonomy_review |

## 优先复核候选

以下只是语义候选，需要阅读原文确认；本审计不会修改专题或路由。

### 11.Skills/11.skills.md

- 第 66 期，2026-11-24：cinecode – 开源 AI 电影制作技能包，一句话生成完整剧本 （item-228bb027c4，title）
- 第 59 期，2026-10-18：ip-as-logo-skill – 开源的 Agent Skill，能生成 IP 吉祥物头像 （item-df6a5bf3d6，title）

### 3.0 MCP+A2A/3.MCP+A2A.md

- 第 67 期，2026-11-27：PoseWear – 开源的 AI 虚拟试衣工具，开箱即用 （item-3a50fe4320，body_review）
- 第 67 期，2026-11-27：OpenGame – OpenDesign 团队开源的全栈 AI 游戏工作室 （item-622b87602e，body_review）

### 3.4 GraphRAG/3.GraphRAG.md

- 第 67 期，2026-11-27：lumimax – 海马云推出的企业级可信知识库基础设施 （item-e971399a25，body_review）

## 定向补采缺口

没有可靠候选时补采该领域的一手来源，验证发布日期、版本、项目状态与现有收录；不得为更新日期重复收录。

- 2.4 RLHF/2.RLHF.md：rlhf、dpo、reward model、偏好优化、对齐
- 3.3 DeepSearch/3.DeepSearch.md：deepsearch、deep search、deep research、researcher、深度研究、深度搜索、研究智能体、研究助手
- 3.5 NLP2SQL/3.NL2SQL.md：nl2sql、text2sql、text-to-sql、text to sql、自然语言查数、数据库问答、sql agent、数据分析智能体
- 8.1 AI数据库/8.AI数据库.md：vector db、向量数据库、ai database、milvus、qdrant
- 8.3 ML/ML.md：machine learning、机器学习、scikit-learn、xgboost
- 8.4 CV/8.CV.md：computer vision、计算机视觉、yolo、目标检测、图像分割

未纳入 taxonomy 的模块需先判断是否适合周报增量；课程、社区、经典教程与企业生态应单独维护，不按每周更新配额考核。
