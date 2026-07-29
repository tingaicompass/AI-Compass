---
name: ai-compass
description: Search and answer questions from the local AI-Compass AI knowledge base. Use when the user asks about AI models, tools, agents, RAG, multimodal systems, evaluation, learning paths, project resources, or recent AI developments covered by this repository; distinguish weekly updates from long-term topic knowledge, retrieve only the needed sources, and cite repository paths.
---

# AI-Compass Local Knowledge Base

Treat the directory containing this `SKILL.md` as the repository root. Use AI-Compass as a curated Chinese knowledge base for AI learning, model and tool discovery, technical architecture, engineering examples, and weekly industry developments.

## Knowledge Model

Use the repository as three connected layers instead of one undifferentiated document collection:

| Layer | Primary sources | Use it for |
| --- | --- | --- |
| Repository map | `README.md` | Module boundaries, learning routes, resource discovery, and finding the right subtree |
| Weekly increments | `weeklyHighlights/latest.md`, `weeklyHighlights/INDEX.md`, issue files | Latest developments, historical timelines, and what changed in a given period |
| Long-term knowledge | Numbered topic directories and `11.blog/` / `11.code/` | Stable technical explanations, comparison, implementation references, and learning material |

Never treat a weekly item as a durable consensus without checking its matching topic page. Conversely, do not present a long-term topic page as evidence of the latest news without reading the weekly index and issue.

## Decide the Entry Point

Classify the request before opening files:

| User intent | Read first | Then read |
| --- | --- | --- |
| “最近”“本周”“最新” | `weeklyHighlights/latest.md` | Its linked issue and, when needed, the matching topic page |
| “过去几周”“演进”“提到过几次” | `weeklyHighlights/INDEX.md` | Only the relevant issue files |
| “是什么”“怎么做”“对比” | The matching long-term topic file | Linked resources or weekly files only when recency matters |
| “怎么学”“路线”“从零开始” | `README.md` | The modules appropriate to the user's level and goal |
| “项目结构”“代码怎么运行”“实现细节” | `README.md` then the target `11.code/` or `11.blog/` subtree | The smallest set of source files that answers the question |

If a request combines time-sensitive and foundational questions, answer in two labeled parts: `本周增量` and `长期沉淀`.

## Topic Routing

Search the most likely topic home before using a repository-wide search. These are the preferred entry files:

| Topic | Preferred file |
| --- | --- |
| Prompt engineering | `1.1 Prompt工程/2.Prompt工程.md` |
| Models, capabilities, and model selection | `1.3 LLM合集-语言/1.LLM合集-语言.md` |
| Multimodal, image, video, audio, and OCR | `1.4 LLM合集-多模态/多模态.md` |
| Embeddings, training, inference, deployment, RLHF | `2.0 Embedding模型/2.Embedding模型.md`, `2.1 LLM训练框架/2.大模型训练框架.md`, `2.2 LLM推理框架+部署/2.LLM训练推理加速框架+部署.md`, `2.4 RLHF/2.RLHF.md` |
| Evaluation and benchmarks | `2.3 LLM评估框架/2.LLM模型评估.md` |
| MCP and A2A | `3.0 MCP+A2A/3.MCP+A2A.md` |
| RAG, workflow, knowledge bases | `3.1 RAG+workflow/3.RAG+workflow.md` |
| Agents and tool use | `3.2 Agent/3.Agent.md` |
| Deep research, GraphRAG, NL2SQL | `3.3 DeepSearch/3.DeepSearch.md`, `3.4 GraphRAG/3.GraphRAG.md`, `3.5 NLP2SQL/3.NL2SQL.md` |
| Robotics and embodied AI | `3.7 AI Robot/3.AIRobot.md` |
| AI products and recommended projects | `5.AI产品/5.AI产品.md`, `7.AI项目推荐/7.AI项目推荐.md` |
| Python development and AI databases | `8.0 Python/PY.md`, `8.1 AI数据库/8.AI数据库.md` |
| Machine learning, CV, recommender systems, reinforcement learning, and knowledge graphs | `8.3 ML/ML.md`, `8.4 CV/8.CV.md`, `8.5 RecommenderSystem/推荐系统.md`, `8.6 RL/RL.md`, `8.7 KnowledgeGraph/图谱.md` |
| Courses, learning platforms, academic tools, and interview preparation | `6.AI课程/6.AI课程.md`, `9.学习平台/学习平台.md`, `9.学术工具/学术工具.md`, `9.面试/面试.md` |
| Runnable examples and learning articles | `11.code/`, `11.blog/` |

Use `rg -n` to locate an unfamiliar term after choosing the most likely subtree. Do not load all topic files or every weekly issue just because the repository is local.

## Retrieval Workflow

1. Identify whether the request needs current information, historical comparison, durable explanation, implementation detail, or a learning route.
2. Open the entry file selected above and use its structure, links, and headings to narrow the scope.
3. Read the minimum primary source passages needed to support the answer. Prefer an official link, repository link, or paper already recorded in AI-Compass when the user asks for further reading.
4. Cross-check claims that mix freshness and theory: the weekly issue establishes when it appeared; the topic page establishes how it fits the longer technical landscape.
5. State uncertainty when the repository has no entry, a weekly claim is unverified by longer-term material, or a version/date may have changed since the stored issue.

## Task Playbooks

### Latest AI developments

Read `weeklyHighlights/latest.md`, then the linked issue. Return the issue date and number, rank only the developments relevant to the user's request, and identify the affected audience or use case. Do not summarize all entries unless the user explicitly asks for a full digest.

### Historical comparison

Use `weeklyHighlights/INDEX.md` to identify the relevant issues first. Build a short timeline from the selected issue files, separate repository facts from your interpretation, and explain whether the observed change is a model release, product iteration, benchmark result, or ecosystem shift.

### Technical explanation or recommendation

Start from the routed long-term topic page. Explain the concept, its role in a system, tradeoffs, and suitable scenarios. When recommending a tool or model, give the selection criteria and distinguish documented capabilities from your inference; do not imply endorsement merely because a project is listed.

### Learning route

Use `README.md` to identify the user's current level and goal. Propose a short ordered route through existing modules, state what each step teaches, and avoid presenting the full repository tree as a required curriculum.

### Project or codebase analysis

Read the relevant module overview before source files. Describe purpose, main components, data/control flow, configuration and external dependencies, then point to the smallest runnable or explanatory file set. Never invent setup instructions, APIs, or test results absent from the repository.

## Answer Contract

- Answer in the user's language; preserve technical names and file paths exactly.
- Label time-sensitive findings as `周度增量` and stable explanations as `长期沉淀` whenever both are used.
- Cite repository-relative paths next to factual claims, for example `weeklyHighlights/49.md` or `3.2 Agent/3.Agent.md`.
- Prefer concise synthesis over link dumping. Include the fewest sources that make the answer auditable.
- Clearly distinguish direct repository facts, source-linked claims, and your own synthesis or recommendation.
- Never claim that a topic, model, or feature is current beyond the latest issue date recorded in `weeklyHighlights/latest.md`.

## Failure Handling

- If a term is absent, say that it is not currently recorded in AI-Compass after searching the relevant subtree; do not fabricate an entry.
- If internal sources disagree, report the conflict with paths and dates rather than silently choosing one.
- If the user requests real-time external facts, explain that AI-Compass is a local snapshot, use the latest recorded issue as the repository baseline, and only browse externally when the task separately authorizes it.
