# weeklyHighlights 目录说明

`weeklyHighlights` 是 AI-Compass 里专门承接“增量内容”的目录，适合放每周精选、前沿速览、专题特刊这类持续追加的内容。

如果整个仓库要被 AI 当作知识库稳定使用，这个目录最重要的目标不是“排版好看”，而是让 AI 能稳定回答下面三类问题：

- 最近一周更新了什么
- 某个模型、产品、公司过去几周出现过几次
- 某个热点只是周报里提到过，还是已经在仓库其他专题里形成长期沉淀

## 为什么当前目录还不够“AI 友好”

当前历史文件主要是 `1.md`、`2.md` 这样的编号文件。这种方式对人工维护很省事，但对 AI 检索会有几个天然短板：

- 文件名不携带时间、主题、实体信息
- 不同周报的标题和正文结构不完全统一
- 缺少统一元数据，AI 很难稳定区分“最新内容”“历史内容”“专题内容”
- 根 README 里虽然有往期列表，但缺少一个专门面向检索的增量索引

## 当前推荐做法

为了兼容已有链接，当前建议采用“保留历史文件名 + 增加索引层”的方式，而不是立刻大规模改名：

- 历史文件继续保留 `1.md`、`2.md` 这类编号文件
- 新增或更新周报时，同步更新 [`latest.md`](./latest.md)
- 新增或更新周报时，同步更新 [`INDEX.md`](./INDEX.md)
- 新写的周报尽量按 [`TEMPLATE.md`](./TEMPLATE.md) 补齐元数据和摘要

这样做的好处是：

- 不破坏 README 和外部平台已经存在的历史链接
- AI 有了稳定的“入口文件”可优先读取
- 后续要不要迁移成日期命名，可以以后再做，不影响当前使用

## 增量内容的最小维护动作

每次新增一期周报，建议至少做这 4 件事：

1. 新建周报文件，正文按 [`TEMPLATE.md`](./TEMPLATE.md) 填写
2. 更新 [`latest.md`](./latest.md) 中的最新期数、标题、路径和日期
3. 在 [`INDEX.md`](./INDEX.md) 中新增一条目录记录
4. 如果周报里出现了仓库已有专题，补一个 `related` 或“延伸阅读”链接

## 推荐元数据字段

建议每篇新增周报都在文件最前面补一个 YAML front matter，至少包含下面这些字段：

```yaml
id:
type:
issue:
published_at:
title:
summary:
tags:
entities:
aliases:
related:
supersedes:
last_updated:
```

这些字段里，最有价值的是：

- `published_at`：让 AI 能按时间线回答“最近/上周/过去几周”
- `summary`：让 AI 不必通读整篇也能先抓住本期重点
- `tags`：让 AI 更容易做主题聚类，比如 `agent`、`multimodal`、`video-generation`
- `entities`：让 AI 更容易追踪模型、产品、公司、框架
- `related`：让 AI 可以把周报和长期专题串起来

## AI 检索时的推荐入口顺序

如果你在本地让 AI 回答问题，建议让它按这个顺序读取：

1. 根目录的 [`README.md`](../README.md)
2. 本目录的 [`latest.md`](./latest.md)
3. 本目录的 [`INDEX.md`](./INDEX.md)
4. 命中的具体周报文件
5. 对应的专题长期文档，例如 `3.*`、`1.*`、`8.*` 等

## 未来如果想进一步升级目录结构

如果后面周报越来越多，可以再从“编号文件”演进到“按年归档”的结构。建议目标形态如下：

```text
weeklyHighlights/
  README.md
  INDEX.md
  latest.md
  TEMPLATE.md
  2026/
    2026-W13-issue-30.md
    2026-W14-issue-31.md
```

这个升级不是现在必须做的。只要 `latest.md`、`INDEX.md` 和元数据先建立起来，仓库已经能明显更适合作为 AI 的本地知识库。
