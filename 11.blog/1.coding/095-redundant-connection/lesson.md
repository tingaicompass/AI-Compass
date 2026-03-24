> 想系统提升编程能力、查看更完整的学习路线，欢迎访问 AI Compass：https://github.com/tingaicompass/AI-Compass
> 仓库持续更新刷题题解、Python 基础和 AI 实战内容，适合想高效进阶的你。

# 📖 第95课:冗余连接

> **模块**:图论 | **难度**:Medium ⭐
> **LeetCode 链接**:https://leetcode.cn/problems/redundant-connection/
> **前置知识**:图的基础知识、DFS
> **预计学习时间**:30分钟

---

## 🎯 题目描述

给你一个无向图,图中有n个节点,节点编号从1到n。图最初是一棵树(连通且无环),但后来增加了一条额外的边,使得图中出现了环。

现在需要找出这条额外的边。如果有多个答案,返回输入数组中最后出现的那条边。

**示例:**
```
输入:edges = [[1,2],[1,3],[2,3]]
输出:[2,3]
解释:
  1
 / \
2---3
去掉[2,3]这条边后,图变成树
```

**约束条件:**
- n == edges.length
- 3 <= n <= 1000
- edges[i].length == 2
- 1 <= edges[i][0], edges[i][1] <= n
- edges[i][0] != edges[i][1]
- 图中没有重复的边

---

### 🧪 边界用例(面试必考)

| 用例类型 | 输入 | 期望输出 | 考察点 |
|---------|------|---------|--------|
| 最小输入 | [[1,2],[2,3],[1,3]] | [1,3] | 基本三角环 |
| 直链+环 | [[1,2],[2,3],[3,4],[1,4],[1,5]] | [1,4] | 复杂环结构 |
| 星形+环 | [[1,2],[1,3],[1,4],[2,3]] | [2,3] | 星形图加环 |
| 大规模 | n=1000个节点 | — | 性能边界 |

---

## 💡 思路引导

### 生活化比喻

> 想象你在建一个微信群聊网络,每个人(节点)可以和其他人建立好友关系(边)。
>
> 🐌 **笨办法**:每次加入一条新的好友关系后,都遍历整个网络看看是否形成了"传话环路"(比如A→B→C→A)。这样每条边都要检查一遍所有已有边,非常慢。
>
> 🚀 **聪明办法**:维护一个"老大哥"系统(并查集)。每个人都记得自己圈子的老大是谁。加好友时,如果发现两个人已经在同一个圈子了(有同一个老大),那这条好友关系就是多余的!这样只需要O(1)级别的查找就能判断。

### 关键洞察

**树有n个节点和n-1条边。当添加第n条边时,一定会形成环,这条边连接的两个节点必然已经联通(有路径相连)。**

---

## 🧠 解题思维链

> 这一节模拟你在面试中"从零开始思考"的过程。

### Step 1:理解题目 → 锁定输入输出

- **输入**:边的列表edges,每条边是[u,v]表示节点u和v相连
- **输出**:返回导致环的最后一条边
- **限制**:按输入顺序添加边,返回最后出现的那条冗余边

### Step 2:先想笨办法(暴力法)

每加入一条新边后,用DFS/BFS检查图中是否出现了环:
- 如果发现环,说明这条边是冗余的
- 时间复杂度:O(n²) - 每条边O(n)次DFS遍历
- 瓶颈在哪:**每次都要重新遍历图来检测环**

### Step 3:瓶颈分析 → 优化方向

暴力法的问题是重复检查已有的连通性:
- 核心问题:"加入边[u,v]时,如何快速判断u和v是否已经连通?"
- 优化思路:"能不能O(1)或O(log n)时间内判断连通性?→用并查集(Union-Find)"

### Step 4:选择武器

- 选用:**并查集(Disjoint Set Union)**
- 理由:并查集专门用于维护动态连通性,支持:
  - `find(x)`:查找x所属集合的代表元 - O(α(n))≈O(1)
  - `union(x,y)`:合并x和y所在集合 - O(α(n))≈O(1)

> 🔑 **模式识别提示**:当题目出现"判断连通性"、"合并集合"、"检测环",优先考虑"并查集"

---

## 🔑 解法一:DFS检测环(直觉法)

### 思路

按顺序添加每条边到图中,每次添加后用DFS检查是否形成环。第一次检测到环时,那条边就是答案。

### 图解过程

```
示例:edges = [[1,2],[1,3],[2,3]]

Step 1:添加边[1,2]
  1---2    ✅ 无环

Step 2:添加边[1,3]
  1---2    ✅ 无环
  |
  3

Step 3:添加边[2,3]
  1---2    ❌ 形成环 1-2-3-1
  |\ /|
  | X |
  |/ \|
  3---+

返回[2,3]
```

### Python代码

```python
from typing import List
from collections import defaultdict


def findRedundantConnection_dfs(edges: List[List[int]]) -> List[int]:
    """
    解法一:DFS检测环
    思路:逐条添加边,每次用DFS检查是否有环
    """
    graph = defaultdict(set)

    def has_cycle(u, v):
        """从u开始DFS,看能否在不走回头路的情况下到达v"""
        visited = set()

        def dfs(node):
            if node == v:
                return True
            visited.add(node)
            for neighbor in graph[node]:
                if neighbor not in visited:
                    if dfs(neighbor):
                        return True
            return False

        return dfs(u)

    # 逐条添加边
    for u, v in edges:
        # 如果u和v已经连通,说明这条边是冗余的
        if u in graph and v in graph and has_cycle(u, v):
            return [u, v]
        # 添加这条边
        graph[u].add(v)
        graph[v].add(u)

    return []


# ✅ 测试
print(findRedundantConnection_dfs([[1,2],[1,3],[2,3]]))  # 期望输出:[2,3]
print(findRedundantConnection_dfs([[1,2],[2,3],[3,4],[1,4],[1,5]]))  # 期望输出:[1,4]
```

### 复杂度分析

- **时间复杂度**:O(n²) — 每条边O(n)次DFS遍历检查环
  - 具体地说:如果输入n=1000条边,最坏需要1000×1000=100万次节点访问
- **空间复杂度**:O(n) — 图的邻接表存储 + DFS递归栈

### 优缺点

- ✅ 思路直观,容易理解
- ✅ 不需要额外数据结构知识
- ❌ 时间复杂度较高,大规模数据会超时
- ❌ 每次都要重新DFS,有大量重复计算

---

## 🏆 解法二:并查集(最优解)

### 优化思路

核心观察:如果两个节点已经在同一个连通分量中,再连接它们就会形成环。并查集可以O(1)时间判断两个节点是否连通。

> 💡 **关键想法**:维护一个"代表元"系统,每个节点记录所属集合的老大。加边时,如果两个节点老大相同,说明已经连通,这条边就是答案。

### 图解过程

```
示例:edges = [[1,2],[1,3],[2,3]]

初始:每个节点是独立集合
parent: 1→1, 2→2, 3→3

Step 1:union(1,2) → 合并1和2
parent: 1→1, 2→1, 3→3
集合: {1,2}, {3}

Step 2:union(1,3) → 合并1和3
parent: 1→1, 2→1, 3→1
集合: {1,2,3}

Step 3:union(2,3) → 发现2和3的根都是1
parent: 不变
❌ 2和3已经连通,这是冗余边!
返回[2,3]
```

**并查集路径压缩优化示意:**
```
优化前:           优化后:
  1                  1
 / \               /|\
2   3    →        2 3 4
    |
    4
直接让所有节点指向根,查找更快
```

### Python代码

```python
def findRedundantConnection(edges: List[List[int]]) -> List[int]:
    """
    解法二:并查集(Union-Find)
    思路:动态维护连通性,O(1)判断环
    """
    n = len(edges)
    parent = list(range(n + 1))  # parent[i]=i表示i是独立集合

    def find(x):
        """查找x的根节点(代表元),带路径压缩"""
        if parent[x] != x:
            parent[x] = find(parent[x])  # 路径压缩:直接指向根
        return parent[x]

    def union(x, y):
        """合并x和y所在集合,返回是否成功"""
        root_x = find(x)
        root_y = find(y)
        if root_x == root_y:
            return False  # 已经在同一集合,无法合并
        parent[root_x] = root_y  # 将x的根指向y的根
        return True

    # 逐条添加边
    for u, v in edges:
        if not union(u, v):
            # union失败,说明u和v已经连通
            return [u, v]

    return []


# ✅ 测试
print(findRedundantConnection([[1,2],[1,3],[2,3]]))  # 期望输出:[2,3]
print(findRedundantConnection([[1,2],[2,3],[3,4],[1,4],[1,5]]))  # 期望输出:[1,4]
print(findRedundantConnection([[1,4],[3,4],[1,3],[1,2],[4,5]]))  # 期望输出:[1,3]
```

### 复杂度分析

- **时间复杂度**:O(n·α(n)) ≈ **O(n)** — α(n)是阿克曼函数的反函数,实际应用中≈4
  - 每条边的find和union操作都是O(α(n))≈O(1)
  - n条边总共O(n)时间,这是**理论最优**
- **空间复杂度**:O(n) — 存储parent数组

**为什么是最优解**:
- 时间已经达到O(n)理论下限(至少要遍历所有边)
- 空间O(n)也是必须的(要记录每个节点的状态)
- 并查集是解决动态连通性问题的标准数据结构

---

## 🐍 Pythonic 写法

使用列表推导和更简洁的find实现:

```python
def findRedundantConnection_pythonic(edges: List[List[int]]) -> List[int]:
    """Pythonic写法:简化并查集实现"""
    parent = {i: i for i in range(1, len(edges) + 2)}

    def find(x):
        if parent[x] != x:
            parent[x] = find(parent[x])
        return parent[x]

    for u, v in edges:
        root_u, root_v = find(u), find(v)
        if root_u == root_v:
            return [u, v]
        parent[root_u] = root_v

    return []
```

这个写法将find和union逻辑内联到主循环中,代码更紧凑。

> ⚠️ **面试建议**:先写清晰版本展示思路,再提Pythonic写法展示语言功底。
> 面试官更看重你的**思考过程**,而非代码行数。

---

## 📊 解法对比

| 维度 | 解法一:DFS检测环 | 🏆 解法二:并查集(最优) |
|------|----------------|----------------------|
| 时间复杂度 | O(n²) | **O(n)** ← 时间最优 |
| 空间复杂度 | O(n) | **O(n)** |
| 代码难度 | 简单 | 中等(需理解并查集) |
| 面试推荐 | ⭐ | **⭐⭐⭐** ← 首选 |
| 适用场景 | 小规模图(<100节点) | **通用,大规模高效** |

**为什么并查集是最优解**:
- 时间O(n)已经是理论最优(必须扫描所有边)
- 并查集是动态连通性问题的标准解法
- 面试中图论题常考并查集,必须掌握

**面试建议**:
1. 先用30秒口述DFS思路(O(n²)),表明你能想到基本解法
2. 立即优化到🏆并查集(O(n)),展示对高级数据结构的掌握
3. **重点讲解并查集的核心思想**:"用代表元判断连通性,路径压缩优化查找"
4. 强调为什么这是最优:时间已达O(n)理论下限,是动态连通性标准解法
5. 手动测试边界用例,展示对并查集的深入理解

---

## 🎤 面试现场

> 模拟面试中的完整对话流程,帮你练习"边想边说"。

**面试官**:请你解决一下这道题。

**你**:(审题30秒)好的,这道题要求找出导致环的那条边。让我先想一下...

我的第一个想法是用DFS:每加入一条边后,检查图中是否有环。如果发现环,那条边就是答案。时间复杂度是O(n²),因为每条边都要O(n)时间DFS检查。

不过我们可以用**并查集**优化到O(n)。核心思路是:维护每个节点所属集合的代表元。加边时,如果两个节点已经有相同代表元,说明它们已连通,这条边就是冗余的。并查集的find和union操作都是O(1)级别。

**面试官**:很好,请写一下并查集的代码。

**你**:(边写边说)
```python
# 初始化parent数组,每个节点是独立集合
parent = list(range(n + 1))

# find函数查找根节点,带路径压缩优化
def find(x):
    if parent[x] != x:
        parent[x] = find(parent[x])  # 压缩路径
    return parent[x]

# 遍历每条边
for u, v in edges:
    root_u, root_v = find(u), find(v)
    if root_u == root_v:  # 已经连通
        return [u, v]
    parent[root_u] = root_v  # 合并集合
```

**面试官**:测试一下?

**你**:用示例`[[1,2],[1,3],[2,3]]`走一遍:
1. 边[1,2]:合并集合,parent={1:1, 2:1}
2. 边[1,3]:合并集合,parent={1:1, 2:1, 3:1}
3. 边[2,3]:find(2)=1, find(3)=1,相同!返回[2,3]✅

再测一个边界:`[[1,2],[2,3],[3,1]]`,第三条边形成三角环,返回[3,1]✅

### 高频追问

| 追问 | 应答策略 |
|------|---------|
| "还有更优解吗?" | 时间已经是O(n)最优(必须遍历所有边),空间O(n)也是必需的。如果要优化常数项,可以用**按秩合并**(union by rank)进一步优化,但渐进复杂度不变。 |
| "如果要返回所有冗余边呢?" | 继续遍历所有边,不在第一次发现时return,而是收集所有导致环的边。复杂度不变。 |
| "路径压缩的作用?" | 将查找路径上所有节点直接指向根,下次查找O(1)。例如链1→2→3→4,压缩后都直接指向4。均摊时间O(α(n))≈O(1)。 |
| "能用BFS做吗?" | 可以,逻辑类似DFS,但时间复杂度仍是O(n²),不如并查集高效。 |

---

## 🎓 知识点总结

### Python技巧卡片 🐍

```python
# 技巧1:并查集标准模板
class UnionFind:
    def __init__(self, n):
        self.parent = list(range(n))
        self.rank = [0] * n  # 按秩合并优化

    def find(self, x):
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]

    def union(self, x, y):
        root_x, root_y = self.find(x), self.find(y)
        if root_x == root_y:
            return False
        # 按秩合并:将矮树合并到高树
        if self.rank[root_x] < self.rank[root_y]:
            self.parent[root_x] = root_y
        elif self.rank[root_x] > self.rank[root_y]:
            self.parent[root_y] = root_x
        else:
            self.parent[root_y] = root_x
            self.rank[root_x] += 1
        return True

# 技巧2:字典实现并查集(节点编号不连续时)
parent = {}
def find(x):
    if x not in parent:
        parent[x] = x
    if parent[x] != x:
        parent[x] = find(parent[x])
    return parent[x]
```

### 💡 底层原理(选读)

> **并查集为什么这么快?**
>
> 并查集使用两个优化技巧:
> 1. **路径压缩(Path Compression)**:find操作时,将路径上所有节点直接指向根。这样下次查找O(1)。
> 2. **按秩合并(Union by Rank)**:合并时,总是将矮树接到高树下,避免树退化成链表。
>
> 这两个优化使得m次操作的总时间复杂度为O(m·α(m)),其中α是阿克曼函数的反函数。在实际应用中,α(m)≤4,可以认为是O(1)。
>
> **阿克曼函数增长极慢**:α(10^80)仅约为4,这是宇宙中原子数量级别!所以实际应用中并查集就是O(1)。

### 算法模式卡片 📐

- **模式名称**:并查集(Union-Find / Disjoint Set Union)
- **适用条件**:
  - 需要维护动态连通性(判断两个元素是否在同一集合)
  - 需要合并集合
  - 检测无向图中的环
  - 最小生成树(Kruskal算法)
- **识别关键词**:"连通性"、"合并集合"、"检测环"、"朋友圈"、"网络连接"
- **模板代码**:
```python
class UnionFind:
    def __init__(self, n):
        self.parent = list(range(n))

    def find(self, x):
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]

    def union(self, x, y):
        root_x, root_y = self.find(x), self.find(y)
        if root_x == root_y:
            return False
        self.parent[root_x] = root_y
        return True
```

### 易错点 ⚠️

1. **parent数组初始化错误**
   - ❌ 错误:`parent = [0] * n` (节点编号从1开始会越界)
   - ✅ 正确:`parent = list(range(n+1))`或从0开始编号

2. **忘记路径压缩**
   - ❌ 错误:
   ```python
   def find(x):
       if parent[x] != x:
           return find(parent[x])  # 没有压缩
       return parent[x]
   ```
   - ✅ 正确:
   ```python
   def find(x):
       if parent[x] != x:
           parent[x] = find(parent[x])  # 压缩路径
       return parent[x]
   ```

3. **union返回值混淆**
   - 本题中,union失败(返回False)才是找到答案的时机
   - 理解:union失败说明两个节点已经连通,这条边是冗余的

---

## 🏗️ 工程实战(选读)

> 这个算法思想在真实项目中的应用,让你知道"学了有什么用"。

- **场景1:社交网络**
  - 微信/Facebook的"共同好友"功能用并查集维护用户连通性
  - 快速判断两个用户是否在同一社交圈

- **场景2:网络连接**
  - 局域网中判断两台电脑是否连通
  - 检测网络拓扑中的冗余线路

- **场景3:图像分割**
  - 计算机视觉中,用并查集合并相似像素区域
  - Kruskal最小生成树算法的核心数据结构

---

## 🏋️ 举一反三

完成本课后,试试这些同类题目来巩固知识:

| 题目 | 难度 | 相关知识点 | 提示 |
|------|------|-----------|------|
| LeetCode 547. 省份数量 | Medium | 并查集/DFS | 统计连通分量个数 |
| LeetCode 200. 岛屿数量 | Medium | 并查集/DFS | 网格中的连通分量 |
| LeetCode 1584. 连接所有点的最小费用 | Medium | 并查集+最小生成树 | Kruskal算法 |
| LeetCode 323. 无向图中连通分量的数目 | Medium | 并查集 | 直接应用模板 |

---

## 📝 课后小测

试试这道变体题,不要看答案,自己先想5分钟!

**题目**:给定一个有向图的边列表,判断图中是否有环。

<details>
<summary>💡 提示(实在想不出来再点开)</summary>

有向图检测环要用**拓扑排序**或**DFS标记法**(白灰黑标记),不能直接用并查集(并查集只适用于无向图)。

</details>

<details>
<summary>✅ 参考答案</summary>

```python
def hasCycle(n: int, edges: List[List[int]]) -> bool:
    """
    有向图检测环:DFS标记法
    0=白色(未访问), 1=灰色(访问中), 2=黑色(已完成)
    """
    from collections import defaultdict

    graph = defaultdict(list)
    for u, v in edges:
        graph[u].append(v)

    color = [0] * n

    def dfs(node):
        if color[node] == 1:  # 访问到灰色节点,说明有环
            return True
        if color[node] == 2:  # 已完成的节点,无需再访问
            return False

        color[node] = 1  # 标记为访问中
        for neighbor in graph[node]:
            if dfs(neighbor):
                return True
        color[node] = 2  # 标记为已完成
        return False

    for i in range(n):
        if color[i] == 0:
            if dfs(i):
                return True
    return False
```

**核心思路**:DFS过程中,如果访问到"灰色"节点(正在DFS栈中的节点),说明有回边,存在环。这与无向图的并查集检测环不同,有向图需要区分"访问中"和"已完成"状态。

</details>

### 📱 关注微信公众号「汀丶人工智能」
🔥 精选AI前沿资讯 | 📚 深度技术解读 | 💡 实战案例分享

### 🤝 欢迎加入AI Compass知识星球
🎯 **更深入的内容** - 独家教程、项目实战、面试指导  
⚡ **更高的更新频率** - 高频资讯推送、专家答疑、技术交流  
🎁 **限时优惠** - 与数千名AI学习者一起成长！
  * [AI Compass知识星球](https://t.zsxq.com/Tj1eS)
  * [🎫 AI Compass知识星球优惠券](https://github.com/tingaicompass/AI-Compass/blob/main/picture/minor/KnowledgePlanet.md)
>星球支持三天内免费退款，请放心订阅。

<table>
<tr>
<td width="50%" valign="top">

## 💬技术博客
* [CSDN](https://blog.csdn.net/sinat_39620217?type=blog)  
* [掘金](https://juejin.cn/user/4020284493662029)
* [知乎](https://www.zhihu.com/people/tingaicompass)
* [公众号](https://github.com/tingaicompass/AI-Compass/blob/main/picture/main/wx.png)
* [知识星球](https://github.com/tingaicompass/AI-Compass/blob/main/picture/minor/KnowledgePlanet.md)

</td>
<td width="50%" valign="top">

## 📍社交媒体
* [头条📬](https://profile.zjurl.cn/rogue/ugc/profile/?active_tab=dongtai&app_name=news_article&device_id=65&media_id=1719833587832835&request_source=1&share_token=b744b824-20ff-420e-b4f7-6080ad127720&tt_from=copy_link&user_id=3287673762&utm_campaign=client_share&utm_medium=toutiao_android&utm_source=copy_link&version_code=120900&version_name=0)
* [抖音🎶](https://v.douyin.com/ZbvqNyHo61I/)
* [小红书📕](https://www.xiaohongshu.com/user/profile/605c395e000000000100108b?xsec_token=YBq0UxPBd23DZ-rGp87wTY2qVctMuK7wWKQU9LsMEaGnw%3D&xsec_source=app_share&xhsshare=CopyLink&appuid=605c395e000000000100108b&apptime=1752306657&share_id=38c139d8155e4692b37a6316559ae8b3&share_channel=copy_link)

</td>
</tr>
</table>

---

> 如果这篇内容对你有帮助，推荐收藏 AI Compass：https://github.com/tingaicompass/AI-Compass
> 更多系统化题解、编程基础和 AI 学习资料都在这里，后续复习和拓展会更省时间。
