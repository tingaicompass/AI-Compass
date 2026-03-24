> 想系统提升编程能力、查看更完整的学习路线，欢迎访问 AI Compass：https://github.com/tingaicompass/AI-Compass
> 仓库持续更新刷题题解、Python 基础和 AI 实战内容，适合想高效进阶的你。

# 📖 第96课:最小高度树

> **模块**:图论 | **难度**:Medium ⭐
> **LeetCode 链接**:https://leetcode.cn/problems/minimum-height-trees/
> **前置知识**:BFS、拓扑排序
> **预计学习时间**:35分钟

---

## 🎯 题目描述

给你一个无向连通树,有n个节点,标记为0到n-1。你需要选择一个节点作为根,使得树的高度最小。

树的高度是根节点到最远叶子节点的最长路径上的边数。

返回所有能使树高度最小的根节点(可能有多个答案)。

**示例:**
```
输入:n = 4, edges = [[1,0],[1,2],[1,3]]
输出:[1]
解释:
   0           1
   |          /|\
   1    vs   0 2 3
  / \
 2   3
以0为根高度=2,以1为根高度=1(最小)
```

**约束条件:**
- 1 <= n <= 2×10^4
- edges.length == n - 1
- 保证输入是一棵树(连通无环)

---

### 🧪 边界用例(面试必考)

| 用例类型 | 输入 | 期望输出 | 考察点 |
|---------|------|---------|--------|
| 单节点 | n=1, edges=[] | [0] | 边界处理 |
| 双节点 | n=2, edges=[[0,1]] | [0,1] | 两个中心 |
| 链状树 | n=6, edges=[[0,1],[1,2],[2,3],[3,4],[4,5]] | [2,3] | 中间两个节点 |
| 星形树 | n=4, edges=[[1,0],[1,2],[1,3]] | [1] | 唯一中心 |
| 大规模 | n=20000 | — | 性能边界 |

---

## 💡 思路引导

### 生活化比喻

> 想象你在组织一个快递配送网络,需要选择一个中心仓库,使得到最远客户的距离最小。
>
> 🐌 **笨办法**:尝试每个位置作为中心,计算到最远点的距离,选最小的。相当于每个节点都跑一遍BFS,时间O(n²)。
>
> 🚀 **聪明办法**:从边缘的客户开始,一层层向内收缩。边缘的客户肯定不适合做中心(到对面边缘太远了)。不断剥掉最外层的"叶子",最后剩下的1-2个节点就是最佳中心!这就像剥洋葱,最里面的核心就是答案。

### 关键洞察

**树的"中心"最多有2个节点,它们位于树的最长路径(直径)的中点。通过逐层剥离叶子节点,最后剩下的就是中心。**

---

## 🧠 解题思维链

> 这一节模拟你在面试中"从零开始思考"的过程。

### Step 1:理解题目 → 锁定输入输出

- **输入**:节点数n和边列表edges,保证是一棵树
- **输出**:返回所有能使树高度最小的根节点列表
- **限制**:n可能很大(20000),需要高效算法

### Step 2:先想笨办法(暴力法)

对每个节点作为根,用BFS计算树的高度:
- 遍历n个节点,每个节点BFS一次
- 时间复杂度:O(n²)
- 瓶颈在哪:**重复计算了大量不必要的信息**

### Step 3:瓶颈分析 → 优化方向

暴力法的问题是:
- 核心问题:"边缘节点(叶子)显然不是最佳根,为什么还要计算它们?"
- 优化思路:"能不能从外向内逐层排除不可能的节点?→逐层剥离叶子"

### Step 4:选择武器

- 选用:**拓扑排序 + BFS层序剥离**
- 理由:
  - 叶子节点(度为1)肯定不是最佳根
  - 逐层剥离叶子,就像剥洋葱,最后剩下的是中心
  - 类似拓扑排序的思想,但从外向内

> 🔑 **模式识别提示**:当题目出现"树的中心"、"最小化到最远点距离",考虑"拓扑排序剥离叶子"

---

## 🔑 解法一:多源BFS暴力(朴素法)

### 思路

对每个节点作为根,用BFS计算树的高度,记录最小高度和对应的根节点。

### 图解过程

```
示例:n=4, edges=[[1,0],[1,2],[1,3]]

测试节点0为根:
    0
    |
    1
   / \
  2   3
高度=2

测试节点1为根:
    1
   /|\
  0 2 3
高度=1 ← 最小

测试节点2为根:
    2
    |
    1
   / \
  0   3
高度=2

测试节点3为根:同理,高度=2

答案:[1]
```

### Python代码

```python
from typing import List
from collections import defaultdict, deque


def findMinHeightTrees_bruteforce(n: int, edges: List[List[int]]) -> List[int]:
    """
    解法一:暴力多源BFS
    思路:每个节点作为根,BFS计算树高度
    """
    if n == 1:
        return [0]

    # 构建邻接表
    graph = defaultdict(list)
    for u, v in edges:
        graph[u].append(v)
        graph[v].append(u)

    def bfs_height(root):
        """从root开始BFS,计算树的高度"""
        visited = {root}
        queue = deque([root])
        height = 0

        while queue:
            size = len(queue)
            for _ in range(size):
                node = queue.popleft()
                for neighbor in graph[node]:
                    if neighbor not in visited:
                        visited.add(neighbor)
                        queue.append(neighbor)
            if queue:  # 还有下一层
                height += 1

        return height

    # 测试每个节点作为根
    min_height = float('inf')
    result = []
    for i in range(n):
        h = bfs_height(i)
        if h < min_height:
            min_height = h
            result = [i]
        elif h == min_height:
            result.append(i)

    return result


# ✅ 测试
print(findMinHeightTrees_bruteforce(4, [[1,0],[1,2],[1,3]]))  # 期望输出:[1]
print(findMinHeightTrees_bruteforce(6, [[3,0],[3,1],[3,2],[3,4],[5,4]]))  # 期望输出:[3,4]
```

### 复杂度分析

- **时间复杂度**:O(n²) — n个节点,每个BFS遍历O(n)
  - 具体地说:如果n=20000,需要约4亿次操作,会超时
- **空间复杂度**:O(n) — 图的邻接表和BFS队列

### 优缺点

- ✅ 思路直观,容易理解
- ✅ 适用于所有图结构
- ❌ 时间复杂度高,大规模数据会超时
- ❌ 做了很多无用功(边缘节点显然不是答案)

---

## 🏆 解法二:拓扑排序剥离叶子(最优解)

### 优化思路

核心观察:
1. **叶子节点(度为1)不可能是最佳根**:它们到对面最远
2. **树的中心在最长路径(直径)的中点**:类似"重心"
3. **从外向内逐层剥离叶子,最后剩下的1-2个节点就是中心**

> 💡 **关键想法**:把树想象成洋葱,一层层剥掉外皮(叶子),中心就露出来了。这类似拓扑排序,但从外向内。

### 图解过程

```
示例:n=6, edges=[[3,0],[3,1],[3,2],[3,4],[5,4]]

初始图(度数标注):
   0(1)   1(1)   2(1)
    \      |     /
       3(4)---4(2)---5(1)

第1轮:剥离度为1的叶子 [0,1,2,5]
剩余: 3---4
度数: 3(1), 4(1)

第2轮:剥离度为1的叶子 [3,4]
剩余: [3,4] ← 这就是答案!

解释:3和4位于树的中心,以它们为根高度都是2(最小)
```

**为什么最后剩1-2个节点?**
```
链状树偶数节点:  0-1-2-3-4-5
剥离: 0,5 → 1,4 → 2,3 ← 剩2个

链状树奇数节点:  0-1-2-3-4
剥离: 0,4 → 1,3 → 2 ← 剩1个
```

### Python代码

```python
def findMinHeightTrees(n: int, edges: List[List[int]]) -> List[int]:
    """
    解法二:拓扑排序剥离叶子
    思路:逐层剥离度为1的叶子,最后剩下的是中心
    """
    # 边界情况
    if n == 1:
        return [0]
    if n == 2:
        return [0, 1]

    # 构建邻接表和度数统计
    graph = defaultdict(set)  # 用set便于删除
    degree = [0] * n
    for u, v in edges:
        graph[u].add(v)
        graph[v].add(u)
        degree[u] += 1
        degree[v] += 1

    # 初始化:找出所有叶子(度为1)
    queue = deque([i for i in range(n) if degree[i] == 1])

    remaining = n  # 剩余节点数
    while remaining > 2:
        size = len(queue)
        remaining -= size

        for _ in range(size):
            leaf = queue.popleft()
            # 删除叶子,更新邻居的度数
            for neighbor in graph[leaf]:
                degree[neighbor] -= 1
                if degree[neighbor] == 1:
                    queue.append(neighbor)

    # 最后剩下的节点就是答案
    return list(queue)


# ✅ 测试
print(findMinHeightTrees(4, [[1,0],[1,2],[1,3]]))  # 期望输出:[1]
print(findMinHeightTrees(6, [[3,0],[3,1],[3,2],[3,4],[5,4]]))  # 期望输出:[3,4]
print(findMinHeightTrees(1, []))  # 期望输出:[0]
print(findMinHeightTrees(2, [[0,1]]))  # 期望输出:[0,1]
```

### 复杂度分析

- **时间复杂度**:O(n) — 每个节点和边只访问一次
  - 每个节点最多入队出队一次
  - 每条边最多被检查两次(从两个端点)
  - 这是**理论最优**
- **空间复杂度**:O(n) — 邻接表和队列

**为什么是最优解**:
- 时间O(n)已经是理论最优(至少要访问所有节点和边)
- 空间O(n)也是必须的(要存储图结构)
- 逐层剥离避免了暴力法的重复计算

---

## 🐍 Pythonic 写法

使用集合推导和更简洁的循环:

```python
def findMinHeightTrees_pythonic(n: int, edges: List[List[int]]) -> List[int]:
    """Pythonic写法:更简洁的实现"""
    if n <= 2:
        return list(range(n))

    # 构建邻接表(用集合便于删除)
    neighbors = [set() for _ in range(n)]
    for u, v in edges:
        neighbors[u].add(v)
        neighbors[v].add(u)

    # 初始化叶子
    leaves = [i for i in range(n) if len(neighbors[i]) == 1]

    remaining = n
    while remaining > 2:
        remaining -= len(leaves)
        new_leaves = []
        for leaf in leaves:
            # 唯一的邻居
            neighbor = neighbors[leaf].pop()
            neighbors[neighbor].remove(leaf)
            if len(neighbors[neighbor]) == 1:
                new_leaves.append(neighbor)
        leaves = new_leaves

    return leaves
```

这个写法用集合直接删除节点,代码更简洁。

> ⚠️ **面试建议**:先写清晰版本展示思路,再提Pythonic写法展示语言功底。
> 面试官更看重你的**思考过程**,而非代码行数。

---

## 📊 解法对比

| 维度 | 解法一:多源BFS | 🏆 解法二:拓扑剥离(最优) |
|------|--------------|------------------------|
| 时间复杂度 | O(n²) | **O(n)** ← 时间最优 |
| 空间复杂度 | O(n) | **O(n)** |
| 代码难度 | 简单 | 中等(需理解拓扑思想) |
| 面试推荐 | ⭐ | **⭐⭐⭐** ← 首选 |
| 适用场景 | 小规模树(<100节点) | **通用,大规模高效** |

**为什么拓扑剥离是最优解**:
- 时间O(n)已经是理论最优(必须访问所有节点)
- 从外向内的思路巧妙避免了重复计算
- 面试中展示了对图论和拓扑排序的深入理解

**面试建议**:
1. 先用30秒口述暴力法思路(O(n²)),表明你能想到基本解法
2. 立即优化到🏆拓扑剥离(O(n)),展示优化能力
3. **重点讲解核心洞察**:"叶子不可能是最佳根,逐层剥离留下中心"
4. 用"剥洋葱"的比喻让面试官快速理解
5. 强调为什么最后剩1-2个节点(树的直径中点)
6. 手动模拟一个示例,展示剥离过程

---

## 🎤 面试现场

> 模拟面试中的完整对话流程,帮你练习"边想边说"。

**面试官**:请你解决一下这道题。

**你**:(审题30秒)好的,这道题要求找出使树高度最小的根节点。让我先想一下...

我的第一个想法是暴力法:对每个节点作为根,用BFS计算树的高度,选最小的。时间复杂度是O(n²),对于n=20000会超时。

不过我们可以用**拓扑排序的思想**优化到O(n)。核心观察是:叶子节点(度为1)显然不是最佳根,因为它们到对面最远。我们可以**从外向内逐层剥离叶子**,就像剥洋葱一样,最后剩下的1-2个节点就是树的中心。

**面试官**:很好,为什么最后剩1-2个节点?

**你**:因为树的中心在最长路径(直径)的中点:
- 如果路径长度是偶数,有2个中心节点
- 如果路径长度是奇数,有1个中心节点

通过逐层剥离叶子,我们最终会收敛到这1-2个中心节点。

**面试官**:请写代码。

**你**:(边写边说)
```python
# 构建邻接表和度数统计
graph = defaultdict(set)
degree = [0] * n
for u, v in edges:
    graph[u].add(v)
    graph[v].add(u)
    degree[u] += 1
    degree[v] += 1

# 初始化所有叶子
queue = deque([i for i in range(n) if degree[i] == 1])

# 逐层剥离
remaining = n
while remaining > 2:
    size = len(queue)
    remaining -= size
    for _ in range(size):
        leaf = queue.popleft()
        for neighbor in graph[leaf]:
            degree[neighbor] -= 1
            if degree[neighbor] == 1:
                queue.append(neighbor)

return list(queue)
```

**面试官**:测试一下?

**你**:用示例`n=6, edges=[[3,0],[3,1],[3,2],[3,4],[5,4]]`走一遍:
```
初始图:
   0(1)   1(1)   2(1)
    \      |     /
       3(4)---4(2)---5(1)

第1轮:剥离叶子 [0,1,2,5]
       3(1)---4(1)

第2轮:剩余 [3,4],remaining=2,停止
返回[3,4]✅
```

### 高频追问

| 追问 | 应答策略 |
|------|---------|
| "为什么一定剩1-2个节点?" | 树的中心在直径中点。直径长度为偶数时有2个中心,奇数时有1个中心。举例:链`0-1-2-3-4`(奇数)剩[2],链`0-1-2-3`(偶数)剩[1,2]。 |
| "能剩3个或更多吗?" | 不可能。如果剩3个或更多,它们还能继续剥离(说明不是中心)。只有当所有剩余节点度数都≤1时停止,此时最多2个。 |
| "还有其他解法吗?" | 可以用**两次BFS求直径**:第一次从任意点找最远点A,第二次从A找最远点B,直径就是AB路径,返回中点。但时间复杂度相同,拓扑剥离更直观。 |
| "如果有多棵树呢?" | 题目保证输入是一棵树。如果有多棵树(森林),对每棵树分别求中心即可。 |

---

## 🎓 知识点总结

### Python技巧卡片 🐍

```python
# 技巧1:用集合作为邻接表(便于删除)
neighbors = [set() for _ in range(n)]
# 删除节点:neighbors[u].remove(v)

# 技巧2:统计度数
degree = [len(neighbors[i]) for i in range(n)]

# 技巧3:列表推导初始化队列
queue = deque([i for i in range(n) if degree[i] == 1])

# 技巧4:边界快速处理
if n <= 2:
    return list(range(n))  # [0] 或 [0,1]
```

### 💡 底层原理(选读)

> **为什么拓扑排序能找到树的中心?**
>
> **定理**:树的中心位于其直径(最长路径)的中点。
>
> **证明思路**:
> 1. 假设直径两端是A和B,长度为d
> 2. 如果选择离A或B太近的节点C作为根,那么到B(或A)的距离会很大
> 3. 只有选择AB路径中点,到两端距离才相等,高度最小
>
> **拓扑剥离为什么能找到中点?**
> - 每次剥离叶子,相当于从直径两端同时向中心收缩
> - 最后剩下的节点就是直径的中点
>
> **为什么最多2个中心?**
> - 直径长度为偶数:中间有2个节点,高度相同
> - 直径长度为奇数:中间有1个节点
>
> **类比**:想象一根绳子,从两端同时烧,最后相遇的地方就是中点。

### 算法模式卡片 📐

- **模式名称**:拓扑排序剥离外层
- **适用条件**:
  - 无向树/图,需要找"中心"节点
  - 从外向内逐层处理
  - 最小化到最远点的距离
- **识别关键词**:"树的中心"、"最小高度"、"重心"、"距离最小化"
- **模板代码**:
```python
def find_center(n, edges):
    # 1. 构建邻接表和度数
    graph = [set() for _ in range(n)]
    for u, v in edges:
        graph[u].add(v)
        graph[v].add(u)

    # 2. 初始化叶子(度为1)
    leaves = [i for i in range(n) if len(graph[i]) <= 1]

    # 3. 逐层剥离
    remaining = n
    while remaining > 2:
        remaining -= len(leaves)
        new_leaves = []
        for leaf in leaves:
            neighbor = graph[leaf].pop()
            graph[neighbor].remove(leaf)
            if len(graph[neighbor]) == 1:
                new_leaves.append(neighbor)
        leaves = new_leaves

    return leaves
```

### 易错点 ⚠️

1. **边界情况处理不当**
   - ❌ 错误:忘记处理n=1或n=2的情况
   - ✅ 正确:
   ```python
   if n == 1:
       return [0]
   if n == 2:
       return [0, 1]
   ```

2. **停止条件错误**
   - ❌ 错误:`while remaining >= 2` (会把最后2个也剥掉)
   - ✅ 正确:`while remaining > 2` (剩2个或更少时停止)

3. **度数更新错误**
   - 删除叶子后,忘记更新邻居的度数
   - 或者更新了度数但没有加入队列

---

## 🏗️ 工程实战(选读)

> 这个算法思想在真实项目中的应用,让你知道"学了有什么用"。

- **场景1:服务器选址**
  - 在分布式系统中选择中心服务器,最小化到各节点的延迟
  - 数据中心选址问题

- **场景2:物流仓储**
  - 在配送网络中选择仓库位置,最小化最远配送距离
  - 快递集散中心选址

- **场景3:社交网络**
  - 找到社交网络中的"意见领袖"(影响力最大的节点)
  - 推荐系统中的中心节点发现

---

## 🏋️ 举一反三

完成本课后,试试这些同类题目来巩固知识:

| 题目 | 难度 | 相关知识点 | 提示 |
|------|------|-----------|------|
| LeetCode 207. 课程表 | Medium | 拓扑排序 | 检测有向图环 |
| LeetCode 210. 课程表II | Medium | 拓扑排序 | 返回拓扑序列 |
| LeetCode 1530. 好叶子节点对的数量 | Medium | DFS+树的直径 | 统计距离≤distance的叶子对 |
| LeetCode 834. 树中距离之和 | Hard | 树形DP | 计算每个节点到其他节点的距离和 |

---

## 📝 课后小测

试试这道变体题,不要看答案,自己先想5分钟!

**题目**:给定一棵树,求树的直径(最长路径的边数)。

<details>
<summary>💡 提示(实在想不出来再点开)</summary>

两次BFS:第一次从任意节点找最远点A,第二次从A找最远点B。AB之间的路径就是直径。

</details>

<details>
<summary>✅ 参考答案</summary>

```python
def treeDiameter(edges: List[List[int]]) -> int:
    """
    树的直径:两次BFS
    第一次:从任意点找最远点A
    第二次:从A找最远点B,距离即为直径
    """
    from collections import defaultdict, deque

    n = len(edges) + 1
    graph = defaultdict(list)
    for u, v in edges:
        graph[u].append(v)
        graph[v].append(u)

    def bfs_farthest(start):
        """从start开始BFS,返回(最远点,最远距离)"""
        visited = {start}
        queue = deque([start])
        distance = 0
        farthest = start

        while queue:
            size = len(queue)
            for _ in range(size):
                node = queue.popleft()
                farthest = node  # 更新最远点
                for neighbor in graph[node]:
                    if neighbor not in visited:
                        visited.add(neighbor)
                        queue.append(neighbor)
            if queue:
                distance += 1

        return farthest, distance

    # 第一次BFS:从0找最远点A
    point_a, _ = bfs_farthest(0)

    # 第二次BFS:从A找最远点B
    _, diameter = bfs_farthest(point_a)

    return diameter


# 测试
print(treeDiameter([[0,1],[1,2],[2,3],[1,4],[4,5]]))  # 输出:4 (路径0-1-4-5或0-1-2-3)
```

**核心思路**:树的直径必定经过某个节点的最长路径。从任意点BFS找到的最远点A,必定是直径的一端。然后从A再BFS找到的最远点B就是直径的另一端。这是一个经典的树的性质。

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
