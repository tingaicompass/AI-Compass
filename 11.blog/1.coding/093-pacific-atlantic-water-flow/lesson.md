> 想系统提升编程能力、查看更完整的学习路线，欢迎访问 AI Compass：https://github.com/tingaicompass/AI-Compass
> 仓库持续更新刷题题解、Python 基础和 AI 实战内容，适合想高效进阶的你。

# 📖 第93课:太平洋大西洋水流问题

> **模块**:图论 | **难度**:Medium ⭐⭐
> **LeetCode 链接**:https://leetcode.cn/problems/pacific-atlantic-water-flow/
> **前置知识**:第89课(岛屿数量)、第90课(腐烂的橘子)
> **预计学习时间**:30分钟

---

## 🎯 题目描述

给定一个 m×n 的矩阵,代表一个陆地的高度图。矩阵的左边界和上边界连接**太平洋**,右边界和下边界连接**大西洋**。

水流只能从高处流向低处或相同高度。找出所有既能流到太平洋又能流到大西洋的坐标。

**示例:**
```
输入:heights = [[1,2,2,3,5],[3,2,3,4,4],[2,4,5,3,1],[6,7,1,4,5],[5,1,1,2,4]]
输出:[[0,4],[1,3],[1,4],[2,2],[3,0],[3,1],[4,0]]
解释:
    太平洋 ~   ~   ~   ~   ~
       ~  1   2   2   3  (5) *
       ~  3   2  (3) (4) (4) *
       ~  2   4  (5)  3   1  *
       ~ (6) (7)  1   4   5  *
       ~ (5)  1   1   2   4  *
          *   *   *   *   * 大西洋
括号中的格子可以同时流向两个大洋
```

**约束条件:**
- 1 <= m, n <= 200
- 0 <= heights[i][j] <= 10^5

---

### 🧪 边界用例(面试必考)

| 用例类型 | 输入 | 期望输出 | 考察点 |
|---------|------|---------|--------|
| 单行单列 | [[1]] | [[0,0]] | 边界处理 |
| 全部可达 | [[1,1],[1,1]] | [[0,0],[0,1],[1,0],[1,1]] | 相同高度 |
| 阶梯型 | [[1,2],[3,4]] | [[1,1]] | 高度递增 |
| 大矩阵 | 200×200 | 视具体高度 | 性能测试 |

---

## 💡 思路引导

### 生活化比喻

> 想象你在一座山峰上倒一杯水,水会顺着山坡流向四周的海洋。
>
> 🐌 **笨办法**:站在每个山顶,用水桶倒水,看看能不能流到太平洋和大西洋。这需要对每个点都做一次"全地图搜索",如果有 m×n 个点,就要搜索 m×n 次,太慢了!
>
> 🚀 **聪明办法**:不如换个思路——从海洋"逆流而上"!我们让水从太平洋往高处爬,标记所有能到达的格子;再让水从大西洋往高处爬,标记所有能到达的格子。最后找出**同时被两次标记**的格子,那就是答案!这样只需要搜索 2 次。

### 关键洞察

**正向想"水往低处流"很难,逆向想"水往高处爬"反而简单!从边界出发逆向搜索,时间复杂度从 O(m²n²) 降到 O(mn)。**

---

## 🧠 解题思维链

> 这一节模拟你在面试中"从零开始思考"的过程。

### Step 1:理解题目 → 锁定输入输出

- **输入**:m×n 的高度矩阵 heights
- **输出**:所有既能流到太平洋又能流到大西洋的坐标列表
- **限制**:水只能从高处流向低处或相同高度(逆向看就是从低处爬向高处)

### Step 2:先想笨办法(暴力法)

对每个格子 (i, j),用 DFS/BFS 看能否到达太平洋和大西洋:
- 如果能到达左边界或上边界 → 能流向太平洋
- 如果能到达右边界或下边界 → 能流向大西洋

- 时间复杂度:O(m² × n²) — 每个格子 O(mn) 次搜索,共 mn 个格子
- 瓶颈在哪:**重复搜索**。相邻格子的路径有大量重叠,却每次都重新搜索一遍。

### Step 3:瓶颈分析 → 优化方向

- 核心问题:从每个点出发搜索,会有大量重复计算
- 优化思路:**逆向思维**——不从每个点搜索到边界,而是从边界搜索到所有能到达的点!
  - 从太平洋边界出发,标记所有"逆流而上"能到达的点
  - 从大西洋边界出发,标记所有"逆流而上"能到达的点
  - 找出同时被标记的点

### Step 4:选择武器

- 选用:**逆向 DFS/BFS**
- 理由:从边界出发只需搜索 2 次(太平洋 1 次 + 大西洋 1 次),相比暴力法的 m×n 次搜索,效率提升巨大!

> 🔑 **模式识别提示**:当题目涉及"从某点出发是否能到达多个目标",优先考虑**逆向搜索**(从目标反推起点)

---

## 🔑 解法一:逆向 DFS(从边界出发)

### 思路

从太平洋和大西洋的边界分别出发,用 DFS"逆流而上"(即从低往高走),标记所有能到达的格子。最后返回同时被两个海洋标记的格子。

### 图解过程

```
示例: heights = [[1,2,2,3,5],
                [3,2,3,4,4],
                [2,4,5,3,1],
                [6,7,1,4,5],
                [5,1,1,2,4]]

Step 1: 从太平洋边界(左边界+上边界)开始 DFS
  起点: (0,0)~(0,4) 和 (0,0)~(4,0)

  P P P P P       P=能到达太平洋的点
  P . . . .       从 (0,0) 高度1 → 可以逆流到高度≥1的邻居
  P . . . .       从 (0,1) 高度2 → 可以逆流到高度≥2的邻居
  P . . . .       ...依次标记
  P . . . .

  最终标记结果(P表示能到太平洋):
  P P P P P
  P P P P P
  P P P P .
  P P . . .
  P . . . .

Step 2: 从大西洋边界(右边界+下边界)开始 DFS
  起点: (0,4)~(4,4) 和 (4,0)~(4,4)

  最终标记结果(A表示能到大西洋):
  . . . . A
  . . . A A
  . . A . .
  A A . . A
  A . . . A

Step 3: 找出同时标记 P 和 A 的格子
  P&A 点: (0,4), (1,3), (1,4), (2,2), (3,0), (3,1), (4,0)
```

### Python代码

```python
from typing import List


def pacificAtlantic(heights: List[List[int]]) -> List[List[int]]:
    """
    解法一:逆向 DFS
    思路:从太平洋和大西洋边界分别出发,逆流而上标记所有可达点
    """
    if not heights or not heights[0]:
        return []

    m, n = len(heights), len(heights[0])

    # 标记能到达太平洋和大西洋的格子
    pacific = [[False] * n for _ in range(m)]
    atlantic = [[False] * n for _ in range(m)]

    def dfs(r: int, c: int, ocean: List[List[bool]], prev_height: int):
        """
        从 (r, c) 开始 DFS,逆流而上标记所有可达点
        prev_height: 来源格子的高度,只能往≥prev_height的方向走
        """
        # 边界检查
        if r < 0 or r >= m or c < 0 or c >= n:
            return
        # 已访问过或高度不够(无法逆流)
        if ocean[r][c] or heights[r][c] < prev_height:
            return

        # 标记当前点可达
        ocean[r][c] = True

        # 向四个方向逆流而上
        for dr, dc in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
            dfs(r + dr, c + dc, ocean, heights[r][c])

    # 从太平洋边界出发(上边界和左边界)
    for i in range(m):
        dfs(i, 0, pacific, heights[i][0])  # 左边界
    for j in range(n):
        dfs(0, j, pacific, heights[0][j])  # 上边界

    # 从大西洋边界出发(下边界和右边界)
    for i in range(m):
        dfs(i, n - 1, atlantic, heights[i][n - 1])  # 右边界
    for j in range(n):
        dfs(m - 1, j, atlantic, heights[m - 1][j])  # 下边界

    # 找出同时能到达两个海洋的点
    result = []
    for i in range(m):
        for j in range(n):
            if pacific[i][j] and atlantic[i][j]:
                result.append([i, j])

    return result


# ✅ 测试
print(pacificAtlantic([[1, 2, 2, 3, 5], [3, 2, 3, 4, 4], [2, 4, 5, 3, 1], [6, 7, 1, 4, 5], [5, 1, 1, 2, 4]]))
# 期望输出:[[0,4],[1,3],[1,4],[2,2],[3,0],[3,1],[4,0]]

print(pacificAtlantic([[1]]))  # 期望输出:[[0,0]]
print(pacificAtlantic([[1, 1], [1, 1]]))  # 期望输出:[[0,0],[0,1],[1,0],[1,1]]
```

### 复杂度分析

- **时间复杂度**:O(m × n) — 每个格子最多被访问 2 次(太平洋 DFS 一次 + 大西洋 DFS 一次)
  - 具体地说:如果矩阵是 200×200,大约需要 200×200×2 = 80,000 次操作
- **空间复杂度**:O(m × n) — 两个标记矩阵 + DFS 递归栈(最坏 O(mn))

### 优缺点

- ✅ 逆向思维巧妙,从 O(m²n²) 优化到 O(mn)
- ✅ 代码清晰,易于理解
- ❌ 使用两个额外矩阵,空间开销较大(可优化为集合)

---

## 🏆 解法二:逆向 BFS(从边界出发,最优解)

### 优化思路

DFS 递归深度可能达到 O(mn),在超大矩阵上可能栈溢出。使用 **BFS 迭代版本**更稳定,且逻辑更清晰。

> 💡 **关键想法**:BFS 用队列代替递归栈,避免栈溢出,且更适合"逐层扩散"的场景。

### 图解过程

```
BFS 从边界出发的扩散过程(以太平洋为例):

初始队列: [(0,0), (0,1), ..., (0,4), (1,0), ..., (4,0)]
           ↓ 逐层扩散
第 1 轮: 处理所有边界点,将可达的邻居加入队列
第 2 轮: 处理新加入的点,继续向更高处扩散
...
直到队列为空(所有可达点都标记完毕)
```

### Python代码

```python
from typing import List
from collections import deque


def pacificAtlanticBFS(heights: List[List[int]]) -> List[List[int]]:
    """
    解法二:逆向 BFS(最优解)
    思路:用 BFS 从边界逆流而上,避免 DFS 栈溢出
    """
    if not heights or not heights[0]:
        return []

    m, n = len(heights), len(heights[0])

    # 用集合记录可达点(比矩阵更节省空间)
    pacific = set()
    atlantic = set()

    def bfs(queue: deque, visited: set):
        """BFS 逆流而上,标记所有可达点"""
        while queue:
            r, c = queue.popleft()
            visited.add((r, c))

            # 向四个方向扩散(只能往高度≥当前高度的方向走)
            for dr, dc in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                nr, nc = r + dr, c + dc
                # 检查边界、是否访问过、高度是否足够
                if (0 <= nr < m and 0 <= nc < n and
                    (nr, nc) not in visited and
                    heights[nr][nc] >= heights[r][c]):
                    queue.append((nr, nc))
                    visited.add((nr, nc))  # 提前标记,避免重复加入队列

    # 初始化太平洋边界队列
    pacific_queue = deque()
    for i in range(m):
        pacific_queue.append((i, 0))  # 左边界
    for j in range(n):
        pacific_queue.append((0, j))  # 上边界

    # 初始化大西洋边界队列
    atlantic_queue = deque()
    for i in range(m):
        atlantic_queue.append((i, n - 1))  # 右边界
    for j in range(n):
        atlantic_queue.append((m - 1, j))  # 下边界

    # 从两个海洋边界分别 BFS
    bfs(pacific_queue, pacific)
    bfs(atlantic_queue, atlantic)

    # 返回交集(同时能到达两个海洋的点)
    return [[r, c] for r, c in pacific & atlantic]


# ✅ 测试
print(pacificAtlanticBFS([[1, 2, 2, 3, 5], [3, 2, 3, 4, 4], [2, 4, 5, 3, 1], [6, 7, 1, 4, 5], [5, 1, 1, 2, 4]]))
# 期望输出:[[0,4],[1,3],[1,4],[2,2],[3,0],[3,1],[4,0]]

print(pacificAtlanticBFS([[1]]))  # 期望输出:[[0,0]]
```

### 复杂度分析

- **时间复杂度**:O(m × n) — 每个格子最多入队一次,出队一次
- **空间复杂度**:O(m × n) — 两个集合 + 队列(队列最大 O(mn))

---

## 🐍 Pythonic 写法

利用 Python 的集合运算简化代码:

```python
# 简化版:用集合交集一行搞定
def pacificAtlanticPythonic(heights: List[List[int]]) -> List[List[int]]:
    if not heights:
        return []
    m, n = len(heights), len(heights[0])

    def dfs_reach(starts):
        """从 starts 出发,返回所有可达点的集合"""
        visited = set(starts)
        stack = list(starts)
        while stack:
            r, c = stack.pop()
            for dr, dc in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                nr, nc = r + dr, c + dc
                if (0 <= nr < m and 0 <= nc < n and
                    (nr, nc) not in visited and
                    heights[nr][nc] >= heights[r][c]):
                    visited.add((nr, nc))
                    stack.append((nr, nc))
        return visited

    # 边界起点
    pacific = [(0, j) for j in range(n)] + [(i, 0) for i in range(1, m)]
    atlantic = [(m - 1, j) for j in range(n)] + [(i, n - 1) for i in range(m - 1)]

    # 集合交集一行搞定
    return [list(p) for p in dfs_reach(pacific) & dfs_reach(atlantic)]
```

> ⚠️ **面试建议**:先写清晰版本展示思路,再提 Pythonic 写法展示语言功底。
> 面试官更看重你的**思考过程**,而非代码行数。

---

## 📊 解法对比

| 维度 | 解法一:逆向 DFS | 🏆 解法二:逆向 BFS(最优) |
|------|--------------|----------------------|
| 时间复杂度 | O(m×n) | **O(m×n)** ← 时间相同 |
| 空间复杂度 | O(m×n) | **O(m×n)** ← 空间相同 |
| 代码难度 | 简单 | 简单 |
| 稳定性 | ⭐⭐(大矩阵可能栈溢出) | **⭐⭐⭐** ← BFS 更稳定 |
| 面试推荐 | ⭐⭐ | **⭐⭐⭐** ← 首选 |
| 适用场景 | 常规矩阵 | **大规模矩阵、生产环境** |

**为什么 BFS 是最优解**:
- 时间和空间复杂度与 DFS 相同,都是 O(mn)
- **BFS 迭代版本避免了递归栈溢出**,在 200×200 等大矩阵上更稳定
- BFS 逻辑更贴合"逐层扩散"的直觉,代码更易维护

**面试建议**:
1. 先用 30 秒说明暴力法(从每个点搜索,O(m²n²))
2. 立即提出逆向优化(从边界搜索,O(mn))
3. **首选 🏆 BFS 实现**,强调其稳定性优势
4. 如果面试官追问递归版本,再补充 DFS 写法

---

## 🎤 面试现场

> 模拟面试中的完整对话流程,帮你练习"边想边说"。

**面试官**:请你解决一下这道题。

**你**:(审题 30 秒)好的,这道题要求找出所有既能流到太平洋又能流到大西洋的格子。我的第一个想法是对每个格子做 DFS,看能否到达两个海洋,时间复杂度是 O(m²n²)。

不过这样会有大量重复搜索。我们可以**逆向思维**:从太平洋和大西洋的边界分别出发,用 BFS"逆流而上"标记所有可达的格子,最后找出同时被两个海洋标记的点。这样只需要搜索 2 次,时间复杂度优化到 O(mn)。

**面试官**:很好,请写一下代码。

**你**:(边写边说)我用两个集合 pacific 和 atlantic 记录可达点,从边界初始化队列,BFS 扩散时只往高度≥当前高度的方向走,最后返回两个集合的交集。

**面试官**:测试一下?

**你**:用示例 [[1,2,2,3,5],...] 走一遍。从太平洋边界(上和左)出发,能标记大部分左上区域;从大西洋边界(下和右)出发,能标记大部分右下区域。两者交集正好是题目要求的 7 个点。再测一个边界情况 [[1]],单格子既在太平洋边界又在大西洋边界,返回 [[0,0]] 正确。

### 高频追问

| 追问 | 应答策略 |
|------|---------|
| "还有更优解吗?" | 时间已经是 O(mn) 最优(至少要遍历所有格子),空间也很难继续优化。可以讨论是否有并行优化的可能。 |
| "为什么用 BFS 而不是 DFS?" | BFS 迭代版本避免递归栈溢出,在大矩阵(200×200)上更稳定。DFS 递归版本代码更简洁,但可能栈溢出。 |
| "如果有 3 个或 4 个海洋?" | 同样思路:从每个海洋边界分别 BFS,最后求所有集合的交集。时间复杂度仍是 O(k×mn),k 为海洋数量。 |
| "能否原地修改矩阵节省空间?" | 可以用负数或特殊值标记访问过的点,但会破坏原数据,一般不推荐。实际项目中维护独立的 visited 集合更安全。 |

---

## 🎓 知识点总结

### Python技巧卡片 🐍

```python
# 技巧1:集合交集 — 找两个集合的公共元素
set1 = {(0, 0), (0, 1), (1, 0)}
set2 = {(0, 1), (1, 0), (1, 1)}
common = set1 & set2  # 交集: {(0, 1), (1, 0)}

# 技巧2:提前标记避免重复入队 — BFS 优化
visited.add((nr, nc))  # 入队时立即标记
queue.append((nr, nc))
# 而不是出队时才标记,否则同一个点可能被重复加入队列

# 技巧3:列表推导式 + 集合运算
result = [[r, c] for r, c in pacific & atlantic]
```

### 💡 底层原理(选读)

> **为什么逆向搜索更高效?**
>
> - **正向思维**:从每个点出发搜索到边界,需要 m×n 次完整搜索,复杂度 O(m²n²)
> - **逆向思维**:从边界出发"逆流而上",只需 2 次搜索(太平洋 + 大西洋),复杂度 O(mn)
>
> 这是一个经典的**多源问题转单源问题**的优化思路:
> - 多源 → 单源:将所有边界点作为一个"超级源点",一次 BFS 就能标记所有可达点
> - 类似问题:多源最短路、多起点 BFS 等

### 算法模式卡片 📐

- **模式名称**:逆向 DFS/BFS(从目标反推起点)
- **适用条件**:
  - 需要判断多个起点能否到达某个/某些目标
  - 正向搜索代价过高(需要多次完整搜索)
  - 目标点数量较少或集中(如边界)
- **识别关键词**:"能否从 A 到达 B"、"多个起点"、"边界"、"水流/传播"
- **模板代码**:
```python
# 逆向 BFS 通用模板
def reverse_bfs(grid, targets):
    """
    从目标集合 targets 反向 BFS,标记所有能到达的点
    """
    visited = set(targets)
    queue = deque(targets)

    while queue:
        x, y = queue.popleft()
        for nx, ny in get_neighbors(x, y):
            if (nx, ny) not in visited and can_reach(nx, ny, x, y):
                visited.add((nx, ny))
                queue.append((nx, ny))

    return visited
```

### 易错点 ⚠️

1. **忘记提前标记导致重复入队**
   - 错误写法:出队时才标记 `visited.add((r,c))`
   - 正确写法:入队时立即标记,避免同一点多次入队导致超时

2. **边界初始化遗漏**
   - 错误:只加左边界和上边界,漏掉角点
   - 正确:左边界 `range(m)` + 上边界 `range(n)` 会自动包含 (0,0)

3. **逆流条件写错**
   - 错误:`heights[nr][nc] > heights[r][c]`(只能往更高处走)
   - 正确:`heights[nr][nc] >= heights[r][c]`(相同高度也能走)

---

## 🏗️ 工程实战(选读)

> 这个算法思想在真实项目中的应用,让你知道"学了有什么用"。

- **场景1:地形分析系统**
  - 气象部门预测洪水覆盖范围:从河流边界逆向扩散,标记所有低于警戒水位的区域

- **场景2:网络可达性分析**
  - 云服务商分析网络拓扑:从边界路由器反向探测,找出所有能访问公网的内网节点

---

## 🏋️ 举一反三

完成本课后,试试这些同类题目来巩固知识:

| 题目 | 难度 | 相关知识点 | 提示 |
|------|------|-----------|------|
| LeetCode 130. 被围绕的区域 | Medium | 边界 DFS/BFS | 从边界的 'O' 出发标记,剩下的 'O' 就是被围绕的 |
| LeetCode 542. 01矩阵 | Medium | 多源 BFS | 从所有 0 出发 BFS,更新到最近 0 的距离 |
| LeetCode 1162. 地图分析 | Medium | 多源 BFS | 从所有陆地出发 BFS,找最远的海洋 |

---

## 📝 课后小测

试试这道变体题,不要看答案,自己先想 5 分钟!

**题目**:给定一个矩阵,代表不同高度的陆地。现在下雨了,水从任意一点都可能溢出到四周相邻的更低处。找出所有"汇水点"——水流到这个点后无法再流向更低处(即该点高度 ≤ 四周所有相邻点)。

<details>
<summary>💡 提示(实在想不出来再点开)</summary>

汇水点就是"局部最低点"。遍历矩阵,检查每个点是否 ≤ 四周所有邻居即可。

</details>

<details>
<summary>✅ 参考答案</summary>

```python
def find_sink_points(heights: List[List[int]]) -> List[List[int]]:
    """找出所有汇水点(局部最低点)"""
    m, n = len(heights), len(heights[0])
    result = []

    for i in range(m):
        for j in range(n):
            is_sink = True
            # 检查四个方向的邻居
            for di, dj in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                ni, nj = i + di, j + dj
                if 0 <= ni < m and 0 <= nj < n:
                    # 如果有任何邻居更低,当前点就不是汇水点
                    if heights[ni][nj] < heights[i][j]:
                        is_sink = False
                        break

            if is_sink:
                result.append([i, j])

    return result

# 测试
print(find_sink_points([[3, 2, 1], [2, 1, 0], [1, 0, 1]]))
# 期望输出:[[1,2]] (高度为0的点是最低点)
```

核心思路:遍历每个点,检查其是否小于等于所有邻居。时间 O(mn),空间 O(1)。

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
