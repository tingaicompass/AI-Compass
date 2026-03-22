# 📖 第91课:课程表

> **模块**:图论 | **难度**:Medium ⭐⭐⭐
> **LeetCode 链接**:https://leetcode.cn/problems/course-schedule/
> **前置知识**:第89课(岛屿数量 - DFS/BFS基础)、第90课(腐烂的橘子 - 多源BFS)
> **预计学习时间**:30分钟

---

## 🎯 题目描述

你需要选修 `numCourses` 门课程,编号为 `0` 到 `numCourses - 1`。

给定一个数组 `prerequisites`,其中 `prerequisites[i] = [ai, bi]` 表示如果想学习课程 `ai`,必须先学习课程 `bi`。

判断是否可能完成所有课程的学习。换句话说,判断课程依赖关系中是否存在环。

**示例 1:**
```
输入:numCourses = 2, prerequisites = [[1,0]]
输出:true
解释:总共2门课,学完课程0后可以学习课程1。
```

**示例 2:**
```
输入:numCourses = 2, prerequisites = [[1,0],[0,1]]
输出:false
解释:需要先学0才能学1,同时需要先学1才能学0,形成死循环。
```

**约束条件:**
- `1 <= numCourses <= 2000` — 课程数量适中
- `0 <= prerequisites.length <= 5000` — 依赖关系最多5000条
- `prerequisites[i].length == 2` — 每条关系包含两个课程
- 所有课程对 `[ai, bi]` **互不相同** — 无重复边

---

### 🧪 边界用例(面试必考)

| 用例类型 | 输入 | 期望输出 | 考察点 |
|---------|------|---------|--------|
| 最小输入 | numCourses=1, prerequisites=[] | true | 无依赖时直接返回true |
| 单向链 | [[1,0],[2,1],[3,2]] | true | 线性依赖无环 |
| 直接环 | [[1,0],[0,1]] | false | 两个节点互相依赖 |
| 复杂环 | [[1,0],[2,1],[0,2]] | false | 三个节点形成环 |
| 独立课程 | numCourses=5, prerequisites=[] | true | 所有课程独立可学 |
| 多连通分量 | [[1,0],[3,2]] | true | 多个独立的依赖链 |

---

## 💡 思路引导

### 生活化比喻

> 想象你在大学选课,有些高级课程要求先修课。
>
> 🐌 **笨办法**:每次尝试选一门课,发现需要先修课A,去选A又发现需要先修课B,去选B又发现需要A...一圈绕回来发现陷入死循环。这就像走迷宫一样,一条路一条路去试错,效率极低。
>
> 🚀 **聪明办法**:教务系统会自动检测"是否存在循环依赖"。它的做法是:**先找出所有不需要先修课的课程(入度为0),选完这些课后,依赖它们的课程就可以解锁了,再继续选解锁的课程**。如果最终所有课程都能被选完,说明没有环;如果还剩课程没选,说明存在环形依赖。

### 关键洞察

**将课程依赖关系建模为有向图,判断能否完成所有课程 = 判断有向图中是否有环**

---

## 🧠 解题思维链

> 这一节模拟你在面试中"从零开始思考"的过程。

### Step 1:理解题目 → 锁定输入输出

- **输入**:课程总数 `numCourses`,依赖关系数组 `prerequisites`
- **输出**:布尔值,true表示能完成所有课程(无环),false表示不能(有环)
- **限制**:需要在合理时间内处理最多2000个节点和5000条边的图

### Step 2:先想笨办法(暴力法)

对每个课程进行DFS深度搜索,记录访问路径。如果在搜索过程中再次遇到路径上的节点,说明有环。

- 时间复杂度:O(V * (V + E)) — 对每个节点都做一次DFS
- 瓶颈在哪:**重复遍历了很多节点和边,且没有利用"已验证无环的节点"的信息**

### Step 3:瓶颈分析 → 优化方向

暴力DFS的问题是:
- 对每个节点单独DFS,导致重复访问
- 没有"记忆化",已经确认无环的节点还要重复检查

优化思路:
- **DFS优化**:用三色标记法(未访问/访问中/已完成)避免重复检查
- **BFS拓扑排序**:利用"入度"概念,从入度为0的节点开始逐层剥离

### Step 4:选择武器

本题有**两种经典解法**:
1. **DFS + 三色标记**(检测回边)
2. **BFS + 拓扑排序**(Kahn算法)

- 选用:**拓扑排序(BFS)**作为主推解法
- 理由:
  - 拓扑排序是有向无环图(DAG)判定的标准算法
  - BFS实现直观,易于理解和编码
  - 面试中更容易讲清楚逻辑

> 🔑 **模式识别提示**:当题目出现"课程依赖"、"任务顺序"、"前置条件"等关键词,优先考虑**拓扑排序**

---

## 🔑 解法一:DFS + 三色标记(环检测)

### 思路

用深度优先搜索遍历图,给每个节点标记三种状态:
- **0 (白色)**:未访问
- **1 (灰色)**:正在访问中(在当前DFS路径上)
- **2 (黑色)**:已完成访问(该节点及其后代都无环)

如果DFS过程中遇到**灰色节点**,说明遇到了回边,存在环。

### 图解过程

```
示例:numCourses = 4, prerequisites = [[1,0],[2,1],[3,2]]

构建邻接表:
  0 -> [1]
  1 -> [2]
  2 -> [3]
  3 -> []

DFS执行过程:

Step 1: 从节点0开始
  访问0 (标记灰色1)
    -> 访问1 (标记灰色1)
      -> 访问2 (标记灰色1)
        -> 访问3 (标记灰色1)
          -> 3无后继,标记黑色2 ✓
        <- 2完成,标记黑色2 ✓
      <- 1完成,标记黑色2 ✓
    <- 0完成,标记黑色2 ✓

结果:所有节点都变为黑色,无环 → 返回true

---

反例:prerequisites = [[1,0],[0,1]]

构建邻接表:
  0 -> [1]
  1 -> [0]

DFS执行:
  访问0 (灰色1)
    -> 访问1 (灰色1)
      -> 访问0 (发现0已是灰色!) ❌ 检测到环!

返回false
```

### Python代码

```python
from typing import List
from collections import defaultdict


def canFinish(numCourses: int, prerequisites: List[List[int]]) -> bool:
    """
    解法一:DFS + 三色标记(环检测)
    思路:用DFS遍历图,通过检测回边判断是否有环
    """
    # 构建邻接表
    graph = defaultdict(list)
    for course, prereq in prerequisites:
        graph[prereq].append(course)  # prereq -> course

    # 0=未访问(白), 1=访问中(灰), 2=已完成(黑)
    color = [0] * numCourses

    def dfs(node):
        """返回True表示无环,False表示有环"""
        if color[node] == 1:  # 灰色节点 → 回边 → 有环
            return False
        if color[node] == 2:  # 黑色节点 → 已验证无环
            return True

        color[node] = 1  # 标记为访问中(灰色)

        # 访问所有邻居
        for neighbor in graph[node]:
            if not dfs(neighbor):  # 如果发现环
                return False

        color[node] = 2  # 标记为已完成(黑色)
        return True

    # 对所有未访问节点执行DFS
    for i in range(numCourses):
        if color[i] == 0:  # 白色节点
            if not dfs(i):
                return False

    return True


# ✅ 测试
print(canFinish(2, [[1, 0]]))           # 期望输出:true
print(canFinish(2, [[1, 0], [0, 1]]))   # 期望输出:false
print(canFinish(4, [[1, 0], [2, 1], [3, 2]]))  # 期望输出:true
print(canFinish(3, [[1, 0], [2, 1], [0, 2]]))  # 期望输出:false (环:0->1->2->0)
```

### 复杂度分析

- **时间复杂度**:O(V + E) — V是课程数,E是依赖关系数
  - 每个节点最多访问一次(白→灰→黑)
  - 每条边最多检查一次
  - 具体地说:如果有2000门课程和5000条依赖,大约需要7000次操作

- **空间复杂度**:O(V + E) — 邻接表O(E) + 颜色数组O(V) + 递归栈O(V)

### 优缺点

- ✅ 直接检测环,逻辑简洁
- ✅ 空间利用高效(只需颜色数组)
- ❌ 递归深度可能很大(极端情况下链式依赖会达到V层)
- ❌ 对于初学者,三色标记理解有一定难度

---

## 🏆 解法二:拓扑排序 BFS(Kahn算法,最优解)

### 优化思路

**核心想法**:
- 有向无环图(DAG)一定可以进行拓扑排序
- 拓扑排序的过程:每次选择入度为0的节点,删除它及其出边,重复此过程
- 如果能删除所有节点,说明无环;如果还剩节点,说明这些节点在环中(入度永远无法变为0)

> 💡 **关键想法**:入度为0的节点就像"没有前置条件的课程",可以直接学习。学完后,依赖它的课程的"前置条件数"减1。

### 图解过程

```
示例:numCourses = 4, prerequisites = [[1,0],[2,0],[3,1],[3,2]]

Step 0: 构建图和入度表
  图:
    0 -> [1, 2]
    1 -> [3]
    2 -> [3]
    3 -> []

  入度:
    0: 0  (无前置)
    1: 1  (需要0)
    2: 1  (需要0)
    3: 2  (需要1和2)

Step 1: 队列初始化
  队列 = [0]  (入度为0的节点)
  已处理 = 0

Step 2: 处理节点0
  弹出0 → 已处理 = 1
  更新邻居:
    1的入度: 1 -> 0 (入队)
    2的入度: 1 -> 0 (入队)
  队列 = [1, 2]

Step 3: 处理节点1
  弹出1 → 已处理 = 2
  更新邻居:
    3的入度: 2 -> 1
  队列 = [2]

Step 4: 处理节点2
  弹出2 → 已处理 = 3
  更新邻居:
    3的入度: 1 -> 0 (入队)
  队列 = [3]

Step 5: 处理节点3
  弹出3 → 已处理 = 4
  队列 = []

结果:已处理 == numCourses (4) → 返回true

---

反例:prerequisites = [[1,0],[0,1]]

入度:
  0: 1
  1: 1

初始队列 = [] (没有入度为0的节点!)
已处理 = 0

结果:已处理 < numCourses → 返回false
```

### Python代码

```python
from typing import List
from collections import defaultdict, deque


def canFinish(numCourses: int, prerequisites: List[List[int]]) -> bool:
    """
    解法二:拓扑排序 BFS (Kahn算法)
    思路:从入度为0的节点开始逐层剥离,能剥离完说明无环
    """
    # 1. 构建图和入度表
    graph = defaultdict(list)
    in_degree = [0] * numCourses

    for course, prereq in prerequisites:
        graph[prereq].append(course)  # prereq -> course
        in_degree[course] += 1

    # 2. 找出所有入度为0的节点(无前置条件的课程)
    queue = deque([i for i in range(numCourses) if in_degree[i] == 0])

    # 3. BFS逐层处理
    processed = 0  # 已处理的课程数

    while queue:
        node = queue.popleft()
        processed += 1  # 选修这门课

        # 更新邻居的入度
        for neighbor in graph[node]:
            in_degree[neighbor] -= 1
            if in_degree[neighbor] == 0:  # 前置条件满足
                queue.append(neighbor)

    # 4. 判断是否所有课程都能学完
    return processed == numCourses


# ✅ 测试
print(canFinish(2, [[1, 0]]))           # 期望输出:True
print(canFinish(2, [[1, 0], [0, 1]]))   # 期望输出:False
print(canFinish(4, [[1, 0], [2, 0], [3, 1], [3, 2]]))  # 期望输出:True
print(canFinish(1, []))                 # 期望输出:True (无依赖)
```

### 复杂度分析

- **时间复杂度**:O(V + E) — 与DFS相同
  - 构建图和入度表:O(E)
  - BFS遍历:每个节点入队出队一次O(V),每条边检查一次O(E)
  - 总计:O(V + E)

- **空间复杂度**:O(V + E)
  - 邻接表O(E) + 入度数组O(V) + 队列O(V)

### 为什么是最优解

1. **时间复杂度O(V+E)已经是理论最优** — 必须至少遍历所有边一次才能判断环
2. **空间复杂度合理** — O(V+E)用于存储图结构,无法避免
3. **无递归栈风险** — 迭代BFS不会栈溢出
4. **逻辑直观易懂** — "入度"概念比三色标记更容易向面试官解释
5. **通用性强** — Kahn算法不仅能判环,还能输出拓扑序列(见举一反三题)

---

## 🐍 Pythonic 写法

利用列表推导式和生成器表达式简化代码:

```python
def canFinish_pythonic(numCourses: int, prerequisites: List[List[int]]) -> bool:
    from collections import defaultdict, deque

    # 一行构建图(使用setdefault)
    graph = defaultdict(list)
    in_degree = [0] * numCourses
    for a, b in prerequisites:
        graph[b].append(a)
        in_degree[a] += 1

    # 一行生成初始队列
    queue = deque(i for i in range(numCourses) if in_degree[i] == 0)

    # BFS + 计数
    processed = sum(1 for _ in iter(lambda: queue and queue.popleft(), None)
                    for neighbor in graph.get(_, [])
                    if not (in_degree.__setitem__(neighbor, in_degree[neighbor] - 1) or
                            in_degree[neighbor] or queue.append(neighbor)))

    # 简洁写法(推荐):
    processed = 0
    while queue:
        processed += 1
        for neighbor in graph[queue.popleft()]:
            in_degree[neighbor] -= 1
            in_degree[neighbor] or queue.append(neighbor)

    return processed == numCourses
```

> ⚠️ **面试建议**:先写清晰版本展示思路,再提"可以用列表推导优化初始化"展示语言功底。过度Pythonic会降低可读性。

---

## 📊 解法对比

| 维度 | 解法一:DFS + 三色标记 | 🏆 解法二:拓扑排序BFS(最优) |
|------|---------------------|---------------------------|
| 时间复杂度 | O(V + E) | **O(V + E)** ← 时间最优 |
| 空间复杂度 | O(V + E) | **O(V + E)** ← 相同 |
| 代码难度 | 中等(需理解三色) | **简单**(入度概念直观) |
| 栈溢出风险 | 有(深度递归) | **无**(迭代BFS) |
| 面试推荐 | ⭐⭐ | **⭐⭐⭐** ← 首选 |
| 扩展性 | 只能判环 | **可输出拓扑序列** |
| 适用场景 | 偏好递归思维 | **通用,面试标准解** |

**为什么BFS拓扑排序是最优解**:
- 时间空间已达理论最优O(V+E),无法进一步优化
- 逻辑清晰,易于向面试官解释"为什么这样做"
- 实现简单,不易出错,面试压力下更稳
- 通用性强,可以扩展到"输出课程学习顺序"(LeetCode 210)

**面试建议**:
1. 先用30秒口述DFS思路:"可以用DFS检测环,但有更直观的方法"
2. 立即优化到🏆BFS拓扑排序:"利用入度概念,从无依赖课程开始逐层剥离"
3. **重点讲解核心逻辑**:"入度为0 = 可学习,学完后更新依赖它的课程的入度"
4. 强调为什么最优:"O(V+E)已是理论下限,且逻辑最清晰"
5. 手动测试边界用例:空图、单节点、直接环、线性链

---

## 🎤 面试现场

> 模拟面试中的完整对话流程,帮你练习"边想边说"。

**面试官**:请你解决一下这道课程表问题。

**你**:(审题30秒)好的,这道题要求判断是否能完成所有课程,给定了课程之间的依赖关系。我理解这本质上是判断**有向图中是否存在环**。

让我先想一下...

**直观想法**:可以用DFS遍历图,用三色标记法检测回边,如果遇到"访问中"的节点就说明有环。

**更好的方法**:用**拓扑排序**(Kahn算法)。核心思路是:
1. 统计每个课程的入度(有多少前置条件)
2. 从入度为0的课程开始学习(无前置条件)
3. 学完一门课后,将依赖它的课程的入度减1
4. 重复此过程,如果最终能学完所有课程,说明无环

我用第二种方法,因为它逻辑更直观,且不会有递归栈溢出风险。

**面试官**:很好,请写一下代码。

**你**:(边写边说关键步骤)
```python
# 1. 先构建邻接表和入度数组
graph = defaultdict(list)
in_degree = [0] * numCourses
for course, prereq in prerequisites:
    graph[prereq].append(course)  # prereq指向course
    in_degree[course] += 1

# 2. 找出所有入度为0的课程
queue = deque([i for i in range(numCourses) if in_degree[i] == 0])

# 3. BFS逐个处理
processed = 0
while queue:
    node = queue.popleft()
    processed += 1  # 学习这门课
    for neighbor in graph[node]:
        in_degree[neighbor] -= 1  # 前置条件-1
        if in_degree[neighbor] == 0:
            queue.append(neighbor)

# 4. 判断是否全部学完
return processed == numCourses
```

**面试官**:测试一下?

**你**:用示例 `[[1,0],[0,1]]` 走一遍:
- 构建图:0->1, 1->0
- 入度:0和1都是1
- 初始队列为空(没有入度为0的节点)
- processed=0 < numCourses=2 → 返回false ✓

再测一个正常情况 `[[1,0],[2,1]]`:
- 图:0->1->2
- 入度:0:0, 1:1, 2:1
- 队列:[0] → 学0 → 1入度变0 → 队列:[1] → 学1 → 2入度变0 → 队列:[2] → 学2
- processed=3 == numCourses ✓

**面试官**:复杂度是多少?

**你**:
- 时间O(V+E):构建图O(E),BFS每个节点和边各访问一次O(V+E)
- 空间O(V+E):邻接表O(E),入度数组和队列O(V)

### 高频追问

| 追问 | 应答策略 |
|------|---------|
| "还有更优解吗?" | "时间O(V+E)已经是理论最优,因为至少要遍历所有边一次才能判断环。空间也无法避免,需要存储图结构。" |
| "如果要输出课程学习顺序呢?" | "完全一样的算法!只需在BFS过程中将出队的节点记录到结果数组,就是拓扑序列。这就是LeetCode 210题。" |
| "能用DFS做吗?" | "可以。用三色标记法:白色(未访问)、灰色(访问中)、黑色(已完成)。遇到灰色节点说明有回边(环)。但面试中BFS更直观。" |
| "如果数据量特别大呢?" | "可以考虑:1) 并行化拓扑排序(多个入度为0的节点可同时处理); 2) 如果内存不足,用外部排序或分治处理子图。" |
| "实际工程中的应用?" | "项目构建系统(如Makefile、Maven)检测循环依赖;任务调度系统(DAG任务流);数据库外键约束检测。" |

---

## 🎓 知识点总结

### Python技巧卡片 🐍

```python
# 技巧1:defaultdict构建邻接表 — 避免KeyError
from collections import defaultdict
graph = defaultdict(list)
graph[0].append(1)  # 自动创建空列表

# 技巧2:列表推导生成初始队列 — 简洁优雅
queue = deque([i for i in range(n) if in_degree[i] == 0])

# 技巧3:短路逻辑简化条件判断
in_degree[neighbor] -= 1
if in_degree[neighbor] == 0:
    queue.append(neighbor)
# 等价于:
in_degree[neighbor] or queue.append(neighbor)  # 0为假,执行append
```

### 💡 底层原理(选读)

> **为什么拓扑排序能检测环?**
>
> 核心原理:**有向无环图(DAG)一定可以被拓扑排序,有环图一定不能**。
>
> **数学证明**:
> 1. 如果图中有环,环上所有节点的入度都 ≥1(每个节点至少有一条入边来自环内)
> 2. Kahn算法每次只处理入度为0的节点
> 3. 环上节点永远无法变为入度0(删除外部入边后,环内入边仍存在)
> 4. 因此有环图一定会剩下节点无法处理
>
> **deque性能**:
> - `popleft()` 和 `append()` 都是O(1)
> - 普通list的 `pop(0)` 是O(n)(需要移动所有元素)
> - 这就是为什么BFS必须用deque而不是list

### 算法模式卡片 📐

- **模式名称**:拓扑排序(Kahn算法)
- **适用条件**:有向图,需要判断是否有环 或 需要输出依赖顺序
- **识别关键词**:"任务依赖"、"课程先修"、"编译顺序"、"循环引用检测"
- **核心要素**:
  1. 入度数组(统计每个节点的入边数)
  2. 队列(存储入度为0的节点)
  3. BFS逐层剥离
- **模板代码**:
```python
# 拓扑排序通用模板
def topological_sort(n, edges):
    from collections import defaultdict, deque

    graph = defaultdict(list)
    in_degree = [0] * n

    for u, v in edges:
        graph[u].append(v)
        in_degree[v] += 1

    queue = deque([i for i in range(n) if in_degree[i] == 0])
    order = []

    while queue:
        node = queue.popleft()
        order.append(node)
        for neighbor in graph[node]:
            in_degree[neighbor] -= 1
            if in_degree[neighbor] == 0:
                queue.append(neighbor)

    # 判断是否有环:
    # 有环 → len(order) < n
    # 无环 → len(order) == n,且order就是拓扑序列
    return order if len(order) == n else []
```

### 易错点 ⚠️

1. **边的方向搞反**
   - 错误:`graph[course].append(prereq)` — 这会建成反图
   - 正确:`graph[prereq].append(course)` — prereq指向course
   - 记忆法:"先修课指向后续课"

2. **入度更新时机错误**
   - 错误:只在初始化时统计入度,BFS中不更新
   - 正确:每次处理节点时,将其邻居的入度减1
   - 记忆法:"学完一门课,依赖它的课的前置条件-1"

3. **判断条件写错**
   - 错误:`return len(queue) == 0` — 队列为空不代表全处理完
   - 正确:`return processed == numCourses` — 必须统计实际处理的节点数

---

## 🏗️ 工程实战(选读)

> 这个算法思想在真实项目中的应用,让你知道"学了有什么用"。

- **场景1:构建系统依赖管理**
  - Maven、Gradle等构建工具检测模块间循环依赖
  - Makefile编译顺序决策(先编译哪个源文件)

- **场景2:任务调度系统**
  - Airflow、Oozie等DAG任务流引擎
  - 检测任务间是否有循环依赖,生成执行顺序

- **场景3:数据库外键约束**
  - 检测表之间是否有循环外键引用
  - 删除表时的顺序决策

- **场景4:包管理器**
  - npm、pip等包管理器解析依赖关系
  - 检测循环依赖,生成安装顺序

---

## 🏋️ 举一反三

完成本课后,试试这些同类题目来巩固知识:

| 题目 | 难度 | 相关知识点 | 提示 |
|------|------|-----------|------|
| LeetCode 210. 课程表II | Medium | 拓扑排序输出序列 | 完全相同算法,只需记录order数组 |
| LeetCode 310. 最小高度树 | Medium | 拓扑排序变体 | 从叶子节点(度为1)开始剥离 |
| LeetCode 444. 序列重建 | Medium | 拓扑排序唯一性 | 判断拓扑序列是否唯一 |
| LeetCode 802. 找到最终的安全状态 | Medium | 反向图拓扑排序 | 找出不在环中的节点 |
| LeetCode 1136. 并行课程 | Medium | 拓扑排序+层数 | BFS层序遍历统计最长路径 |

---

## 📝 课后小测

试试这道变体题,不要看答案,自己先想5分钟!

**题目**:给定课程依赖关系,如果可以完成所有课程,返回一种**合法的学习顺序**;如果不能,返回空数组。(LeetCode 210)

<details>
<summary>💡 提示(实在想不出来再点开)</summary>

完全相同的拓扑排序算法!唯一区别:在BFS过程中,每次 `popleft()` 时将节点加入结果数组。

</details>

<details>
<summary>✅ 参考答案</summary>

```python
def findOrder(numCourses: int, prerequisites: List[List[int]]) -> List[int]:
    from collections import defaultdict, deque

    graph = defaultdict(list)
    in_degree = [0] * numCourses

    for course, prereq in prerequisites:
        graph[prereq].append(course)
        in_degree[course] += 1

    queue = deque([i for i in range(numCourses) if in_degree[i] == 0])
    order = []  # 唯一改动:记录顺序

    while queue:
        node = queue.popleft()
        order.append(node)  # 记录学习顺序
        for neighbor in graph[node]:
            in_degree[neighbor] -= 1
            if in_degree[neighbor] == 0:
                queue.append(neighbor)

    return order if len(order) == numCourses else []
```

**核心改动**:增加一行 `order.append(node)`,BFS的出队顺序就是拓扑序列!

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
