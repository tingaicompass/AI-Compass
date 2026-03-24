> 想系统提升编程能力、查看更完整的学习路线，欢迎访问 AI Compass：https://github.com/tingaicompass/AI-Compass
> 仓库持续更新刷题题解、Python 基础和 AI 实战内容，适合想高效进阶的你。

# 📖 第81课:最小路径和

> **模块**:动态规划 | **难度**:Medium ⭐⭐
> **LeetCode 链接**:https://leetcode.cn/problems/minimum-path-sum/
> **前置知识**:第80课(不同路径)、第72课(杨辉三角)
> **预计学习时间**:25分钟

---

## 🎯 题目描述

给定一个 m x n 的网格,网格中每个格子都有一个非负数。你需要从左上角出发,走到右下角,每一步只能向右或向下移动。请找出一条路径,使得路径上的所有数字之和最小。

**示例:**
```
输入:grid = [[1,3,1],
             [1,5,1],
             [4,2,1]]
输出:7
解释:路径 1→3→1→1→1 的总和最小
```

**约束条件:**
- m == grid.length
- n == grid[i].length
- 1 <= m, n <= 200
- 0 <= grid[i][j] <= 100

---

### 🧪 边界用例(面试必考)

| 用例类型 | 输入 | 期望输出 | 考察点 |
|---------|------|---------|--------|
| 单格子 | [[5]] | 5 | 最小输入 |
| 单行 | [[1,2,3]] | 6 | 只能向右 |
| 单列 | [[1],[2],[3]] | 6 | 只能向下 |
| 包含0 | [[0,0],[0,0]] | 0 | 特殊值处理 |
| 最大规模 | 200x200网格 | — | 性能边界 |

---

## 💡 思路引导

### 生活化比喻
> 想象你在一个山地公园徒步,从西北角的入口走到东南角的出口。每走过一个路段都要消耗一定的体力(格子上的数字),你只能向东或向南走。
>
> 🐌 **笨办法**:尝试所有可能的路径,记录每条路径的总体力消耗,最后选最省力的那条。这意味着每个岔路口都要"分身"去探索,共有C(m+n-2, m-1)条路径,数量庞大。
>
> 🚀 **聪明办法**:站在任何一个岔路口时,只需要知道"从起点到这个路口的最省力走法"。因为无论过去怎么走,从这个路口往后的最优路径只取决于"这个路口的最小累计体力值"。这就是动态规划的核心思想!

### 关键洞察
**每个格子(i,j)的最小路径和 = 当前格子的值 + min(从上方来的最小路径和, 从左方来的最小路径和)**

---

## 🧠 解题思维链

> 这一节模拟你在面试中"从零开始思考"的过程。

### Step 1:理解题目 → 锁定输入输出
- **输入**:m x n的二维网格,每个格子有非负整数
- **输出**:从(0,0)到(m-1,n-1)的最小路径和(整数)
- **限制**:每步只能向右或向下,无法回退或斜走

### Step 2:先想笨办法(暴力法)
用DFS遍历所有可能的路径,每条路径记录总和,最后取最小值。
- 时间复杂度:O(2^(m+n)) — 每个格子都有向右/向下两种选择
- 瓶颈在哪:大量重复计算,比如到达(1,1)的路径会被多次计算

### Step 3:瓶颈分析 → 优化方向
观察发现,到达同一个格子(i,j)的最小路径和是固定的,与之后的路径无关。
- 核心问题:重复计算到达每个格子的最小路径和
- 优化思路:用表格记录每个格子的最小路径和,避免重复计算 → 动态规划

### Step 4:选择武器
- 选用:**网格DP**
- 理由:符合最优子结构(大问题可分解为子问题),且有重叠子问题(同一格子会被多次访问)

> 🔑 **模式识别提示**:当题目出现"网格中的路径问题+求最值",优先考虑"网格DP"

---

## 🔑 解法一:二维DP(标准网格DP)

### 思路
建立一个与原网格同样大小的dp数组,dp[i][j]表示从(0,0)到(i,j)的最小路径和。通过填表的方式,从左上角逐步推导到右下角。

### 图解过程

```
原始网格:
  0  1  2
0[1, 3, 1]
1[1, 5, 1]
2[4, 2, 1]

Step 1: 初始化第一行和第一列(只有一个方向可走)
dp[0][0] = 1
dp[0][1] = 1+3 = 4
dp[0][2] = 4+1 = 5
dp[1][0] = 1+1 = 2
dp[2][0] = 2+4 = 6

  0  1  2
0[1, 4, 5]
1[2, ?, ?]
2[6, ?, ?]

Step 2: 填充dp[1][1]
从上方来:dp[0][1] + grid[1][1] = 4+5 = 9
从左方来:dp[1][0] + grid[1][1] = 2+5 = 7
取较小值:dp[1][1] = 7

  0  1  2
0[1, 4, 5]
1[2, 7, ?]
2[6, ?, ?]

Step 3: 继续填充
dp[1][2] = min(dp[0][2], dp[1][1]) + 1 = min(5,7) + 1 = 6
dp[2][1] = min(dp[1][1], dp[2][0]) + 2 = min(7,6) + 2 = 8
dp[2][2] = min(dp[1][2], dp[2][1]) + 1 = min(6,8) + 1 = 7

最终DP表:
  0  1  2
0[1, 4, 5]
1[2, 7, 6]
2[6, 8, 7] ← 答案
```

**边界示例 — 单行网格[1,2,3]:**
```
dp = [1, 3, 6] (每步累加)
```

### Python代码

```python
from typing import List


def minPathSum(grid: List[List[int]]) -> int:
    """
    解法一:二维DP
    思路:dp[i][j]表示从(0,0)到(i,j)的最小路径和
    """
    if not grid or not grid[0]:
        return 0

    m, n = len(grid), len(grid[0])
    dp = [[0] * n for _ in range(m)]

    # 初始化起点
    dp[0][0] = grid[0][0]

    # 初始化第一行(只能从左边来)
    for j in range(1, n):
        dp[0][j] = dp[0][j - 1] + grid[0][j]

    # 初始化第一列(只能从上边来)
    for i in range(1, m):
        dp[i][0] = dp[i - 1][0] + grid[i][0]

    # 填充其余格子
    for i in range(1, m):
        for j in range(1, n):
            dp[i][j] = grid[i][j] + min(dp[i - 1][j], dp[i][j - 1])

    return dp[m - 1][n - 1]


# ✅ 测试
print(minPathSum([[1,3,1],[1,5,1],[4,2,1]]))  # 期望输出:7
print(minPathSum([[1,2,3],[4,5,6]]))          # 期望输出:12
print(minPathSum([[1]]))                       # 期望输出:1
```

### 复杂度分析
- **时间复杂度**:O(m*n) — 需要填充整个dp表,每个格子计算一次
  - 具体地说:如果网格是100x100,大约需要10,000次操作
- **空间复杂度**:O(m*n) — dp表与原网格同样大小

### 优缺点
- ✅ 思路清晰,容易理解和实现
- ✅ 适合初学者掌握网格DP的标准模式
- ❌ 空间占用较大,当网格很大时可能内存不够

---

## 🏆 解法二:一维DP滚动数组(最优解 — 空间优化)

### 优化思路
观察到填表时,计算dp[i][j]只需要用到dp[i-1][j]和dp[i][j-1],即只需要"上一行"和"当前行的左边"。因此可以用一维数组滚动更新,节省空间。

> 💡 **关键想法**:一维数组中的dp[j]在更新前存储的是"上一行的dp[j]"(即dp[i-1][j]),更新后变为"当前行的dp[j]"(即dp[i][j])

### 图解过程

```
网格:
  0  1  2
0[1, 3, 1]
1[1, 5, 1]
2[4, 2, 1]

初始:dp = [1, 4, 5] (第一行)

处理第二行(i=1):
j=0: dp[0] = grid[1][0] + dp[0] = 1+1 = 2
     dp = [2, 4, 5]
j=1: dp[1] = grid[1][1] + min(dp[0], dp[1]) = 5+min(2,4) = 7
     dp = [2, 7, 5]
j=2: dp[2] = grid[1][2] + min(dp[1], dp[2]) = 1+min(7,5) = 6
     dp = [2, 7, 6]

处理第三行(i=2):
j=0: dp[0] = 4+2 = 6 → dp = [6, 7, 6]
j=1: dp[1] = 2+min(6,7) = 8 → dp = [6, 8, 6]
j=2: dp[2] = 1+min(8,6) = 7 → dp = [6, 8, 7] ← 答案
```

### Python代码

```python
def minPathSumOptimized(grid: List[List[int]]) -> int:
    """
    解法二:一维DP滚动数组
    思路:用一维数组滚动更新,dp[j]复用为"上一行的j"和"当前行的j"
    """
    if not grid or not grid[0]:
        return 0

    m, n = len(grid), len(grid[0])
    dp = [0] * n

    # 初始化第一行
    dp[0] = grid[0][0]
    for j in range(1, n):
        dp[j] = dp[j - 1] + grid[0][j]

    # 逐行更新
    for i in range(1, m):
        # 更新当前行的第一列(只能从上方来)
        dp[0] += grid[i][0]
        # 更新当前行的其余列
        for j in range(1, n):
            dp[j] = grid[i][j] + min(dp[j], dp[j - 1])
            #               上方来 ↑   左方来 ←

    return dp[n - 1]


# ✅ 测试
print(minPathSumOptimized([[1,3,1],[1,5,1],[4,2,1]]))  # 期望输出:7
print(minPathSumOptimized([[1,2,3],[4,5,6]]))          # 期望输出:12
```

### 复杂度分析
- **时间复杂度**:O(m*n) — 与解法一相同,每个格子计算一次
- **空间复杂度**:O(n) — 只需要一维数组,大幅节省空间

---

## ⚡ 解法三:原地修改(极致空间优化,面试慎用)

### 优化思路
如果允许修改原数组,可以直接在grid上进行DP操作,空间复杂度降为O(1)。

> ⚠️ **注意**:这种做法破坏了原始数据,在实际工程中通常不推荐,但在算法面试中可以作为"展示优化思路"的加分项。

### Python代码

```python
def minPathSumInPlace(grid: List[List[int]]) -> int:
    """
    解法三:原地修改
    思路:直接在原网格上累加,grid[i][j]最终存储从(0,0)到(i,j)的最小路径和
    """
    if not grid or not grid[0]:
        return 0

    m, n = len(grid), len(grid[0])

    # 初始化第一行
    for j in range(1, n):
        grid[0][j] += grid[0][j - 1]

    # 初始化第一列
    for i in range(1, m):
        grid[i][0] += grid[i - 1][0]

    # 填充其余格子
    for i in range(1, m):
        for j in range(1, n):
            grid[i][j] += min(grid[i - 1][j], grid[i][j - 1])

    return grid[m - 1][n - 1]


# ✅ 测试
print(minPathSumInPlace([[1,3,1],[1,5,1],[4,2,1]]))  # 期望输出:7
```

### 复杂度分析
- **时间复杂度**:O(m*n)
- **空间复杂度**:O(1) — 不使用额外空间

---

## 🐍 Pythonic 写法

利用zip和列表推导式的简洁写法:

```python
# 使用functools.reduce进行行级滚动更新
from functools import reduce

def minPathSumPythonic(grid: List[List[int]]) -> int:
    """Pythonic写法:使用reduce进行行级更新"""
    def update_row(prev_row, curr_row):
        new_row = [prev_row[0] + curr_row[0]]
        for i in range(1, len(curr_row)):
            new_row.append(curr_row[i] + min(new_row[i-1], prev_row[i]))
        return new_row

    # 初始化第一行
    first_row = grid[0]
    for i in range(1, len(first_row)):
        first_row[i] += first_row[i-1]

    return reduce(update_row, grid[1:], first_row)[-1]
```

这个写法展示了函数式编程的思想,将"逐行更新"抽象为reduce操作。

> ⚠️ **面试建议**:先写清晰版本展示思路,再提Pythonic写法展示语言功底。
> 面试官更看重你的**思考过程**,而非代码行数。

---

## 📊 解法对比

| 维度 | 解法一:二维DP | 🏆 解法二:一维DP(最优) | 解法三:原地修改 |
|------|--------------|---------------------|----------------|
| 时间复杂度 | O(m*n) | **O(m*n)** ← 时间最优 | O(m*n) |
| 空间复杂度 | O(m*n) | **O(n)** ← 空间优化 | O(1) |
| 代码难度 | 简单 | 中等 | 简单 |
| 面试推荐 | ⭐⭐ | **⭐⭐⭐** ← 首选 | ⭐ |
| 适用场景 | 学习标准DP | **面试首选** | 允许修改原数组时 |

**为什么解法二是最优解**:
- 时间复杂度O(m*n)已经是最优(至少要遍历所有格子一遍)
- 空间从O(m*n)优化到O(n),在网格很大时内存节省显著(如200x200网格,从40000降到200)
- 不破坏原数组,工程实践中更安全
- 代码仍然清晰易懂,面试中容易写对

---

## 🎤 面试现场

> 模拟面试中的完整对话流程,帮你练习"边想边说"。

**面试官**:请你解决一下这道"最小路径和"问题。

**你**:(审题30秒)好的,这道题要求从网格左上角走到右下角,每步只能向右或向下,找出路径和最小的路径。让我先想一下...

我的第一个想法是用DFS暴力枚举所有路径,时间复杂度是O(2^(m+n)),对于大规模网格会超时。

不过这是一个典型的网格DP问题。我们可以用dp[i][j]表示从起点到(i,j)的最小路径和。状态转移方程是:dp[i][j] = grid[i][j] + min(dp[i-1][j], dp[i][j-1])。

时间复杂度优化到O(m*n),空间可以进一步优化到O(n)使用滚动数组。

**面试官**:很好,请写一下代码。

**你**:(边写边说)
```python
def minPathSum(grid):
    m, n = len(grid), len(grid[0])
    dp = [0] * n

    # 初始化第一行
    dp[0] = grid[0][0]
    for j in range(1, n):
        dp[j] = dp[j-1] + grid[0][j]

    # 逐行更新
    for i in range(1, m):
        dp[0] += grid[i][0]  # 第一列只能从上方来
        for j in range(1, n):
            dp[j] = grid[i][j] + min(dp[j], dp[j-1])
            # dp[j]是上一行的值,dp[j-1]是当前行左边的值

    return dp[n-1]
```

**面试官**:测试一下?

**你**:用示例[[1,3,1],[1,5,1],[4,2,1]]走一遍...
初始dp=[1,4,5],处理第二行后dp=[2,7,6],处理第三行后dp=[6,8,7],返回7。正确!

再测一个边界情况[[1]],返回1。结果正确。

### 高频追问

| 追问 | 应答策略 |
|------|---------|
| "还有更优解吗?" | "时间O(m*n)已经是最优(必须遍历所有格子),空间已从O(m*n)优化到O(n)。如果允许修改原数组,可以做到O(1)空间,但破坏了输入数据。" |
| "如果网格非常大怎么办?" | "可以分块处理,每次加载部分数据到内存,或使用流式处理。如果网格在磁盘上,可以按行读取,只在内存中保留一行的DP数组。" |
| "能不能空间O(1)?" | "可以直接在原网格上修改,但会破坏输入数据。实际工程中不推荐,除非明确允许。" |
| "路径可以往回走呢?" | "那就不能用DP了,需要用Dijkstra最短路径算法或BFS,复杂度会上升到O(mn*log(mn))。" |

---

## 🎓 知识点总结

### Python技巧卡片 🐍
```python
# 技巧1:二维数组初始化 — 避免浅拷贝陷阱
dp = [[0] * n for _ in range(m)]  # ✅ 正确
dp = [[0] * n] * m  # ❌ 错误:m行共享同一个列表

# 技巧2:滚动数组原地更新 — 复用变量节省空间
for i in range(1, m):
    dp[0] += grid[i][0]  # 先更新第一个元素
    for j in range(1, n):
        dp[j] = grid[i][j] + min(dp[j], dp[j-1])
        # dp[j]还未更新时是"上一行的值",更新后变为"当前行的值"
```

### 💡 底层原理(选读)

> **为什么滚动数组可以工作?**
>
> 在二维DP中,计算dp[i][j]时只依赖dp[i-1][j](上方)和dp[i][j-1](左方)。滚动数组利用了"从左到右更新"的顺序:
> - 当前位置j更新前,dp[j]存储的是上一行的值(即dp[i-1][j])
> - 当前位置j更新后,dp[j]存储的是当前行的值(即dp[i][j])
> - dp[j-1]已经在本轮更新过,是当前行左边的值
>
> 这种"旧值被新值覆盖但仍能及时使用"的技巧是滚动数组的核心。

### 算法模式卡片 📐
- **模式名称**:网格DP(Grid DP)
- **适用条件**:
  1. 在m×n网格中找最优路径/方案
  2. 每步只能向右/向下移动(单向性)
  3. 当前状态只依赖相邻格子的状态
- **识别关键词**:"网格"、"路径"、"最小/最大"、"只能向右/向下"
- **模板代码**:
```python
# 标准网格DP模板
def gridDP(grid):
    m, n = len(grid), len(grid[0])
    dp = [[0] * n for _ in range(m)]

    # 初始化第一行和第一列
    dp[0][0] = grid[0][0]
    for i in range(1, m):
        dp[i][0] = dp[i-1][0] + grid[i][0]
    for j in range(1, n):
        dp[0][j] = dp[0][j-1] + grid[0][j]

    # 填充其余格子
    for i in range(1, m):
        for j in range(1, n):
            dp[i][j] = grid[i][j] + 某个函数(dp[i-1][j], dp[i][j-1])

    return dp[m-1][n-1]

# 空间优化版(滚动数组)
def gridDPOptimized(grid):
    m, n = len(grid), len(grid[0])
    dp = [0] * n

    # 初始化第一行
    dp[0] = grid[0][0]
    for j in range(1, n):
        dp[j] = dp[j-1] + grid[0][j]

    # 逐行更新
    for i in range(1, m):
        dp[0] += grid[i][0]
        for j in range(1, n):
            dp[j] = grid[i][j] + 某个函数(dp[j], dp[j-1])

    return dp[n-1]
```

### 易错点 ⚠️
1. **忘记初始化第一行和第一列**
   - 错误:直接从dp[1][1]开始填表
   - 解释:第一行只能从左边累加,第一列只能从上边累加,需要单独初始化
   - 正确做法:先初始化边界,再填充内部格子

2. **滚动数组更新顺序错误**
   - 错误:
   ```python
   for j in range(1, n):
       dp[j] = grid[i][j] + min(dp[j], dp[j-1])
   dp[0] += grid[i][0]  # ❌ 第一列应该先更新
   ```
   - 解释:如果先更新j>=1的列,dp[1]会错误地使用未更新的dp[0]
   - 正确做法:每行先更新dp[0],再从左到右更新其余列

3. **混淆"路径数"和"路径和"**
   - 本题求的是"最小路径和",状态转移是min(上,左)+当前值
   - 如果是第80课的"不同路径"(求路径数),状态转移是dp[i][j] = dp[i-1][j] + dp[i][j-1]
   - 注意题目要求的是"最值"还是"计数"

---

## 🏗️ 工程实战(选读)

> 这个算法思想在真实项目中的应用,让你知道"学了有什么用"。

- **场景1:成本优化路径规划**
  - 在云计算中,数据从源节点传输到目标节点,每个中继节点都有流量成本
  - 可以建模为网格DP,找出最低成本的传输路径

- **场景2:图像处理中的Seam Carving**
  - 图像内容感知缩放算法,通过移除"能量最低"的像素缝(seam)来缩小图像
  - 从上到下找一条路径,使得路径上的能量和最小,正是网格DP的应用

- **场景3:游戏关卡设计**
  - 在策略游戏中,角色从起点移动到终点,地形有不同的移动成本
  - 用网格DP计算最优移动路径,引导玩家体验

---

## 🏋️ 举一反三

完成本课后,试试这些同类题目来巩固知识:

| 题目 | 难度 | 相关知识点 | 提示 |
|------|------|-----------|------|
| LeetCode 62. 不同路径 | Medium | 网格DP路径计数 | 状态转移是加法而非取min |
| LeetCode 63. 不同路径II | Medium | 网格DP+障碍处理 | 遇到障碍格子时dp值为0 |
| LeetCode 174. 地下城游戏 | Hard | 逆向网格DP | 从终点倒推,维护"最低健康值" |
| LeetCode 931. 下降路径最小和 | Medium | 网格DP三方向 | 可以向左下/正下/右下移动 |
| LeetCode 120. 三角形最小路径和 | Medium | 变形网格DP | 不是矩形而是三角形,状态转移类似 |

---

## 📝 课后小测

试试这道变体题,不要看答案,自己先想5分钟!

**题目**:如果网格中每个格子还有一个"通行证"要求(0或1),只有持有通行证才能通过某些格子。你最多可以获得K张通行证。求从左上到右下的最小路径和。

输入:grid = [[1,3,1],[1,5,1],[4,2,1]], pass = [[0,1,0],[0,1,0],[0,0,0]], k = 1
解释:pass[i][j]=1表示该格子需要通行证

<details>
<summary>💡 提示(实在想不出来再点开)</summary>

增加一个维度:dp[i][j][p]表示到达(i,j)且已使用p张通行证的最小路径和

</details>

<details>
<summary>✅ 参考答案</summary>

```python
def minPathSumWithPass(grid, pass_grid, k):
    """
    三维DP:dp[i][j][p]表示到(i,j)使用p张通行证的最小路径和
    """
    m, n = len(grid), len(grid[0])
    INF = float('inf')
    dp = [[[INF] * (k+2) for _ in range(n)] for _ in range(m)]

    # 初始化起点
    if pass_grid[0][0] == 0:
        dp[0][0][0] = grid[0][0]
    else:
        dp[0][0][1] = grid[0][0]

    # 填表
    for i in range(m):
        for j in range(n):
            if i == 0 and j == 0:
                continue
            for p in range(k+2):
                # 从上方来
                if i > 0:
                    if pass_grid[i][j] == 0:
                        dp[i][j][p] = min(dp[i][j][p], dp[i-1][j][p] + grid[i][j])
                    elif p > 0:
                        dp[i][j][p] = min(dp[i][j][p], dp[i-1][j][p-1] + grid[i][j])
                # 从左方来(类似逻辑)
                if j > 0:
                    if pass_grid[i][j] == 0:
                        dp[i][j][p] = min(dp[i][j][p], dp[i][j-1][p] + grid[i][j])
                    elif p > 0:
                        dp[i][j][p] = min(dp[i][j][p], dp[i][j-1][p-1] + grid[i][j])

    return min(dp[m-1][n-1])

# 测试
print(minPathSumWithPass([[1,3,1],[1,5,1],[4,2,1]],
                         [[0,1,0],[0,1,0],[0,0,0]], 1))
```

核心思路:增加一个维度记录"已使用的通行证数量",遇到需要通行证的格子时,从p-1状态转移到p状态。

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
