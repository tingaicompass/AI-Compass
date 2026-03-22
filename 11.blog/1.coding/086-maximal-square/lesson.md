# 📖 第86课:最大正方形

> **模块**:动态规划 | **难度**:Medium ⭐⭐
> **LeetCode 链接**:https://leetcode.cn/problems/maximal-square/
> **前置知识**:第80课(不同路径 - 网格DP基础)
> **预计学习时间**:30分钟

---

## 🎯 题目描述

给你一个由`'0'`和`'1'`组成的二维字符矩阵,找出只包含`'1'`的最大正方形,并返回其面积。

**示例:**
```
输入:matrix = [
  ["1","0","1","0","0"],
  ["1","0","1","1","1"],
  ["1","1","1","1","1"],
  ["1","0","0","1","0"]
]
输出:4
解释:最大的正方形边长为2(标记为粗体部分)
  1 0 1 0 0
  1 0 【1 1】1
  1 1 【1 1】1
  1 0 0 1 0
面积 = 2 × 2 = 4
```

**约束条件:**
- 1 ≤ m, n ≤ 300 (矩阵行列数)
- matrix[i][j]是`'0'`或`'1'`

---

### 🧪 边界用例(面试必考)

| 用例类型 | 输入 | 期望输出 | 考察点 |
|---------|------|---------|--------|
| 单元素1 | matrix=[["1"]] | 1 | 最小正方形 |
| 单元素0 | matrix=[["0"]] | 0 | 全为0 |
| 全为1 | matrix=[["1","1"],["1","1"]] | 4 | 整体是正方形 |
| 无正方形 | matrix=[["1","0"],["0","1"]] | 1 | 只有1×1正方形 |
| 大规模 | m=n=300 | — | 性能边界 |

---

## 💡 思路引导

### 生活化比喻

> 想象你在铺地砖,想找出能铺出的最大正方形区域。
>
> 🐌 **笨办法**:枚举所有可能的正方形左上角和边长,逐个检查是否全为1,时间复杂度O(m·n·min(m,n)²)会很慢。
>
> 🚀 **聪明办法**:对于每个格子,记录"以它为右下角的最大正方形边长"。如果它的上方、左方、左上方三个邻居都能构成正方形,那么当前格子可以把边长扩大1。一遍扫描,O(m·n)搞定。

### 关键洞察

**以每个格子为右下角的最大正方形边长,取决于它的上、左、左上三个邻居的最小值 + 1。**

---

## 🧠 解题思维链

> 这一节模拟你在面试中"从零开始思考"的过程。

### Step 1:理解题目 → 锁定输入输出

- **输入**:二维字符矩阵`matrix`,元素为`'0'`或`'1'`
- **输出**:整数,最大正方形的面积
- **限制**:正方形必须全为`'1'`,边长可变

### Step 2:先想笨办法(暴力法)

枚举所有可能的正方形:
1. 遍历每个位置作为左上角
2. 枚举边长1, 2, 3, ..., min(m, n)
3. 检查这个正方形内是否全为`'1'`

- 时间复杂度:O(m·n·min(m,n)²) — 三层循环 + 检查正方形内部
- 瓶颈在哪:重复检查同一个正方形的子区域,效率低

### Step 3:瓶颈分析 → 优化方向

暴力法重复计算了大量子问题。考虑换个角度:
- 核心问题:如何高效判断一个正方形是否全为1?
- 优化思路:能否利用已知的小正方形信息,推导出大正方形?

### Step 4:选择武器

- 选用:**二维动态规划**
- 理由:
  1. 定义`dp[i][j]`为以`(i,j)`为右下角的最大正方形边长
  2. 如果`matrix[i][j] == '1'`,则`dp[i][j]`取决于上、左、左上三个邻居的最小值
  3. 状态转移:`dp[i][j] = min(dp[i-1][j], dp[i][j-1], dp[i-1][j-1]) + 1`

> 🔑 **模式识别提示**:当题目涉及"二维矩阵最优解"且可以从子问题推导时,优先考虑"二维DP"

---

## 🔑 解法一:暴力枚举(直觉法)

### 思路

枚举所有可能的正方形左上角和边长,逐个检查是否全为`'1'`。

### 图解过程

```
输入:matrix = [
  ["1","0","1","0","0"],
  ["1","0","1","1","1"],
  ["1","1","1","1","1"],
  ["1","0","0","1","0"]
]

枚举左上角(0,0),边长1:
  检查matrix[0][0]='1' ✓
  最大边长更新为1

枚举左上角(1,2),边长2:
  检查matrix[1][2:4][1:3]是否全为'1'
  1 1
  1 1 ✓
  最大边长更新为2

...依次枚举所有可能

问题:重复检查同一区域,效率低
```

### Python代码

```python
from typing import List


def maximalSquare_brute(matrix: List[List[str]]) -> int:
    """
    解法一:暴力枚举
    思路:枚举所有正方形,逐个检查是否全为'1'
    """
    if not matrix or not matrix[0]:
        return 0

    m, n = len(matrix), len(matrix[0])
    max_side = 0

    # 枚举左上角
    for i in range(m):
        for j in range(n):
            if matrix[i][j] == '1':
                # 枚举边长
                max_len = min(m - i, n - j)
                for length in range(1, max_len + 1):
                    # 检查正方形是否全为'1'
                    is_square = True
                    for r in range(i, i + length):
                        for c in range(j, j + length):
                            if matrix[r][c] == '0':
                                is_square = False
                                break
                        if not is_square:
                            break
                    if is_square:
                        max_side = max(max_side, length)
                    else:
                        break  # 更大的边长也不可能全为1

    return max_side * max_side


# ✅ 测试
matrix1 = [["1","0","1","0","0"],["1","0","1","1","1"],["1","1","1","1","1"],["1","0","0","1","0"]]
print(maximalSquare_brute(matrix1))  # 期望输出:4

matrix2 = [["0","1"],["1","0"]]
print(maximalSquare_brute(matrix2))  # 期望输出:1
```

### 复杂度分析

- **时间复杂度**:O(m·n·min(m,n)²) — 三层循环枚举 + 检查正方形内部
  - 具体地说:如果m=n=100,大约需要100×100×100²=100,000,000次操作,约1秒
- **空间复杂度**:O(1) — 只用了几个变量

### 优缺点

- ✅ 思路直观,易于理解
- ❌ 时间复杂度高,对于300×300的矩阵会接近超时
- ❌ 重复检查同一区域,存在大量冗余计算

---

## 🏆 解法二:二维DP(最优解)

### 优化思路

核心洞察:**以(i,j)为右下角的最大正方形边长,取决于它的上、左、左上三个邻居的最小值**。

定义`dp[i][j]`为以`(i,j)`为右下角的最大正方形边长。
- 如果`matrix[i][j] == '0'`,则`dp[i][j] = 0`
- 如果`matrix[i][j] == '1'`,则`dp[i][j] = min(dp[i-1][j], dp[i][j-1], dp[i-1][j-1]) + 1`

> 💡 **关键想法**:如果三个邻居的最小值是k,说明可以在它们的基础上扩展出边长k+1的正方形!

### 图解过程

```
输入:matrix = [
  ["1","0","1","0","0"],
  ["1","0","1","1","1"],
  ["1","1","1","1","1"],
  ["1","0","0","1","0"]
]

初始化:dp同大小矩阵,第一行第一列直接复制matrix

DP过程:
i=1, j=2:
  matrix[1][2]='1'
  上:dp[0][2]=1, 左:dp[1][1]=0, 左上:dp[0][1]=0
  dp[1][2] = min(1,0,0)+1 = 1

i=1, j=3:
  matrix[1][3]='1'
  上:dp[0][3]=0, 左:dp[1][2]=1, 左上:dp[0][2]=1
  dp[1][3] = min(0,1,1)+1 = 1

i=2, j=2:
  matrix[2][2]='1'
  上:dp[1][2]=1, 左:dp[2][1]=1, 左上:dp[1][1]=0
  dp[2][2] = min(1,1,0)+1 = 1

i=2, j=3:
  matrix[2][3]='1'
  上:dp[1][3]=1, 左:dp[2][2]=1, 左上:dp[1][2]=1
  dp[2][3] = min(1,1,1)+1 = 2 ← 最大边长

i=2, j=4:
  matrix[2][4]='1'
  上:dp[1][4]=1, 左:dp[2][3]=2, 左上:dp[1][3]=1
  dp[2][4] = min(1,2,1)+1 = 2

最终:最大边长max_side=2, 面积=4
```

**为什么是min三个邻居?**
```
假设三个邻居的边长分别是a, b, c:
  左上: c×c正方形
  上:   a×a正方形
  左:   b×b正方形

当前格子要构成正方形,必须满足:
  1. 左上c×c正方形向右下扩展1格
  2. 上a×a正方形向下扩展1格
  3. 左b×b正方形向右扩展1格

所以最大边长 = min(a, b, c) + 1
```

### Python代码

```python
def maximalSquare(matrix: List[List[str]]) -> int:
    """
    解法二:二维DP(最优解)
    思路:dp[i][j]表示以(i,j)为右下角的最大正方形边长
    """
    if not matrix or not matrix[0]:
        return 0

    m, n = len(matrix), len(matrix[0])
    # dp[i][j] = 以(i,j)为右下角的最大正方形边长
    dp = [[0] * n for _ in range(m)]
    max_side = 0

    # 初始化第一行和第一列
    for i in range(m):
        dp[i][0] = int(matrix[i][0])
        max_side = max(max_side, dp[i][0])
    for j in range(n):
        dp[0][j] = int(matrix[0][j])
        max_side = max(max_side, dp[0][j])

    # DP填表
    for i in range(1, m):
        for j in range(1, n):
            if matrix[i][j] == '1':
                # 取三个邻居的最小值 + 1
                dp[i][j] = min(dp[i-1][j], dp[i][j-1], dp[i-1][j-1]) + 1
                max_side = max(max_side, dp[i][j])

    return max_side * max_side


# ✅ 测试
matrix1 = [["1","0","1","0","0"],["1","0","1","1","1"],["1","1","1","1","1"],["1","0","0","1","0"]]
print(maximalSquare(matrix1))  # 期望输出:4

matrix2 = [["0","1"],["1","0"]]
print(maximalSquare(matrix2))  # 期望输出:1

matrix3 = [["1"]]
print(maximalSquare(matrix3))  # 期望输出:1
```

### 复杂度分析

- **时间复杂度**:O(m·n) — 遍历矩阵一次,每个格子O(1)计算
  - 具体地说:如果m=n=300,只需要300×300=90,000次操作,非常快
- **空间复杂度**:O(m·n) — DP表的大小

### 为什么是最优解

- ✅ 时间复杂度O(m·n)已经是最优(至少要遍历矩阵一次)
- ✅ 空间复杂度O(m·n)可以优化到O(n)(见解法三),但面试中这个版本更清晰
- ✅ 代码简洁,状态转移公式优雅
- ✅ 通过"右下角推导"巧妙避免了枚举所有正方形的暴力

---

## ⚡ 解法三:空间优化DP(进阶)

### 优化思路

注意到`dp[i][j]`只依赖于上一行的`dp[i-1][j]`、`dp[i-1][j-1]`和当前行的`dp[i][j-1]`,可以用滚动数组将空间优化到O(n)。

> 💡 **关键想法**:只保留上一行的DP结果,滚动更新当前行!

### Python代码

```python
def maximalSquare_optimized(matrix: List[List[str]]) -> int:
    """
    解法三:空间优化DP
    思路:滚动数组优化空间到O(n)
    """
    if not matrix or not matrix[0]:
        return 0

    m, n = len(matrix), len(matrix[0])
    dp = [0] * n
    max_side = 0
    prev = 0  # 保存左上角的值(dp[i-1][j-1])

    # 初始化第一行
    for j in range(n):
        dp[j] = int(matrix[0][j])
        max_side = max(max_side, dp[j])

    # DP滚动更新
    for i in range(1, m):
        for j in range(n):
            temp = dp[j]  # 保存当前值(下一轮的左上角)
            if j == 0:
                dp[j] = int(matrix[i][j])
            elif matrix[i][j] == '1':
                # dp[j]是上方,dp[j-1]是左方,prev是左上方
                dp[j] = min(dp[j], dp[j-1], prev) + 1
                max_side = max(max_side, dp[j])
            else:
                dp[j] = 0
            prev = temp  # 更新左上角值
        prev = 0  # 每行开始时重置

    return max_side * max_side


# ✅ 测试
matrix1 = [["1","0","1","0","0"],["1","0","1","1","1"],["1","1","1","1","1"],["1","0","0","1","0"]]
print(maximalSquare_optimized(matrix1))  # 期望输出:4
```

### 复杂度分析

- **时间复杂度**:O(m·n) — 与解法二相同
- **空间复杂度**:O(n) — 只用一维数组,节省空间

---

## 🐍 Pythonic 写法

利用Python的`zip`和列表推导式,可以写出更简洁的版本:

```python
def maximalSquare_pythonic(matrix: List[List[str]]) -> int:
    """Pythonic版本:利用zip简化边界处理"""
    if not matrix:
        return 0

    m, n = len(matrix), len(matrix[0])
    # 在上方和左侧添加虚拟边界(全为0),避免边界判断
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    max_side = 0

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if matrix[i-1][j-1] == '1':
                dp[i][j] = min(dp[i-1][j], dp[i][j-1], dp[i-1][j-1]) + 1
                max_side = max(max_side, dp[i][j])

    return max_side * max_side
```

这个写法通过添加虚拟边界,无需单独初始化第一行第一列,代码更简洁。

> ⚠️ **面试建议**:先写清晰版本(解法二)展示思路,再提空间优化(解法三)或Pythonic写法展示优化能力。

---

## 📊 解法对比

| 维度 | 解法一:暴力枚举 | 🏆 解法二:二维DP(最优) | 解法三:空间优化DP |
|------|--------------|---------------------|----------------|
| 时间复杂度 | O(m·n·min²) | **O(m·n)** ← 时间最优 | **O(m·n)** |
| 空间复杂度 | O(1) | O(m·n) | **O(n)** ← 空间更优 |
| 代码难度 | 简单 | 简单 | 中等(需理解滚动数组) |
| 面试推荐 | ⭐ | **⭐⭐⭐** ← 首选 | ⭐⭐ |
| 适用场景 | 仅用于理解问题 | **面试首选,清晰易懂** | 追问空间优化时使用 |

**为什么解法二是最优解**:
- 时间复杂度O(m·n)已经是理论最优(必须遍历矩阵)
- 空间复杂度O(m·n)虽然可以优化到O(n),但面试中清晰版本更重要
- 状态转移公式优雅:`min(三个邻居) + 1`
- 代码结构清晰,符合二维DP标准模板

**面试建议**:
1. 先花30秒说明暴力法思路(枚举所有正方形),但时间复杂度O(m·n·min²)较高
2. 重点讲解🏆二维DP的核心思想:"以右下角为基准,取三个邻居的最小值+1"
3. 强调状态定义:`dp[i][j]`表示以`(i,j)`为右下角的最大正方形边长
4. 展示状态转移:`dp[i][j] = min(dp[i-1][j], dp[i][j-1], dp[i-1][j-1]) + 1`(当matrix[i][j]='1')
5. 如果面试官追问空间优化,再展示解法三的滚动数组技巧

---

## 🎤 面试现场

> 模拟面试中的完整对话流程,帮你练习"边想边说"。

**面试官**:请你解决一下这道题。

**你**:(审题30秒)好的,这道题要求找到矩阵中只包含'1'的最大正方形的面积。我的第一个想法是暴力枚举所有可能的正方形,时间复杂度O(m·n·min(m,n)²),对于300×300的矩阵会比较慢。

我注意到这是一个典型的**二维DP问题**。关键洞察是:以每个格子为右下角的最大正方形边长,取决于它的上、左、左上三个邻居的最小值。

我会定义`dp[i][j]`为以`(i,j)`为右下角的最大正方形边长。状态转移是`dp[i][j] = min(三个邻居) + 1`(当matrix[i][j]='1')。时间复杂度优化到O(m·n)。

**面试官**:为什么是三个邻居的最小值?

**你**:因为要构成正方形,必须保证:
1. 上方有边长为a的正方形
2. 左方有边长为b的正方形
3. 左上方有边长为c的正方形

当前格子要扩展正方形,边长受限于三者的最小值。举例:如果上方只有边长1的正方形,那当前格子最多只能扩展到边长2。

**面试官**:很好,请写一下代码。

**你**:(边写边说关键步骤)
```python
# 1. 初始化DP表
dp = [[0] * n for _ in range(m)]

# 2. 第一行第一列直接复制matrix
for i in range(m):
    dp[i][0] = int(matrix[i][0])
for j in range(n):
    dp[0][j] = int(matrix[0][j])

# 3. DP填表
for i in range(1, m):
    for j in range(1, n):
        if matrix[i][j] == '1':
            dp[i][j] = min(dp[i-1][j], dp[i][j-1], dp[i-1][j-1]) + 1
            max_side = max(max_side, dp[i][j])
```

**面试官**:测试一下?

**你**:用示例`[["1","0","1","0","0"],["1","0","1","1","1"],["1","1","1","1","1"],["1","0","0","1","0"]]`走一遍...(手动模拟关键位置的DP值)。在位置(2,3)时,三个邻居都是1,所以dp[2][3]=2,是最大边长。最终面积=4。再测边界情况`[["1"]]`,输出1。结果正确。

### 高频追问

| 追问 | 应答策略 |
|------|---------|
| "能不能优化空间?" | 可以,用滚动数组优化到O(n)。只保留上一行的DP结果,滚动更新当前行,需要一个变量保存左上角的值 |
| "如果求最大矩形呢?" | 那是第37课(LeetCode 85),需要用单调栈,逐行转化为柱状图最大矩形问题 |
| "如果矩阵很大,存不进内存?" | 可以流式处理,每次读入一行,用滚动数组更新DP,空间复杂度O(n) |
| "为什么不用前缀和?" | 前缀和能快速求区间和,但判断正方形是否全为1需要精确检查,DP更高效 |

---

## 🎓 知识点总结

### Python技巧卡片 🐍

```python
# 技巧1:添加虚拟边界简化边界判断
dp = [[0] * (n + 1) for _ in range(m + 1)]
# 这样无需单独处理第一行第一列

# 技巧2:滚动数组优化空间
dp = [0] * n
prev = 0  # 保存左上角的值
for i in range(m):
    for j in range(n):
        temp = dp[j]  # 保存当前值(下一轮的左上角)
        # 更新dp[j]
        prev = temp

# 技巧3:矩阵转字符串(调试用)
def print_matrix(matrix):
    for row in matrix:
        print(' '.join(map(str, row)))
```

### 💡 底层原理(选读)

> **二维DP的核心思想**:
> 1. **状态定义**:通常定义`dp[i][j]`表示以`(i,j)`为结束/右下角的最优解
> 2. **状态转移**:当前格子的值由相邻格子(上、左、左上等)推导
> 3. **边界处理**:第一行第一列需要单独初始化,或添加虚拟边界简化
> 4. **优化方向**:如果只依赖上一行,可以用滚动数组优化空间
>
> **本题的巧妙之处**:
> - 状态转移公式`min(三个邻居) + 1`简洁优雅
> - 通过"以右下角为基准"的定义,避免了枚举所有正方形的暴力
> - 可以扩展到求最大矩形、最大加号等类似问题

### 算法模式卡片 📐

- **模式名称**:二维DP(网格DP)
- **适用条件**:
  1. 问题涉及二维矩阵或网格
  2. 当前格子的最优解可以由相邻格子推导
  3. 需要求最大/最小/计数等优化问题
- **识别关键词**:
  - "二维矩阵"、"网格"、"正方形"、"矩形"
  - "最大面积"、"最长路径"、"路径计数"
  - 题目要求优化某个二维区域的属性
- **模板代码**:
```python
def grid_dp(matrix):
    m, n = len(matrix), len(matrix[0])
    dp = [[0] * n for _ in range(m)]

    # 初始化第一行第一列
    for i in range(m):
        dp[i][0] = init_value(matrix[i][0])
    for j in range(n):
        dp[0][j] = init_value(matrix[0][j])

    # DP填表
    for i in range(1, m):
        for j in range(1, n):
            dp[i][j] = transition_function(
                dp[i-1][j],    # 上
                dp[i][j-1],    # 左
                dp[i-1][j-1]   # 左上
            )

    return dp[m-1][n-1]  # 或 max(max(row) for row in dp)
```

### 易错点 ⚠️

1. **状态定义错误**:
   - ❌ 错误:定义`dp[i][j]`为以`(i,j)`为左上角的最大正方形边长
   - ✅ 正确:定义为以`(i,j)`为右下角的最大正方形边长
   - 原因:以右下角为基准,可以从三个邻居(上、左、左上)推导,状态转移清晰

2. **边界初始化遗漏**:
   - ❌ 错误:第一行第一列没有单独初始化,直接从(1,1)开始DP
   - ✅ 正确:第一行第一列需要单独处理,或添加虚拟边界
   - 原因:`dp[i][j]`依赖于`dp[i-1][j]`、`dp[i][j-1]`、`dp[i-1][j-1]`,边界没有依赖项

3. **返回值错误**:
   - ❌ 错误:返回`max_side`(最大边长)
   - ✅ 正确:返回`max_side * max_side`(面积)
   - 原因:题目要求返回面积,不是边长

4. **空间优化时左上角处理**:
   - ❌ 错误:滚动数组时忘记保存左上角的值(`dp[i-1][j-1]`)
   - ✅ 正确:用`prev`变量保存上一轮的`dp[j]`(即左上角的值)
   - 原因:滚动更新时,`dp[j]`会被覆盖,需要提前保存

---

## 🏗️ 工程实战(选读)

> 这个算法思想在真实项目中的应用,让你知道"学了有什么用"。

- **场景1**:图像处理 — 在二值图像中检测最大的白色正方形区域,用于目标识别或压缩算法
- **场景2**:芯片设计 — 在集成电路布局中,找到最大的可用正方形区域放置功能模块
- **场景3**:地图应用 — 在卫星地图中检测最大的空地/绿地区域,用于城市规划或建筑选址

---

## 🏋️ 举一反三

完成本课后,试试这些同类题目来巩固知识:

| 题目 | 难度 | 相关知识点 | 提示 |
|------|------|-----------|------|
| LeetCode 85. 最大矩形 | Hard | 单调栈 + 逐行DP | 逐行转化为柱状图最大矩形问题(第36课) |
| LeetCode 1277. 统计全为1的正方形子矩阵 | Medium | 二维DP | 与本题几乎相同,DP定义略有不同 |
| LeetCode 764. 最大加号标志 | Medium | 二维DP | 类似思路,但要考虑四个方向的最小值 |
| LeetCode 1914. 循环轮转矩阵 | Medium | 二维DP(变体) | 需要考虑矩阵旋转对DP的影响 |

---

## 📝 课后小测

试试这道变体题,不要看答案,自己先想5分钟!

**题目**:给定一个01矩阵,找出只包含1的最大**矩形**的面积(不一定是正方形)。例如:
```
matrix = [
  ["1","0","1","0","0"],
  ["1","0","1","1","1"],
  ["1","1","1","1","1"],
  ["1","0","0","1","0"]
]
输出:6 (2×3的矩形)
```

<details>
<summary>💡 提示(实在想不出来再点开)</summary>

逐行构建柱状图,每一行都是一个"柱状图最大矩形"问题(第36课),用单调栈求解。

</details>

<details>
<summary>✅ 参考答案</summary>

```python
def maximalRectangle(matrix: List[List[str]]) -> int:
    """
    逐行构建柱状图 + 单调栈求最大矩形
    """
    if not matrix or not matrix[0]:
        return 0

    n = len(matrix[0])
    heights = [0] * n
    max_area = 0

    # 逐行处理
    for row in matrix:
        # 更新柱状图高度
        for j in range(n):
            if row[j] == '1':
                heights[j] += 1
            else:
                heights[j] = 0

        # 对当前柱状图求最大矩形(单调栈)
        max_area = max(max_area, largestRectangleArea(heights))

    return max_area


def largestRectangleArea(heights: List[int]) -> int:
    """单调栈求柱状图最大矩形(第36课)"""
    stack = []
    max_area = 0
    heights.append(0)

    for i, h in enumerate(heights):
        while stack and heights[stack[-1]] > h:
            height = heights[stack.pop()]
            width = i if not stack else i - stack[-1] - 1
            max_area = max(max_area, height * width)
        stack.append(i)

    heights.pop()
    return max_area
```

核心思路:
1. 逐行扫描矩阵,维护每一列的"连续1的高度"作为柱状图
2. 对每一行的柱状图,用单调栈求最大矩形面积
3. 时间复杂度O(m·n),空间复杂度O(n)

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
