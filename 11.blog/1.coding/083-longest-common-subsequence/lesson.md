# 📖 第83课:最长公共子序列

> **模块**:动态规划 | **难度**:Medium ⭐⭐⭐
> **LeetCode 链接**:https://leetcode.cn/problems/longest-common-subsequence/
> **前置知识**:第71课(爬楼梯)、第72课(杨辉三角)
> **预计学习时间**:30分钟

---

## 🎯 题目描述

给定两个字符串 text1 和 text2,返回它们的最长公共子序列的长度。如果不存在公共子序列,返回 0。

一个字符串的子序列是指:在不改变剩余字符顺序的前提下,删除某些字符(也可以不删)后组成的新字符串。

**示例:**
```
输入:text1 = "abcde", text2 = "ace"
输出:3
解释:最长公共子序列是 "ace",长度为 3
```

```
输入:text1 = "abc", text2 = "abc"
输出:3
解释:最长公共子序列是 "abc",长度为 3
```

```
输入:text1 = "abc", text2 = "def"
输出:0
解释:两个字符串没有公共子序列
```

**约束条件:**
- 1 <= text1.length, text2.length <= 1000
- text1 和 text2 仅由小写英文字符组成

---

### 🧪 边界用例(面试必考)

| 用例类型 | 输入 | 期望输出 | 考察点 |
|---------|------|---------|--------|
| 完全相同 | "abc", "abc" | 3 | 基本功能 |
| 完全不同 | "abc", "def" | 0 | 无交集处理 |
| 空字符串 | "", "abc" | 0 | 边界处理 |
| 单字符 | "a", "a" | 1 | 最小有效输入 |
| 逆序 | "abc", "cba" | 1 | 仅一个字符匹配 |
| 大规模 | 1000长度字符串 | — | 性能边界 |

---

## 💡 思路引导

### 生活化比喻
> 想象你和朋友各自独立观看了一场电影,并且各自记录了印象深刻的场景。
>
> 🐌 **笨办法**:你们想知道有多少场景是共同记住的。最笨的方法是你列出所有可能的场景组合(比如第1、3、5场,或第2、4场等),然后逐个检查朋友是否也按这个顺序记住了这些场景。这样的组合数量是指数级的,非常慢!
>
> 🚀 **聪明办法**:你们可以从头开始对比。如果当前场景都记住了,那么"共同记忆长度+1";如果某个人不记得当前场景,那就跳过这个场景,看下一个。关键是用一个表格记录"到目前为止的最长共同记忆",避免重复比较。

### 关键洞察
**二维DP的本质:用表格记录"前i个vs前j个"的子问题答案,避免重复计算。**

---

## 🧠 解题思维链

> 这一节模拟你在面试中"从零开始思考"的过程。

### Step 1:理解题目 → 锁定输入输出
- **输入**:两个字符串 text1 和 text2,长度分别为 m 和 n
- **输出**:最长公共子序列的长度(整数)
- **限制**:子序列可以不连续,但必须保持原相对顺序

### Step 2:先想笨办法(暴力法)
枚举 text1 的所有子序列(2^m 种),对每个子序列检查是否也是 text2 的子序列。
- 时间复杂度:O(2^m * n)
- 瓶颈在哪:子序列数量呈指数增长,根本无法接受

### Step 3:瓶颈分析 → 优化方向
观察暴力法的重复计算:
- 比如计算 "abc" 和 "ac" 时,我们会多次计算 "ab" 和 "a" 的 LCS
- 核心问题:同一个子问题(前 i 个字符 vs 前 j 个字符)被重复计算
- 优化思路:能否用一个表格存储子问题答案,查表代替重算?

### Step 4:选择武器
- 选用:**二维动态规划**
- 理由:
  1. 具有最优子结构:LCS(text1, text2) 可以由 LCS(text1[:-1], text2[:-1]) 等子问题推导
  2. 存在重叠子问题:大量子问题被重复计算
  3. 二维DP表可以记录所有子问题的答案

> 🔑 **模式识别提示**:当题目涉及"两个字符串的匹配、对比、编辑"时,优先考虑"二维DP"

---

## 🔑 解法一:递归暴力搜索(理解思路)

### 思路
直接用递归定义 LCS:
- 如果 text1[i] == text2[j],那么 LCS = 1 + LCS(i-1, j-1)
- 否则,LCS = max(LCS(i-1, j), LCS(i, j-1))

### 图解过程

```
示例:text1="ace", text2="abcde"

递归树(部分展示):
           lcs("ace", "abcde")
          /                    \
    lcs("ac", "abcde")       lcs("ace", "abcd")
    (e != e? 不,这里简化)
         /      \
   lcs("a", "abcde")  lcs("ac", "abcd")
       ...              ...
                (大量重复计算)

问题:同一个 lcs("ac", "abcd") 会被计算多次!
```

### Python代码

```python
def longest_common_subsequence_recursive(text1: str, text2: str) -> int:
    """
    解法一:递归暴力搜索
    思路:直接用递归定义求解,会超时
    """
    def lcs(i: int, j: int) -> int:
        # 基础情况:任一字符串为空
        if i < 0 or j < 0:
            return 0

        # 如果当前字符相同
        if text1[i] == text2[j]:
            return 1 + lcs(i - 1, j - 1)
        else:
            # 否则取跳过 text1[i] 或 text2[j] 的较大值
            return max(lcs(i - 1, j), lcs(i, j - 1))

    return lcs(len(text1) - 1, len(text2) - 1)


# ✅ 测试(小规模输入)
print(longest_common_subsequence_recursive("ace", "ace"))  # 期望输出:3
print(longest_common_subsequence_recursive("abc", "def"))  # 期望输出:0
# print(longest_common_subsequence_recursive("abcde", "ace"))  # 会超时
```

### 复杂度分析
- **时间复杂度**:O(2^(m+n)) — 每个字符都有"选或不选"两种分支,指数爆炸
  - 具体地说:如果 m=n=10,大约需要 2^20 = 100万次递归调用
- **空间复杂度**:O(m+n) — 递归栈深度

### 优缺点
- ✅ 思路清晰,直接体现递归定义
- ❌ 时间复杂度太高,存在大量重复计算,无法通过测试

---

## ⚡ 解法二:记忆化递归(剪枝优化)

### 优化思路
在递归基础上加入备忘录(memo),将计算过的子问题结果存起来,遇到重复子问题直接返回。

> 💡 **关键想法**:用字典记录 memo[(i, j)] = LCS长度,避免重复计算

### 图解过程

```
text1="ace", text2="abcde"

备忘录表(逐步填充):
第一次计算 lcs(2, 4) → 存入 memo[(2, 4)] = 3
后续再遇到 lcs(2, 4) → 直接返回 memo[(2, 4)]

对比递归树:
- 暴力递归:每个节点都重新计算
- 记忆化:同一个 (i, j) 只计算一次
```

### Python代码

```python
def longest_common_subsequence_memo(text1: str, text2: str) -> int:
    """
    解法二:记忆化递归
    思路:用字典缓存子问题结果
    """
    memo = {}

    def lcs(i: int, j: int) -> int:
        if i < 0 or j < 0:
            return 0

        # 查备忘录
        if (i, j) in memo:
            return memo[(i, j)]

        if text1[i] == text2[j]:
            result = 1 + lcs(i - 1, j - 1)
        else:
            result = max(lcs(i - 1, j), lcs(i, j - 1))

        # 存入备忘录
        memo[(i, j)] = result
        return result

    return lcs(len(text1) - 1, len(text2) - 1)


# ✅ 测试
print(longest_common_subsequence_memo("abcde", "ace"))  # 期望输出:3
print(longest_common_subsequence_memo("abc", "abc"))  # 期望输出:3
print(longest_common_subsequence_memo("abc", "def"))  # 期望输出:0
```

### 复杂度分析
- **时间复杂度**:O(m*n) — 每个 (i, j) 子问题只计算一次,共 m*n 个子问题
- **空间复杂度**:O(m*n) — 备忘录存储 + 递归栈

---

## 🏆 解法三:二维DP表(最优解)

### 优化思路
将记忆化递归改为自底向上的DP表填充,消除递归调用开销。

> 💡 **关键想法**:用 dp[i][j] 表示 text1[0..i-1] 和 text2[0..j-1] 的 LCS 长度

### 图解过程

```
示例:text1="ace", text2="abcde"

构建 DP 表(行=text1,列=text2):

     ""  a  b  c  d  e
""   0   0  0  0  0  0
a    0   1  1  1  1  1  ← text1[0]='a' 匹配 text2[0]='a',dp[1][1]=1
c    0   1  1  2  2  2  ← text1[1]='c' 匹配 text2[2]='c',dp[2][3]=dp[1][2]+1=2
e    0   1  1  2  2  3  ← text1[2]='e' 匹配 text2[4]='e',dp[3][5]=dp[2][4]+1=3

状态转移:
- 如果 text1[i-1]==text2[j-1]:dp[i][j]=dp[i-1][j-1]+1
- 否则:dp[i][j]=max(dp[i-1][j], dp[i][j-1])

关键理解:
- dp[i-1][j]:跳过 text1 当前字符
- dp[i][j-1]:跳过 text2 当前字符
- dp[i-1][j-1]+1:两个字符匹配,长度+1
```

### Python代码

```python
def longest_common_subsequence(text1: str, text2: str) -> int:
    """
    解法三:二维DP表(最优解)
    思路:自底向上填充 DP 表
    """
    m, n = len(text1), len(text2)
    # 初始化 DP 表,多一行一列处理边界
    dp = [[0] * (n + 1) for _ in range(m + 1)]

    # 填充 DP 表
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if text1[i - 1] == text2[j - 1]:
                # 字符匹配:长度+1
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                # 字符不匹配:取跳过其中一个的最大值
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])

    return dp[m][n]


# ✅ 测试
print(longest_common_subsequence("abcde", "ace"))  # 期望输出:3
print(longest_common_subsequence("abc", "abc"))  # 期望输出:3
print(longest_common_subsequence("abc", "def"))  # 期望输出:0
print(longest_common_subsequence("", "abc"))  # 期望输出:0
print(longest_common_subsequence("a", "a"))  # 期望输出:1
```

### 复杂度分析
- **时间复杂度**:O(m*n) — 双重循环填充表格,每个格子计算一次
  - 具体地说:如果 m=1000, n=1000,大约需要 100万次操作
- **空间复杂度**:O(m*n) — DP 表空间

### 为什么是最优解
- 时间已达理论最优:必须至少考察所有字符组合才能得出答案
- 代码清晰易懂,面试中容易写对
- 可以进一步优化空间到 O(min(m,n)),但时间无法再降

---

## 🚀 解法四:空间优化DP(进阶)

### 优化思路
观察到 dp[i][j] 只依赖于 dp[i-1][j-1]、dp[i-1][j]、dp[i][j-1],因此可以用两行滚动数组代替整个表格。

> 💡 **关键想法**:只保留"上一行"和"当前行",节省空间

### Python代码

```python
def longest_common_subsequence_optimized(text1: str, text2: str) -> int:
    """
    解法四:空间优化DP
    思路:用滚动数组降低空间复杂度
    """
    m, n = len(text1), len(text2)
    # 确保 text2 是较短的字符串,优化空间
    if m < n:
        text1, text2 = text2, text1
        m, n = n, m

    # 只需要两行:上一行和当前行
    prev = [0] * (n + 1)
    curr = [0] * (n + 1)

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if text1[i - 1] == text2[j - 1]:
                curr[j] = prev[j - 1] + 1
            else:
                curr[j] = max(prev[j], curr[j - 1])
        # 滚动数组:当前行变成下一轮的上一行
        prev, curr = curr, prev

    return prev[n]


# ✅ 测试
print(longest_common_subsequence_optimized("abcde", "ace"))  # 期望输出:3
print(longest_common_subsequence_optimized("abc", "abc"))  # 期望输出:3
print(longest_common_subsequence_optimized("abc", "def"))  # 期望输出:0
```

### 复杂度分析
- **时间复杂度**:O(m*n) — 与解法三相同
- **空间复杂度**:O(min(m,n)) — 只需两行数组,总空间为 2*min(m,n)

---

## 🐍 Pythonic 写法

利用 Python 的函数式特性简化代码:

```python
from functools import lru_cache

def longest_common_subsequence_lru(text1: str, text2: str) -> int:
    """
    Pythonic 写法:用 @lru_cache 自动缓存
    """
    @lru_cache(maxsize=None)
    def lcs(i: int, j: int) -> int:
        if i < 0 or j < 0:
            return 0
        if text1[i] == text2[j]:
            return 1 + lcs(i - 1, j - 1)
        return max(lcs(i - 1, j), lcs(i, j - 1))

    return lcs(len(text1) - 1, len(text2) - 1)

# ✅ 测试
print(longest_common_subsequence_lru("abcde", "ace"))  # 期望输出:3
```

**解释**:
- `@lru_cache` 装饰器自动实现记忆化,无需手动维护 memo 字典
- 代码更简洁,但性能与解法二相当(递归有开销)

> ⚠️ **面试建议**:先写清晰的二维DP版本(解法三),再提 Pythonic 写法展示语言功底。
> 面试官更看重你的**DP状态设计思路**,而非代码行数。

---

## 📊 解法对比

| 维度 | 解法一:暴力递归 | 解法二:记忆化递归 | 🏆 解法三:二维DP表(最优) | 解法四:空间优化DP |
|------|--------------|----------------|---------------------|---------------|
| 时间复杂度 | O(2^(m+n)) | O(m*n) | **O(m*n)** ← 时间最优 | O(m*n) |
| 空间复杂度 | O(m+n) | O(m*n) | O(m*n) | **O(min(m,n))** ← 空间最优 |
| 代码难度 | 简单 | 中等 | 中等 | 较难 |
| 面试推荐 | ⭐ | ⭐⭐ | **⭐⭐⭐** ← 首选 | ⭐⭐ |
| 适用场景 | 理解递归定义 | 辅助理解DP | **面试首选,清晰易懂** | 空间受限场景 |

**为什么解法三是最优解**:
- 时间复杂度 O(m*n) 已经是理论最优(必须检查所有字符对)
- 代码结构清晰,DP 表可视化强,面试中容易讲解和调试
- 空间虽然是 O(m*n),但对于题目约束(最大1000)完全可接受

**面试建议**:
1. 先用1分钟口述暴力递归思路(解法一),表明你理解问题本质
2. 立即优化到🏆最优解(解法三:二维DP表),展示DP设计能力
3. **重点讲解DP状态转移方程**:"字符匹配则+1,不匹配则取max"
4. 如果面试官追问空间优化,再展示解法四
5. 手动在纸上画一个小 DP 表(如 3x3),演示状态转移过程

---

## 🎤 面试现场

> 模拟面试中的完整对话流程,帮你练习"边想边说"。

**面试官**:请你解决一下这道题——最长公共子序列。

**你**:(审题30秒)好的,这道题要求找两个字符串的最长公共子序列长度。这是一个经典的二维DP问题。

我的第一个想法是暴力递归:定义 lcs(i, j) 表示 text1[0..i] 和 text2[0..j] 的 LCS。如果当前字符相同,结果就是 1 + lcs(i-1, j-1);否则就是跳过其中一个字符的最大值。但这样会有指数级的重复计算。

所以我会用二维DP来优化,用一个 dp[m+1][n+1] 的表格,dp[i][j] 表示 text1 前 i 个字符和 text2 前 j 个字符的 LCS 长度。状态转移方程是:
- 如果 text1[i-1] == text2[j-1],dp[i][j] = dp[i-1][j-1] + 1
- 否则,dp[i][j] = max(dp[i-1][j], dp[i][j-1])

时间复杂度是 O(m*n),空间复杂度也是 O(m*n)。

**面试官**:很好,请写一下代码。

**你**:(边写边说)
```python
def longestCommonSubsequence(text1: str, text2: str) -> int:
    m, n = len(text1), len(text2)
    # 初始化 DP 表,第 0 行/列全为 0 表示空字符串
    dp = [[0] * (n + 1) for _ in range(m + 1)]

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if text1[i-1] == text2[j-1]:
                dp[i][j] = dp[i-1][j-1] + 1  # 字符匹配
            else:
                dp[i][j] = max(dp[i-1][j], dp[i][j-1])  # 取较大者

    return dp[m][n]
```

**面试官**:测试一下?

**你**:用示例 "ace" 和 "abcde" 走一遍。我在纸上画个 4x6 的表格...(手动模拟)
- 当 i=1, j=1 时,text1[0]='a' == text2[0]='a',dp[1][1]=1
- 当 i=2, j=3 时,text1[1]='c' == text2[2]='c',dp[2][3]=2
- 最终 dp[3][5]=3

再测一个边界情况:空字符串 "" 和 "abc",因为第0行全为0,直接返回0。结果正确。

### 高频追问

| 追问 | 应答策略 |
|------|---------|
| "还有更优解吗?" | 时间 O(m*n) 已经最优,但空间可以优化到 O(min(m,n)),用滚动数组只保留两行。不过面试中通常不要求这个优化。 |
| "如何输出具体的LCS字符串?" | 需要从 dp[m][n] 反向回溯:如果 text1[i-1]==text2[j-1] 就记录该字符并移到 dp[i-1][j-1],否则往较大值方向移动。 |
| "空间能不能O(1)?" | 不能,因为DP状态依赖于历史信息,至少需要 O(n) 存储一行数据。 |
| "如果要求最长公共子串(连续)呢?" | 那是另一道题,DP状态转移会不同:只有字符匹配时才能累加,不匹配时直接置0。 |

---

## 🎓 知识点总结

### Python技巧卡片 🐍
```python
# 技巧1:二维列表初始化 — 避免浅拷贝陷阱
dp = [[0] * (n + 1) for _ in range(m + 1)]  # ✅ 正确
# dp = [[0] * (n + 1)] * (m + 1)  # ❌ 错误:所有行是同一个对象

# 技巧2:索引对齐 — DP表多一行一列
# dp[i][j] 对应 text1[i-1] 和 text2[j-1],第0行/列表示空字符串

# 技巧3:滚动数组技巧 — 降低空间复杂度
prev, curr = curr, prev  # 交换两行,无需重新分配内存
```

### 💡 底层原理(选读)

> **为什么二维DP适合双字符串匹配问题?**
>
> 1. **状态表示**:dp[i][j] 天然对应"前i个 vs 前j个"的组合空间
> 2. **状态转移**:当前状态只依赖于"左上、上、左"三个相邻状态,符合局部性原理
> 3. **边界处理**:第0行/列表示空字符串,避免特殊判断
>
> **LCS 与编辑距离的联系**:
> - LCS 关注"相同部分有多长"
> - 编辑距离关注"不同部分要改多少"
> - 公式:编辑距离 ≈ (m+n) - 2*LCS (在只允许插入删除时)

### 算法模式卡片 📐
- **模式名称**:二维DP(双字符串匹配)
- **适用条件**:两个序列的匹配、对比、编辑问题
- **识别关键词**:最长公共、编辑距离、交错字符串、通配符匹配
- **模板代码**:
```python
def two_string_dp(s1: str, s2: str) -> int:
    m, n = len(s1), len(s2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]

    # 初始化第0行/列(根据具体题目)
    for i in range(m + 1):
        dp[i][0] = 初始值
    for j in range(n + 1):
        dp[0][j] = 初始值

    # 状态转移
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if s1[i-1] == s2[j-1]:
                dp[i][j] = dp[i-1][j-1] + ...  # 匹配时的逻辑
            else:
                dp[i][j] = min/max(dp[i-1][j], dp[i][j-1], ...)  # 不匹配时的逻辑

    return dp[m][n]
```

### 易错点 ⚠️
1. **索引越界**:dp[i][j] 对应 text1[i-1],写成 text1[i] 会越界
   - **为什么错**:DP表比原字符串多一行一列
   - **正确做法**:始终用 i-1 和 j-1 访问原字符串

2. **初始化错误**:忘记初始化第0行/列为0
   - **为什么错**:第0行/列表示空字符串,LCS长度应该是0
   - **正确做法**:初始化时显式创建 (m+1)x(n+1) 全0表格

3. **状态转移搞反**:把 dp[i-1][j-1] 写成 dp[i+1][j+1]
   - **为什么错**:DP是从小到大推导,依赖的是"已计算的历史状态"
   - **正确做法**:画一个小表格,确认依赖关系(左上、上、左)

---

## 🏗️ 工程实战(选读)

> 这个算法思想在真实项目中的应用,让你知道"学了有什么用"。

- **场景1**:版本控制系统(Git)的 diff 算法 — 比较两个文件版本,找出最长的相同代码块
- **场景2**:DNA序列比对 — 生物信息学中比较两条基因序列的相似度,LCS 长度越大越相似
- **场景3**:抄袭检测 — 比较两篇文章的句子序列,找出最长的公共段落
- **场景4**:智能合并工具 — 合并两个代码分支时,基于 LCS 找出共同的基础部分

---

## 🏋️ 举一反三

完成本课后,试试这些同类题目来巩固知识:

| 题目 | 难度 | 相关知识点 | 提示 |
|------|------|-----------|------|
| LeetCode 72. 编辑距离 | Hard | 二维DP、字符串匹配 | 状态转移多了"替换"操作,三种操作取最小 |
| LeetCode 583. 两个字符串的删除操作 | Medium | 二维DP、LCS | 删除次数 = (m+n) - 2*LCS |
| LeetCode 712. 两个字符串的最小ASCII删除和 | Medium | 二维DP | 类似LCS,但要记录删除的字符ASCII和 |
| LeetCode 1035. 不相交的线 | Medium | 二维DP、LCS变形 | 本质上就是求LCS,换了一个几何描述 |
| LeetCode 516. 最长回文子序列 | Medium | 二维DP、区间DP | 类似LCS,但是把字符串和它的反转求LCS |

---

## 📝 课后小测

试试这道变体题,不要看答案,自己先想5分钟!

**题目**:给定字符串 s,求它的最长回文子序列长度。(回文子序列可以不连续)

<details>
<summary>💡 提示(实在想不出来再点开)</summary>

把 s 和它的反转 reverse(s) 求 LCS 即可!因为回文的特点是"正着读和倒着读一样"。

</details>

<details>
<summary>✅ 参考答案</summary>

```python
def longest_palindrome_subsequence(s: str) -> int:
    """
    思路:s 的最长回文子序列 = LCS(s, reverse(s))
    """
    return longest_common_subsequence(s, s[::-1])

# 测试
print(longest_palindrome_subsequence("bbbab"))  # 输出:4 (bbbb)
print(longest_palindrome_subsequence("cbbd"))  # 输出:2 (bb)
```

**核心思路**:回文意味着从左往右和从右往左读是一样的,所以 s 和 reverse(s) 的公共部分就是回文部分。求它们的 LCS 即可得到最长回文子序列。

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
