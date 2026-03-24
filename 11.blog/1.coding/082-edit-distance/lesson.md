> 想系统提升编程能力、查看更完整的学习路线，欢迎访问 AI Compass：https://github.com/tingaicompass/AI-Compass
> 仓库持续更新刷题题解、Python 基础和 AI 实战内容，适合想高效进阶的你。

# 📖 第82课:编辑距离

> **模块**:动态规划 | **难度**:Medium ⭐⭐⭐
> **LeetCode 链接**:https://leetcode.cn/problems/edit-distance/
> **前置知识**:第83课(最长公共子序列)、字符串DP基础
> **预计学习时间**:35分钟

---

## 🎯 题目描述

给定两个字符串word1和word2,请计算将word1转换成word2所需的最少操作次数。你可以对一个字符串进行如下三种操作:
1. 插入一个字符
2. 删除一个字符
3. 替换一个字符

**示例:**
```
输入:word1 = "horse", word2 = "ros"
输出:3
解释:
horse -> rorse (将'h'替换为'r')
rorse -> rose  (删除'r')
rose  -> ros   (删除'e')
```

**约束条件:**
- 0 <= word1.length, word2.length <= 500
- word1和word2由小写英文字母组成

---

### 🧪 边界用例(面试必考)

| 用例类型 | 输入 | 期望输出 | 考察点 |
|---------|------|---------|--------|
| 空字符串 | word1="", word2="abc" | 3 | 需要插入3次 |
| 完全相同 | word1="abc", word2="abc" | 0 | 无需操作 |
| 完全不同 | word1="abc", word2="xyz" | 3 | 全部替换 |
| 一个空一个非空 | word1="a", word2="" | 1 | 删除1次 |
| 最大长度 | 两个500字符 | — | 性能边界 |

---

## 💡 思路引导

### 生活化比喻
> 想象你在用Word文档编辑一篇文章,要把原文(word1)改成目标版本(word2)。你有三个编辑工具:打字机(插入)、橡皮擦(删除)、涂改液(替换)。
>
> 🐌 **笨办法**:随机尝试各种编辑序列,记录哪个序列步数最少。但3种操作的组合爆炸,对于长度为n的字符串,可能的操作序列数量是天文数字。
>
> 🚀 **聪明办法**:想象两个字符串像两条路,你站在word1的某个位置i和word2的某个位置j。此时的"最少操作次数"只取决于:
> - 如果当前字符相同,不需要操作,直接看下一个字符
> - 如果不同,尝试三种操作,选最少的那个
>
> 这个问题的关键是:**子问题的最优解可以推导出大问题的最优解**。这正是DP的核心思想!

### 关键洞察
**编辑距离dp[i][j]表示word1的前i个字符转换为word2的前j个字符所需的最少操作数。可以通过比较word1[i-1]和word2[j-1]来递推。**

---

## 🧠 解题思维链

> 这一节模拟你在面试中"从零开始思考"的过程。

### Step 1:理解题目 → 锁定输入输出
- **输入**:两个字符串word1和word2
- **输出**:最少操作次数(整数)
- **限制**:只能进行插入、删除、替换三种操作

### Step 2:先想笨办法(暴力法)
用递归枚举所有可能的操作序列,对于每个位置尝试三种操作,计算最少步数。
- 时间复杂度:O(3^(m+n)) — 每个位置有3种选择
- 瓶颈在哪:大量重复子问题,比如word1[0:2]转换为word2[0:3]会被多次计算

### Step 3:瓶颈分析 → 优化方向
观察发现,相同的子问题(word1的前i个字符转换为word2的前j个字符)会被反复计算。
- 核心问题:重复计算子问题
- 优化思路:用二维表记录已计算的子问题结果 → 动态规划

### Step 4:选择武器
- 选用:**二维DP(双字符串匹配)**
- 理由:两个字符串的对比问题,需要二维状态表示"word1前i个字符"和"word2前j个字符"的关系

> 🔑 **模式识别提示**:当题目出现"两个字符串的转换/匹配/距离",优先考虑"二维DP"

---

## 🔑 解法一:递归+记忆化(自顶向下DP)

### 思路
从最终状态(word1全长,word2全长)开始递归,对于每个位置:
- 如果当前字符相同,跳过,看下一个字符
- 如果不同,尝试插入/删除/替换三种操作,取最小值

用哈希表记录已计算的状态,避免重复计算。

### 图解过程

```
word1 = "horse", word2 = "ros"

递归树(部分):
                    dp(5,3)
                  /    |    \
          插入'r'    删除'e'   替换'e'为's'
           /          |          \
        dp(5,2)    dp(4,3)      dp(4,2)
         ...        ...          ...

记忆化后,相同的(i,j)只计算一次
```

### Python代码

```python
from typing import Dict, Tuple


def minDistanceMemo(word1: str, word2: str) -> int:
    """
    解法一:递归+记忆化
    思路:自顶向下,用memo记录已计算的子问题
    """
    memo: Dict[Tuple[int, int], int] = {}

    def dp(i: int, j: int) -> int:
        """返回word1[0:i]转换为word2[0:j]的最少操作数"""
        # 记忆化剪枝
        if (i, j) in memo:
            return memo[(i, j)]

        # 基础情况:一个字符串为空
        if i == 0:
            return j  # 需要插入j个字符
        if j == 0:
            return i  # 需要删除i个字符

        # 当前字符相同,无需操作
        if word1[i-1] == word2[j-1]:
            result = dp(i-1, j-1)
        else:
            # 三种操作取最小
            insert_op = dp(i, j-1) + 1    # 在word1插入word2[j-1]
            delete_op = dp(i-1, j) + 1    # 删除word1[i-1]
            replace_op = dp(i-1, j-1) + 1 # 替换word1[i-1]为word2[j-1]
            result = min(insert_op, delete_op, replace_op)

        memo[(i, j)] = result
        return result

    return dp(len(word1), len(word2))


# ✅ 测试
print(minDistanceMemo("horse", "ros"))       # 期望输出:3
print(minDistanceMemo("intention", "execution"))  # 期望输出:5
print(minDistanceMemo("", "abc"))            # 期望输出:3
```

### 复杂度分析
- **时间复杂度**:O(m*n) — 共有m*n个子问题,每个计算一次
  - 具体地说:如果word1长度100,word2长度100,大约需要10,000次操作
- **空间复杂度**:O(m*n) — memo表 + 递归调用栈O(m+n)

### 优缺点
- ✅ 思路直观,容易理解递归关系
- ✅ 适合理解"状态转移"的含义
- ❌ 递归调用栈开销,可能栈溢出
- ❌ 不如迭代版本高效

---

## 🏆 解法二:二维DP迭代(最优解 — 标准做法)

### 优化思路
将递归改为迭代,建立二维DP表dp[i][j],从小到大填表,避免递归开销。

> 💡 **关键想法**:dp[i][j]的定义是"word1前i个字符转换为word2前j个字符的最少操作数"。状态转移分两种情况:
> 1. word1[i-1] == word2[j-1]:dp[i][j] = dp[i-1][j-1]
> 2. word1[i-1] != word2[j-1]:dp[i][j] = 1 + min(dp[i-1][j], dp[i][j-1], dp[i-1][j-1])

### 图解过程

```
word1 = "horse", word2 = "ros"

DP表构建过程(dp[i][j]表示word1前i个字符转换为word2前j个字符):

    ""  r  o  s
""   0  1  2  3
h    1  1  2  3
o    2  2  1  2
r    3  2  2  2
s    4  3  3  2
e    5  4  4  3

详细推导dp[5][3](horse -> ros):
word1[4]='e', word2[2]='s', 不相同
- 插入:dp[5][2] + 1 = 4 + 1 = 5
- 删除:dp[4][3] + 1 = 2 + 1 = 3
- 替换:dp[4][2] + 1 = 2 + 1 = 3
取最小值:dp[5][3] = 3

第一行初始化:[0, 1, 2, 3] (空串变为"r","ro","ros"需要插入1,2,3次)
第一列初始化:[0, 1, 2, 3, 4, 5] (各长度word1变为空串需要删除对应次数)
```

**第二个示例 — 完全相同的字符串:**
```
word1 = "abc", word2 = "abc"

    ""  a  b  c
""   0  1  2  3
a    1  0  1  2
b    2  1  0  1
c    3  2  1  0  ← 答案

对角线都是0(字符相同时继承左上角)
```

### Python代码

```python
def minDistance(word1: str, word2: str) -> int:
    """
    解法二:二维DP迭代
    思路:dp[i][j]表示word1前i个字符转换为word2前j个字符的最少操作数
    """
    m, n = len(word1), len(word2)
    dp = [[0] * (n+1) for _ in range(m+1)]

    # 初始化第一行(空串变为word2需要插入j次)
    for j in range(n+1):
        dp[0][j] = j

    # 初始化第一列(word1变为空串需要删除i次)
    for i in range(m+1):
        dp[i][0] = i

    # 填充DP表
    for i in range(1, m+1):
        for j in range(1, n+1):
            if word1[i-1] == word2[j-1]:
                # 字符相同,无需操作
                dp[i][j] = dp[i-1][j-1]
            else:
                # 字符不同,三种操作取最小
                dp[i][j] = 1 + min(
                    dp[i-1][j],    # 删除word1[i-1]
                    dp[i][j-1],    # 插入word2[j-1]
                    dp[i-1][j-1]   # 替换word1[i-1]为word2[j-1]
                )

    return dp[m][n]


# ✅ 测试
print(minDistance("horse", "ros"))            # 期望输出:3
print(minDistance("intention", "execution"))  # 期望输出:5
print(minDistance("abc", "abc"))              # 期望输出:0
```

### 复杂度分析
- **时间复杂度**:O(m*n) — 填充m*n的DP表
- **空间复杂度**:O(m*n) — DP表大小

---

## ⚡ 解法三:一维DP滚动数组(空间优化)

### 优化思路
观察到填表时,计算dp[i][j]只需要用到dp[i-1][j-1]、dp[i-1][j]和dp[i][j-1],即只需要"上一行"和"当前行的左边"。可以用一维数组滚动更新,节省空间。

> 💡 **关键技巧**:需要额外变量prev保存"左上角"的值(即dp[i-1][j-1]),因为更新dp[j]后会覆盖原来的值。

### Python代码

```python
def minDistanceOptimized(word1: str, word2: str) -> int:
    """
    解法三:一维DP滚动数组
    思路:用一维数组滚动更新,prev保存左上角值
    """
    m, n = len(word1), len(word2)
    dp = list(range(n+1))  # 初始化第一行:[0, 1, 2, ..., n]

    for i in range(1, m+1):
        prev = dp[0]  # 保存左上角(即dp[i-1][j-1])
        dp[0] = i     # 更新第一列

        for j in range(1, n+1):
            temp = dp[j]  # 保存更新前的dp[j](即dp[i-1][j])

            if word1[i-1] == word2[j-1]:
                dp[j] = prev
            else:
                dp[j] = 1 + min(dp[j], dp[j-1], prev)
                #               上方   左方    左上

            prev = temp  # prev更新为原来的dp[j],供下一轮使用

    return dp[n]


# ✅ 测试
print(minDistanceOptimized("horse", "ros"))  # 期望输出:3
print(minDistanceOptimized("intention", "execution"))  # 期望输出:5
```

### 复杂度分析
- **时间复杂度**:O(m*n) — 与解法二相同
- **空间复杂度**:O(n) — 只需要一维数组

---

## 🐍 Pythonic 写法

利用Python的zip和列表推导:

```python
# 使用zip同时迭代两个字符串
def minDistancePythonic(word1: str, word2: str) -> int:
    """Pythonic写法:利用zip简化代码"""
    m, n = len(word1), len(word2)
    if m == 0: return n
    if n == 0: return m

    dp = [[0] * (n+1) for _ in range(m+1)]
    dp[0] = list(range(n+1))
    for i in range(m+1):
        dp[i][0] = i

    for i, c1 in enumerate(word1, 1):
        for j, c2 in enumerate(word2, 1):
            dp[i][j] = dp[i-1][j-1] if c1 == c2 else \
                       1 + min(dp[i-1][j], dp[i][j-1], dp[i-1][j-1])

    return dp[m][n]
```

这个写法用enumerate简化了索引计算,更加Pythonic。

> ⚠️ **面试建议**:先写清晰版本展示思路,再提Pythonic写法展示语言功底。
> 面试官更看重你的**思考过程**,而非代码行数。

---

## 📊 解法对比

| 维度 | 解法一:递归+记忆化 | 🏆 解法二:二维DP(最优) | 解法三:一维DP |
|------|------------------|---------------------|-------------|
| 时间复杂度 | O(m*n) | **O(m*n)** ← 时间最优 | O(m*n) |
| 空间复杂度 | O(m*n) | **O(m*n)** ← 清晰易懂 | O(n) |
| 代码难度 | 中等 | 简单 | 较难 |
| 面试推荐 | ⭐⭐ | **⭐⭐⭐** ← 首选 | ⭐⭐ |
| 适用场景 | 理解递归关系 | **面试标准答案** | 空间受限场景 |

**为什么解法二是最优解**:
- 时间复杂度O(m*n)已经是最优(需要比较所有字符对)
- 代码清晰,易于理解和实现,面试中不容易出错
- 空间O(m*n)可接受,且便于理解DP转移过程
- 解法三虽然空间更优,但代码复杂度增加,性价比不高

---

## 🎤 面试现场

> 模拟面试中的完整对话流程,帮你练习"边想边说"。

**面试官**:请你解决一下这道"编辑距离"问题。

**你**:(审题30秒)好的,这道题要求计算将word1转换为word2的最少操作次数,可以进行插入、删除、替换三种操作。这是一道经典的DP问题。

我的第一个想法是用递归枚举所有可能的操作序列,但时间复杂度是O(3^(m+n)),会超时。

不过这是一个典型的双字符串匹配DP问题。我们可以用dp[i][j]表示word1前i个字符转换为word2前j个字符的最少操作数。

状态转移方程是:
- 如果word1[i-1] == word2[j-1],则dp[i][j] = dp[i-1][j-1]
- 否则dp[i][j] = 1 + min(dp[i-1][j], dp[i][j-1], dp[i-1][j-1])

时间复杂度O(m*n),空间复杂度O(m*n)。

**面试官**:很好,请写一下代码。

**你**:(边写边说)
```python
def minDistance(word1, word2):
    m, n = len(word1), len(word2)
    dp = [[0] * (n+1) for _ in range(m+1)]

    # 初始化边界
    for j in range(n+1):
        dp[0][j] = j  # 空串变为word2需要插入j次
    for i in range(m+1):
        dp[i][0] = i  # word1变为空串需要删除i次

    # 填充DP表
    for i in range(1, m+1):
        for j in range(1, n+1):
            if word1[i-1] == word2[j-1]:
                dp[i][j] = dp[i-1][j-1]  # 字符相同无需操作
            else:
                dp[i][j] = 1 + min(dp[i-1][j],   # 删除
                                   dp[i][j-1],   # 插入
                                   dp[i-1][j-1]) # 替换

    return dp[m][n]
```

**面试官**:三种操作分别对应什么?

**你**:
- dp[i-1][j] + 1:删除word1[i-1],然后word1前i-1个字符匹配word2前j个字符
- dp[i][j-1] + 1:在word1插入word2[j-1],然后word1前i个字符匹配word2前j-1个字符
- dp[i-1][j-1] + 1:替换word1[i-1]为word2[j-1],然后前面的字符匹配

**面试官**:测试一下?

**你**:用示例"horse" -> "ros"走一遍...
初始化后,逐行填表,最终dp[5][3] = 3。正确!

再测一个边界情况:word1="",word2="abc",返回3(需要插入3次)。正确!

### 高频追问

| 追问 | 应答策略 |
|------|---------|
| "还有更优解吗?" | "时间O(m*n)已经是最优(必须比较所有字符对)。空间可以优化到O(n)使用滚动数组,但代码复杂度会增加,通常不必要。" |
| "如何打印出具体的操作序列?" | "需要在DP过程中记录每个状态的选择(插入/删除/替换),最后从dp[m][n]回溯到dp[0][0],逆序输出操作序列。" |
| "如果只能替换不能插入删除呢?" | "那就是汉明距离问题,只需比较对应位置的字符,不同的数量就是答案,O(n)时间。" |
| "实际应用场景?" | "拼写检查、DNA序列比对、版本控制中的diff算法、语音识别中的词距计算等。" |

---

## 🎓 知识点总结

### Python技巧卡片 🐍
```python
# 技巧1:二维数组初始化 — 同时初始化边界
dp = [[0] * (n+1) for _ in range(m+1)]
for j in range(n+1):
    dp[0][j] = j
for i in range(m+1):
    dp[i][0] = i

# 技巧2:enumerate简化索引计算
for i, c1 in enumerate(word1, 1):  # 从1开始计数
    for j, c2 in enumerate(word2, 1):
        # 此时i,j直接对应dp表的行列,c1,c2是字符

# 技巧3:滚动数组保存左上角值
prev = dp[0]  # 保存左上角
for j in range(1, n+1):
    temp = dp[j]  # 保存更新前的值
    dp[j] = ...
    prev = temp  # prev更新为原来的dp[j]
```

### 💡 底层原理(选读)

> **为什么编辑距离是对称的?**
>
> word1转换为word2的操作,可以"反向"理解:
> - word1删除一个字符 ≈ word2插入一个字符
> - word1插入一个字符 ≈ word2删除一个字符
> - 替换是对称的
>
> 因此minDistance(word1, word2) == minDistance(word2, word1)
>
> **DP状态转移的本质?**
>
> dp[i][j]表示"两个前缀字符串的编辑距离"。状态转移时:
> - 如果末尾字符相同,问题规模缩小为"去掉末尾字符后的前缀"
> - 如果不同,尝试三种操作,选最少的那个
>
> 这种"从子问题推导大问题"的思路是DP的精髓。

### 算法模式卡片 📐
- **模式名称**:双字符串DP(Two-String DP)
- **适用条件**:
  1. 涉及两个字符串的匹配/转换/距离问题
  2. 当前状态只依赖于两个字符串的前缀状态
  3. 求最优解(最小/最大/计数)
- **识别关键词**:"编辑距离"、"最长公共子序列"、"匹配"、"转换"
- **模板代码**:
```python
# 双字符串DP标准模板
def twoStringDP(s1: str, s2: str) -> int:
    m, n = len(s1), len(s2)
    dp = [[0] * (n+1) for _ in range(m+1)]

    # 初始化第一行和第一列
    for i in range(m+1):
        dp[i][0] = 初始值(i)
    for j in range(n+1):
        dp[0][j] = 初始值(j)

    # 填充DP表
    for i in range(1, m+1):
        for j in range(1, n+1):
            if s1[i-1] == s2[j-1]:
                dp[i][j] = dp[i-1][j-1] + 匹配时的值
            else:
                dp[i][j] = min/max(
                    dp[i-1][j] + 操作1,
                    dp[i][j-1] + 操作2,
                    dp[i-1][j-1] + 操作3
                )

    return dp[m][n]
```

### 易错点 ⚠️
1. **索引越界**
   - 错误:dp[i][j]对应word1[i]和word2[j]
   - 解释:dp[i][j]表示前i个字符,对应word1[0:i],所以应该是word1[i-1]
   - 正确做法:始终记住dp的索引比字符串索引大1

2. **边界初始化错误**
   - 错误:忘记初始化dp[0][j]或dp[i][0]
   - 解释:dp[0][j]表示空串转换为word2前j个字符,需要插入j次
   - 正确做法:第一行初始化为[0,1,2,...,n],第一列初始化为[0,1,2,...,m]

3. **三种操作理解混淆**
   - 错误:不清楚dp[i-1][j]、dp[i][j-1]、dp[i-1][j-1]分别对应什么操作
   - 正确理解:
     - dp[i-1][j]:word1前i-1个字符已匹配word2前j个字符,需要删除word1[i-1]
     - dp[i][j-1]:word1前i个字符已匹配word2前j-1个字符,需要插入word2[j-1]
     - dp[i-1][j-1]:word1前i-1个字符已匹配word2前j-1个字符,需要替换word1[i-1]为word2[j-1]

---

## 🏗️ 工程实战(选读)

> 这个算法思想在真实项目中的应用,让你知道"学了有什么用"。

- **场景1:拼写检查与纠错**
  - Word、Google文档等拼写检查器,通过计算输入词与词典中词的编辑距离,推荐相似词
  - 编辑距离小于阈值(如2)的词作为候选纠正

- **场景2:DNA序列比对**
  - 生物信息学中比较DNA/蛋白质序列的相似性
  - 编辑距离(Levenshtein距离)是序列比对的基础算法

- **场景3:版本控制系统diff**
  - Git等版本控制系统计算文件差异时,本质是计算两个文本的编辑距离
  - 优化后的算法可以输出具体的修改操作序列

- **场景4:语音识别与自然语言处理**
  - 计算两个词的"发音相似度",用编辑距离衡量
  - 用于语音输入的容错和智能纠正

---

## 🏋️ 举一反三

完成本课后,试试这些同类题目来巩固知识:

| 题目 | 难度 | 相关知识点 | 提示 |
|------|------|-----------|------|
| LeetCode 1143. 最长公共子序列 | Medium | 双字符串DP | 状态转移类似,但是求最长而非最短 |
| LeetCode 583. 两个字符串的删除操作 | Medium | 双字符串DP | 只能删除,不能插入和替换 |
| LeetCode 712. 两个字符串的最小ASCII删除和 | Medium | 双字符串DP+权重 | 删除操作有权重(字符ASCII值) |
| LeetCode 115. 不同的子序列 | Hard | 双字符串DP计数 | 计数而非求最值 |
| LeetCode 161. 相隔为1的编辑距离 | Medium | 双字符串比较 | 判断编辑距离是否恰好为1 |

---

## 📝 课后小测

试试这道变体题,不要看答案,自己先想5分钟!

**题目**:如果每种操作有不同的成本:插入成本2,删除成本3,替换成本1。求最小总成本。

输入:word1 = "abc", word2 = "yabd"
解释:替换'a'为'y'(成本1),插入'd'(成本2),总成本3

<details>
<summary>💡 提示(实在想不出来再点开)</summary>

修改状态转移方程,将+1改为+对应操作的成本

</details>

<details>
<summary>✅ 参考答案</summary>

```python
def minCostDistance(word1: str, word2: str) -> int:
    """
    带权重的编辑距离
    插入成本2,删除成本3,替换成本1
    """
    m, n = len(word1), len(word2)
    dp = [[0] * (n+1) for _ in range(m+1)]

    # 初始化边界
    for j in range(n+1):
        dp[0][j] = j * 2  # 插入成本2
    for i in range(m+1):
        dp[i][0] = i * 3  # 删除成本3

    # 填充DP表
    for i in range(1, m+1):
        for j in range(1, n+1):
            if word1[i-1] == word2[j-1]:
                dp[i][j] = dp[i-1][j-1]
            else:
                dp[i][j] = min(
                    dp[i-1][j] + 3,    # 删除成本3
                    dp[i][j-1] + 2,    # 插入成本2
                    dp[i-1][j-1] + 1   # 替换成本1
                )

    return dp[m][n]

# 测试
print(minCostDistance("abc", "yabd"))
# abc -> yabc (替换a为y,成本1)
# yabc -> yabd (替换c为d,成本1,或删除c+插入d,成本5)
# 最小成本:1+1=2
```

核心思路:在标准编辑距离的基础上,将每种操作的+1改为+对应成本即可。

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
