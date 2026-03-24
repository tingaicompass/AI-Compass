> 想系统提升编程能力、查看更完整的学习路线，欢迎访问 AI Compass：https://github.com/tingaicompass/AI-Compass
> 仓库持续更新刷题题解、Python 基础和 AI 实战内容，适合想高效进阶的你。

# 📖 第84课:正则表达式匹配

> **模块**:动态规划 | **难度**:Hard ⭐⭐
> **LeetCode 链接**:https://leetcode.cn/problems/regular-expression-matching/
> **前置知识**:第83课(最长公共子序列)
> **预计学习时间**:40分钟

---

## 🎯 题目描述

给定一个字符串 s 和一个字符模式 p,实现支持 '.' 和 '*' 的正则表达式匹配。

- '.' 匹配任意单个字符
- '*' 匹配零个或多个前面的那一个元素

匹配应该覆盖整个字符串 s,而不是部分字符串。

**示例:**
```
输入:s = "aa", p = "a"
输出:false
解释:"a" 无法匹配 "aa" 整个字符串
```

```
输入:s = "aa", p = "a*"
输出:true
解释:'*' 表示零个或多个 'a',可以匹配 "aa"
```

```
输入:s = "ab", p = ".*"
输出:true
解释:'.' 匹配任意字符,'*' 表示可以重复,'.*' 可以匹配任意字符串
```

```
输入:s = "aab", p = "c*a*b"
输出:true
解释:c* 匹配0个c,a* 匹配2个a,b 匹配1个b
```

**约束条件:**
- 1 <= s.length <= 20
- 1 <= p.length <= 20
- s 只包含小写英文字母
- p 只包含小写英文字母,以及字符 . 和 *
- 保证每次出现字符 * 时,前面都有一个有效字符

---

### 🧪 边界用例(面试必考)

| 用例类型 | s | p | 期望输出 | 考察点 |
|---------|---|---|---------|--------|
| 完全匹配 | "abc" | "abc" | true | 基本功能 |
| 星号匹配0个 | "ab" | "c*ab" | true | * 匹配零个 |
| 星号匹配多个 | "aaa" | "a*" | true | * 匹配多个 |
| 点号匹配 | "ab" | "." | true | . 匹配单字符 |
| 点星组合 | "aab" | ".*" | true | .* 万能匹配 |
| 空字符串 | "" | "a*" | true | 空串匹配 |
| 无法匹配 | "aa" | "a" | false | 长度不匹配 |

---

## 💡 思路引导

### 生活化比喻
> 想象你在玩一个"密码匹配"游戏,输入密码(字符串s)需要和门锁的规则(模式p)匹配才能打开。
>
> 门锁的规则很特殊:
> - `?` 牌(对应 `.`):可以代表任意一个字符,就像通配符
> - `*` 牌(对应 `*`):可以让前面那张牌重复0次、1次或多次
>
> 🐌 **笨办法**:对于每个 `*`,你尝试让它匹配0个、1个、2个...字符,然后递归检查剩余部分。这样会产生大量重复的子问题。
>
> 🚀 **聪明办法**:用一个"决策表格"记录"密码前i位是否匹配规则前j位",每次只需要看左上、上、左三个格子的结果,就能快速判断当前格子。

### 关键洞察
**'*' 的关键是"可以匹配0个或多个",需要分情况讨论:匹配0个(跳过 pattern),或匹配至少1个(消耗 s 中的一个字符后继续匹配)。**

---

## 🧠 解题思维链

> 这一节模拟你在面试中"从零开始思考"的过程。

### Step 1:理解题目 → 锁定输入输出
- **输入**:字符串 s (长度 m) 和模式 p (长度 n)
- **输出**:布尔值,表示 s 是否完全匹配 p
- **限制**:
  - `.` 可以匹配任意单个字符
  - `*` 必须和前面的字符配合,表示"前面字符出现0次或多次"
  - 必须完全匹配,不能只匹配部分

### Step 2:先想笨办法(暴力递归)
从左到右逐字符匹配:
- 如果当前字符匹配(或 p[j] 是 `.`),继续匹配下一个
- 如果 p[j+1] 是 `*`,分两种情况:
  1. `*` 匹配0个:跳过 p[j] 和 `*`,继续匹配
  2. `*` 匹配至少1个:消耗 s[i],p 不动,继续匹配

时间复杂度:O(2^(m+n)),每个 `*` 都会产生分支

### Step 3:瓶颈分析 → 优化方向
- 核心问题:同一个子问题(s[i:] 是否匹配 p[j:])会被重复计算
- 优化思路:用 DP 表记录所有子问题的答案

### Step 4:选择武器
- 选用:**二维动态规划**
- 理由:
  1. 状态定义清晰:dp[i][j] = s[0:i] 是否匹配 p[0:j]
  2. 状态转移复杂但可枚举:根据 p[j-1] 是否为 `*` 分情况
  3. 二维DP表可以记录所有子问题

> 🔑 **模式识别提示**:当题目涉及"字符串匹配 + 复杂规则(通配符、正则)"时,优先考虑"二维DP"

---

## 🔑 解法一:递归暴力搜索(理解思路)

### 思路
直接用递归模拟匹配过程,分情况讨论:
1. 如果 p 的下一个字符不是 `*`:当前字符必须匹配,递归检查剩余部分
2. 如果 p 的下一个字符是 `*`:分两种情况递归

### 图解过程

```
示例:s="aab", p="c*a*b"

递归树(简化):
      match("aab", "c*a*b")
         /                \
    (c* 匹配0个)        (c* 匹配1个c,但 s 没有 c,失败)
   match("aab", "a*b")
       /          \
  (a* 匹配0个)   (a* 匹配1个a)
 match("aab", "b")  match("ab", "a*b")
      失败             /        \
                 (a* 匹配1个a) (a* 匹配2个a,失败)
                match("b", "b") → 成功!
```

### Python代码

```python
def is_match_recursive(s: str, p: str) -> bool:
    """
    解法一:递归暴力搜索
    思路:直接模拟匹配过程,会超时
    """
    # 基础情况:模式为空
    if not p:
        return not s

    # 第一个字符是否匹配(考虑 '.')
    first_match = bool(s) and (p[0] == s[0] or p[0] == '.')

    # 如果下一个字符是 '*'
    if len(p) >= 2 and p[1] == '*':
        # 情况1:'*' 匹配0个,跳过 p[0] 和 '*'
        # 情况2:'*' 匹配至少1个,消耗 s[0]
        return (is_match_recursive(s, p[2:]) or
                (first_match and is_match_recursive(s[1:], p)))
    else:
        # 没有 '*',当前字符必须匹配,继续下一个
        return first_match and is_match_recursive(s[1:], p[1:])


# ✅ 测试
print(is_match_recursive("aa", "a"))  # 期望输出:False
print(is_match_recursive("aa", "a*"))  # 期望输出:True
print(is_match_recursive("ab", ".*"))  # 期望输出:True
```

### 复杂度分析
- **时间复杂度**:O(2^(m+n)) — 每个 `*` 都会产生分支,指数级
- **空间复杂度**:O(m+n) — 递归栈深度

### 优缺点
- ✅ 思路清晰,直接模拟匹配过程
- ❌ 存在大量重复子问题,无法通过大规模测试

---

## ⚡ 解法二:记忆化递归(剪枝优化)

### 优化思路
在递归基础上加入备忘录,缓存已计算的子问题。

> 💡 **关键想法**:用字典 memo[(i, j)] 记录 s[i:] 是否匹配 p[j:]

### Python代码

```python
def is_match_memo(s: str, p: str) -> bool:
    """
    解法二:记忆化递归
    思路:用字典缓存子问题结果
    """
    memo = {}

    def dp(i: int, j: int) -> bool:
        # 查备忘录
        if (i, j) in memo:
            return memo[(i, j)]

        # 基础情况:模式用完
        if j == len(p):
            result = (i == len(s))
        else:
            # 第一个字符是否匹配
            first_match = (i < len(s)) and (p[j] == s[i] or p[j] == '.')

            # 下一个字符是否为 '*'
            if j + 1 < len(p) and p[j + 1] == '*':
                # 情况1:'*' 匹配0个
                # 情况2:'*' 匹配至少1个
                result = (dp(i, j + 2) or
                         (first_match and dp(i + 1, j)))
            else:
                # 没有 '*',必须匹配
                result = first_match and dp(i + 1, j + 1)

        # 存入备忘录
        memo[(i, j)] = result
        return result

    return dp(0, 0)


# ✅ 测试
print(is_match_memo("aab", "c*a*b"))  # 期望输出:True
print(is_match_memo("aa", "a"))  # 期望输出:False
print(is_match_memo("aa", "a*"))  # 期望输出:True
```

### 复杂度分析
- **时间复杂度**:O(m*n) — 每个 (i, j) 子问题只计算一次
- **空间复杂度**:O(m*n) — 备忘录 + 递归栈

---

## 🏆 解法三:二维DP表(最优解)

### 优化思路
将记忆化递归改为自底向上的DP表填充,消除递归调用。

> 💡 **关键想法**:dp[i][j] 表示 s[0:i] 是否匹配 p[0:j]

### 图解过程

```
示例:s="aab", p="c*a*b"

构建 DP 表(行=s,列=p):

       ""  c  *  a  *  b
""     T   F  T  F  T  F
a      F   F  F  T  T  F
a      F   F  F  F  T  F
b      F   F  F  F  F  T

关键状态转移:
1. dp[0][0] = True (空串匹配空模式)
2. dp[0][2] = True (空串可以匹配 "c*",因为 * 可以匹配0个 c)
3. dp[1][3] = True (s[0]='a' 匹配 p[2]='a')
4. dp[3][5] = True (最终结果)

状态转移方程:
- 如果 p[j-1] != '*':
    dp[i][j] = dp[i-1][j-1] && (s[i-1] == p[j-1] || p[j-1] == '.')
- 如果 p[j-1] == '*':
    情况1:* 匹配0个,dp[i][j] = dp[i][j-2]
    情况2:* 匹配至少1个,dp[i][j] = dp[i-1][j] && (s[i-1] == p[j-2] || p[j-2] == '.')
```

### Python代码

```python
def is_match(s: str, p: str) -> bool:
    """
    解法三:二维DP表(最优解)
    思路:自底向上填充 DP 表
    """
    m, n = len(s), len(p)
    # 初始化 DP 表
    dp = [[False] * (n + 1) for _ in range(m + 1)]
    dp[0][0] = True  # 空串匹配空模式

    # 初始化第0行:空串 s 匹配模式 p
    # 只有 "a*b*c*" 这种形式可以匹配空串
    for j in range(2, n + 1):
        if p[j - 1] == '*':
            dp[0][j] = dp[0][j - 2]

    # 填充 DP 表
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if p[j - 1] == '*':
                # 情况1:'*' 匹配0个前面的字符
                dp[i][j] = dp[i][j - 2]
                # 情况2:'*' 匹配至少1个前面的字符
                if s[i - 1] == p[j - 2] or p[j - 2] == '.':
                    dp[i][j] = dp[i][j] or dp[i - 1][j]
            else:
                # 当前字符必须匹配(或 p[j-1] 是 '.')
                if s[i - 1] == p[j - 1] or p[j - 1] == '.':
                    dp[i][j] = dp[i - 1][j - 1]

    return dp[m][n]


# ✅ 测试
print(is_match("aa", "a"))  # 期望输出:False
print(is_match("aa", "a*"))  # 期望输出:True
print(is_match("ab", ".*"))  # 期望输出:True
print(is_match("aab", "c*a*b"))  # 期望输出:True
print(is_match("", "a*"))  # 期望输出:True
print(is_match("", ".*"))  # 期望输出:True
```

### 复杂度分析
- **时间复杂度**:O(m*n) — 双重循环填充表格,每个格子计算一次
  - 具体地说:如果 m=20, n=20,大约需要 400 次操作
- **空间复杂度**:O(m*n) — DP 表空间

### 为什么是最优解
- 时间已达理论最优:必须检查所有字符组合
- 代码逻辑清晰,状态转移方程明确
- 面试中容易在白板上演示和调试

---

## 🐍 Pythonic 写法

利用 Python 的 `@lru_cache` 装饰器:

```python
from functools import lru_cache

def is_match_lru(s: str, p: str) -> bool:
    """
    Pythonic 写法:用 @lru_cache 自动缓存
    """
    @lru_cache(maxsize=None)
    def dp(i: int, j: int) -> bool:
        if j == len(p):
            return i == len(s)

        first_match = (i < len(s)) and (p[j] == s[i] or p[j] == '.')

        if j + 1 < len(p) and p[j + 1] == '*':
            return (dp(i, j + 2) or
                   (first_match and dp(i + 1, j)))
        return first_match and dp(i + 1, j + 1)

    return dp(0, 0)

# ✅ 测试
print(is_match_lru("aab", "c*a*b"))  # 期望输出:True
```

> ⚠️ **面试建议**:先写清晰的二维DP版本(解法三),再提 Pythonic 写法展示语言功底。
> 面试官更看重你的**状态转移设计思路**,而非代码行数。

---

## 📊 解法对比

| 维度 | 解法一:暴力递归 | 解法二:记忆化递归 | 🏆 解法三:二维DP表(最优) |
|------|--------------|----------------|---------------------|
| 时间复杂度 | O(2^(m+n)) | O(m*n) | **O(m*n)** ← 时间最优 |
| 空间复杂度 | O(m+n) | O(m*n) | O(m*n) |
| 代码难度 | 简单 | 中等 | 中等 |
| 面试推荐 | ⭐ | ⭐⭐ | **⭐⭐⭐** ← 首选 |
| 适用场景 | 理解递归定义 | 辅助理解DP | **面试首选,清晰易懂** |

**为什么解法三是最优解**:
- 时间复杂度 O(m*n) 已经是理论最优(必须检查所有字符对)
- DP表可视化强,便于在白板上演示状态转移
- 初始化逻辑清晰(第0行处理空串匹配模式的情况)

**面试建议**:
1. 先用1分钟口述递归思路(解法一),强调"遇到 * 要分两种情况"
2. 立即优化到🏆最优解(解法三:二维DP表),展示DP设计能力
3. **重点讲解状态转移方程**:
   - 如果 p[j-1] 不是 `*`,就看当前字符是否匹配
   - 如果 p[j-1] 是 `*`,分"匹配0个"和"匹配至少1个"两种情况
4. 手动画一个 3x4 的 DP 表,演示如何填充
5. 强调初始化第0行的特殊逻辑:"a*b*" 可以匹配空串

---

## 🎤 面试现场

> 模拟面试中的完整对话流程,帮你练习"边想边说"。

**面试官**:请你实现一个正则表达式匹配,支持 '.' 和 '*'。

**你**:(审题30秒)好的,这道题要实现简化版的正则匹配。'.' 可以匹配任意单字符,'*' 可以让前面的字符重复0次或多次。

我的第一个想法是递归模拟:如果 p 的下一个字符是 `*`,就分两种情况——`*` 匹配0个(跳过),或匹配至少1个(消耗 s 的一个字符)。但这样会有大量重复计算。

所以我会用二维DP优化。定义 dp[i][j] 表示 s 前 i 个字符是否匹配 p 前 j 个字符。状态转移分两种情况:
1. 如果 p[j-1] 不是 `*`:当前字符必须匹配,dp[i][j] = dp[i-1][j-1] && (s[i-1] == p[j-1] || p[j-1] == '.')
2. 如果 p[j-1] 是 `*`:
   - `*` 匹配0个:dp[i][j] = dp[i][j-2]
   - `*` 匹配至少1个:dp[i][j] = dp[i-1][j] (前提是 s[i-1] 匹配 p[j-2])

时间复杂度 O(m*n),空间复杂度 O(m*n)。

**面试官**:很好,请写一下代码。

**你**:(边写边说)
```python
def isMatch(s: str, p: str) -> bool:
    m, n = len(s), len(p)
    dp = [[False] * (n + 1) for _ in range(m + 1)]
    dp[0][0] = True  # 空串匹配空模式

    # 初始化第0行:处理 "a*b*" 匹配空串的情况
    for j in range(2, n + 1):
        if p[j-1] == '*':
            dp[0][j] = dp[0][j-2]

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if p[j-1] == '*':
                # '*' 匹配0个
                dp[i][j] = dp[i][j-2]
                # '*' 匹配至少1个
                if s[i-1] == p[j-2] or p[j-2] == '.':
                    dp[i][j] = dp[i][j] or dp[i-1][j]
            else:
                # 当前字符必须匹配
                if s[i-1] == p[j-1] or p[j-1] == '.':
                    dp[i][j] = dp[i-1][j-1]

    return dp[m][n]
```

**面试官**:测试一下?

**你**:用示例 s="aab", p="c*a*b" 走一遍。我在纸上画个 4x6 的表格...(手动模拟)
- dp[0][0] = True
- dp[0][2] = True (c* 匹配空串)
- dp[1][3] = True (第一个 'a' 匹配)
- 最终 dp[3][5] = True

再测一个边界:s="", p="a*" → dp[0][2]=True,因为 a* 可以匹配0个 a。结果正确。

### 高频追问

| 追问 | 应答策略 |
|------|---------|
| "如何优化空间?" | 可以用滚动数组降到 O(n),但逻辑会复杂一些。通常面试不要求这个优化。 |
| "如果还要支持 '+' 呢?" | '+' 表示前面字符至少出现1次,状态转移类似 `*`,但"匹配0个"的分支去掉。 |
| "时间能否更优?" | 不能,O(m*n) 已经是理论下限,必须检查所有字符组合。 |
| "实际中用什么库?" | Python 用 `re` 模块,底层是 NFA(非确定有限自动机)实现,性能更好。 |

---

## 🎓 知识点总结

### Python技巧卡片 🐍
```python
# 技巧1:二维DP初始化第0行/列
dp[0][0] = True  # 空串匹配空模式
for j in range(2, n + 1, 2):  # 只考虑偶数位置(因为 * 必须跟在字符后面)
    if p[j-1] == '*':
        dp[0][j] = dp[0][j-2]

# 技巧2:处理 '*' 的两种情况用 or 连接
dp[i][j] = dp[i][j-2] or (first_match and dp[i-1][j])

# 技巧3:字符匹配的简洁写法
first_match = (s[i-1] == p[j-1] or p[j-1] == '.')
```

### 💡 底层原理(选读)

> **为什么正则匹配是 Hard 题?**
>
> 1. **状态复杂**:'*' 可以匹配0~无穷个字符,状态空间巨大
> 2. **非局部性**:'*' 的效果取决于后续匹配结果,不能贪心
> 3. **边界繁多**:空串、纯 '*' 模式、嵌套 '*' 等情况需要特殊处理
>
> **实际正则引擎如何实现?**
> - 实际的正则引擎(如 Python 的 `re` 模块)使用 NFA(非确定有限自动机)
> - NFA 可以并行探索多条匹配路径,避免回溯
> - 本题的 DP 解法本质上是模拟了 NFA 的状态转移

### 算法模式卡片 📐
- **模式名称**:二维DP(复杂状态转移)
- **适用条件**:字符串匹配 + 通配符/正则规则
- **识别关键词**:通配符匹配、正则表达式、模式匹配
- **模板代码**:
```python
def complex_match(s: str, p: str) -> bool:
    m, n = len(s), len(p)
    dp = [[False] * (n + 1) for _ in range(m + 1)]
    dp[0][0] = True

    # 初始化第0行(根据具体规则)
    for j in range(1, n + 1):
        if 特殊规则(p[j-1]):
            dp[0][j] = ...

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if 特殊字符(p[j-1]):
                # 分多种情况讨论
                dp[i][j] = 情况1 or 情况2 or ...
            else:
                # 普通匹配
                dp[i][j] = (s[i-1] == p[j-1]) and dp[i-1][j-1]

    return dp[m][n]
```

### 易错点 ⚠️
1. **初始化第0行遗漏**:忘记处理 "a*b*" 匹配空串的情况
   - **为什么错**:p="a*" 可以匹配 s="",但如果不初始化 dp[0][2]=True,会误判为 False
   - **正确做法**:遍历 p,对于每个 `*`,设置 dp[0][j] = dp[0][j-2]

2. **'*' 的状态转移搞混**:把"匹配0个"和"匹配至少1个"的逻辑写反
   - **为什么错**:"匹配0个"应该跳过 p[j-2] 和 p[j-1](即 `*`),即 dp[i][j-2]
   - **正确做法**:画图理解,"匹配0个"往左跳2格,"匹配至少1个"往上跳1格

3. **索引越界**:访问 p[j+1] 之前未检查 j+1 < len(p)
   - **为什么错**:在递归版本中容易忘记边界检查
   - **正确做法**:访问 p[j+1] 前先判断 `if j+1 < len(p) and p[j+1] == '*'`

---

## 🏗️ 工程实战(选读)

> 这个算法思想在真实项目中的应用,让你知道"学了有什么用"。

- **场景1**:文本编辑器的"查找"功能 — 支持 `*` 和 `?` 通配符搜索文件
- **场景2**:Shell 命令行的文件名匹配 — `ls *.txt` 中的 `*` 就是通配符
- **场景3**:URL 路由匹配 — Web 框架中 `/user/:id/*` 路径匹配
- **场景4**:日志分析工具 — 用正则表达式过滤日志行

---

## 🏋️ 举一反三

完成本课后,试试这些同类题目来巩固知识:

| 题目 | 难度 | 相关知识点 | 提示 |
|------|------|-----------|------|
| LeetCode 44. 通配符匹配 | Hard | 二维DP、通配符 | 类似本题,但 `*` 可以匹配任意序列(不需要前面的字符) |
| LeetCode 72. 编辑距离 | Medium | 二维DP、字符串匹配 | 同样是双字符串DP,但状态转移是插入/删除/替换 |
| LeetCode 115. 不同的子序列 | Hard | 二维DP | 计数问题,dp[i][j] 表示方案数而非布尔值 |
| LeetCode 97. 交错字符串 | Medium | 二维DP | 判断 s3 是否由 s1 和 s2 交错组成 |

---

## 📝 课后小测

试试这道变体题,不要看答案,自己先想5分钟!

**题目**:如果正则表达式增加 `+` 符号,表示前面字符至少出现1次,如何修改代码?

<details>
<summary>💡 提示(实在想不出来再点开)</summary>

`+` 的状态转移类似 `*`,但去掉"匹配0个"的分支。即:
- `*`:dp[i][j] = dp[i][j-2] (匹配0个) or dp[i-1][j] (匹配至少1个)
- `+`:dp[i][j] = dp[i-1][j] (必须匹配至少1个)

</details>

<details>
<summary>✅ 参考答案</summary>

```python
def is_match_with_plus(s: str, p: str) -> bool:
    """
    扩展版:支持 '.' '*' 和 '+'
    '+' 表示前面字符至少出现1次
    """
    m, n = len(s), len(p)
    dp = [[False] * (n + 1) for _ in range(m + 1)]
    dp[0][0] = True

    # 初始化第0行('+' 无法匹配空串,只有 '*' 可以)
    for j in range(2, n + 1):
        if p[j-1] == '*':
            dp[0][j] = dp[0][j-2]

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if p[j-1] == '*':
                # '*' 匹配0个或多个
                dp[i][j] = dp[i][j-2]
                if s[i-1] == p[j-2] or p[j-2] == '.':
                    dp[i][j] = dp[i][j] or dp[i-1][j]
            elif p[j-1] == '+':
                # '+' 必须匹配至少1个
                if s[i-1] == p[j-2] or p[j-2] == '.':
                    dp[i][j] = dp[i-1][j] or dp[i-1][j-2]
            else:
                # 普通字符或 '.'
                if s[i-1] == p[j-1] or p[j-1] == '.':
                    dp[i][j] = dp[i-1][j-1]

    return dp[m][n]

# 测试
print(is_match_with_plus("a", "a+"))  # 输出:True
print(is_match_with_plus("", "a+"))  # 输出:False ('+' 至少1个)
print(is_match_with_plus("aa", "a+"))  # 输出:True
```

**核心思路**:`+` 与 `*` 的唯一区别是不能匹配0个,所以去掉 dp[i][j-2] 分支,只保留"匹配至少1个"的逻辑。

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
