> 想系统提升编程能力、查看更完整的学习路线，欢迎访问 AI Compass：https://github.com/tingaicompass/AI-Compass
> 仓库持续更新刷题题解、Python 基础和 AI 实战内容，适合想高效进阶的你。

# 📖 第18课：最长回文子串

> **模块**：字符串 | **难度**：Medium ⭐⭐⭐
> **LeetCode 链接**：https://leetcode.cn/problems/longest-palindromic-substring/
> **前置知识**：第14课（无重复字符的最长子串）
> **预计学习时间**：30分钟

---

## 🎯 题目描述

给定一个字符串 `s`，请你找出其中最长的回文子串。

**回文串**：正着读和倒着读都一样的字符串，比如 `"aba"`、`"racecar"`。

**示例1：**
```
输入：s = "babad"
输出："bab"
解释："aba" 也是有效答案
```

**示例2：**
```
输入：s = "cbbd"
输出："bb"
```

**约束条件：**
- `1 <= s.length <= 1000`
- `s` 仅由数字和英文字母组成

---

### 🧪 边界用例（面试必考）

| 用例类型 | 输入 | 期望输出 | 考察点 |
|---------|------|---------|--------|
| 最小输入 | `s = "a"` | `"a"` | 单字符必定回文 |
| 全相同字符 | `s = "aaaa"` | `"aaaa"` | 整个串都是回文 |
| 无回文（长度>1） | `s = "abc"` | `"a"` 或 `"b"` 或 `"c"` | 单字符回文 |
| 偶数长度回文 | `s = "abba"` | `"abba"` | 中心是两个字符 |
| 奇数长度回文 | `s = "aba"` | `"aba"` | 中心是单个字符 |
| 大规模边界 | `s.length = 1000` | — | 性能测试 |

---

## 💡 思路引导

### 生活化比喻
> 想象你在读一本侦探小说，里面藏着一句最长的"回文暗语"...
>
> 🐌 **笨办法**：把书里每个单词、每句话、每段文字都拿出来，从头读一遍，再从尾读一遍，看是不是一样。如果书有1000页，你得检查成千上万种组合，累死人！
>
> 🚀 **聪明办法**：回文有个特点——从中心向两边看，左右是镜像对称的！就像"上海自来水来自海上"这句话，你只需要站在中心（"来"字），然后同时往左右看，一旦发现不对称就停下。这样你只需要在每个可能的"中心点"试一次，效率高得多！

### 关键洞察
**回文串的核心特征是"中心对称"，从中心向两边扩展检查，比枚举所有子串高效得多。**

---

## 🧠 解题思维链

> 这一节模拟你在面试中"从零开始思考"的过程。

### Step 1：理解题目 → 锁定输入输出
- **输入**：一个字符串 `s`，长度 1 ~ 1000
- **输出**：`s` 中最长的回文子串（字符串类型，不是长度）
- **限制**：如果有多个答案，返回任意一个即可

### Step 2：先想笨办法（暴力法）
最直接的想法：枚举所有可能的子串，逐个检查是否为回文。
- 枚举所有子串：两层循环确定起点和终点，共 O(n²) 种子串
- 检查每个子串是否回文：需要 O(n) 时间（逐字符对比）
- **总时间复杂度**：O(n³)
- **瓶颈在哪**：每次检查回文都要重新遍历整个子串，大量重复工作

### Step 3：瓶颈分析 → 优化方向
暴力法的核心问题：**为每个子串都从头到尾检查一遍回文，没有利用回文的结构特性**。

回文的关键特征是什么？**中心对称！**
- 核心问题：能不能利用"对称性"来减少重复计算？
- 优化思路：与其检查所有子串，不如**从每个可能的中心出发，向两边扩展**

### Step 4：选择武器
- 选用：**中心扩展法（Expand Around Center）**
- 理由：
  - 回文串的中心可能是单个字符（奇数长度，如 `"aba"`）
  - 也可能是两个字符之间（偶数长度，如 `"abba"`）
  - 对每个中心，向两边扩展只需 O(n)，总共 O(n²)

> 🔑 **模式识别提示**：当题目出现"回文"、"对称"、"镜像"关键词，优先考虑"中心扩展法"或"动态规划"

---

## 🔑 解法一：暴力枚举（朴素思路）

### 思路
枚举所有可能的子串（起点i，终点j），对每个子串检查是否为回文。保留最长的那个。

### 图解过程

```
示例：s = "babad"

第1步：枚举所有子串
i=0, j=0: "b" → 回文 ✅ (长度1)
i=0, j=1: "ba" → 非回文 ❌
i=0, j=2: "bab" → 回文 ✅ (长度3) ← 目前最长
i=0, j=3: "baba" → 非回文 ❌
i=0, j=4: "babad" → 非回文 ❌
i=1, j=1: "a" → 回文 ✅ (长度1)
i=1, j=2: "ab" → 非回文 ❌
i=1, j=3: "aba" → 回文 ✅ (长度3)
i=1, j=4: "abad" → 非回文 ❌
...
最终答案："bab" 或 "aba" (长度3)

第2步：检查 "bab" 是否回文
  b a b
  ↑   ↑  相同 ✅
    ↑    中心字符 ✅
  → 是回文
```

边界情况演示：`s = "cbbd"`
```
枚举过程：
i=0: "c", "cb", "cbb", "cbbd" → 最长 "c"
i=1: "b", "bb", "bbd" → 最长 "bb" ← 答案
i=2: "b", "bd" → 最长 "b"
i=3: "d" → "d"
最终答案："bb" (长度2)
```

### Python代码

```python
def longest_palindrome_brute(s: str) -> str:
    """
    解法一：暴力枚举
    思路：枚举所有子串，逐个检查是否回文
    """
    def is_palindrome(sub: str) -> bool:
        """辅助函数：检查字符串是否为回文"""
        return sub == sub[::-1]

    n = len(s)
    max_len = 0
    result = ""

    # 枚举所有可能的子串
    for i in range(n):
        for j in range(i, n):
            substring = s[i:j+1]  # 子串 s[i...j]
            if is_palindrome(substring):
                if len(substring) > max_len:
                    max_len = len(substring)
                    result = substring

    return result


# ✅ 测试
print(longest_palindrome_brute("babad"))  # 期望输出："bab" 或 "aba"
print(longest_palindrome_brute("cbbd"))   # 期望输出："bb"
print(longest_palindrome_brute("a"))      # 期望输出："a"
print(longest_palindrome_brute("ac"))     # 期望输出："a" 或 "c"
```

### 复杂度分析
- **时间复杂度**：O(n³) — 两层循环枚举子串 O(n²)，检查回文 O(n)，共 O(n³)
  - 具体地说：如果输入规模 n=1000，大约需要 10^9 次操作，在LeetCode上会**超时 ⏱️**
- **空间复杂度**：O(1) — 只用了几个变量

### 优缺点
- ✅ **优点**：思路直观，容易理解和实现
- ❌ **缺点**：时间复杂度过高，n>100 就会明显变慢，无法通过大数据测试用例

---

## ⚡ 解法二：中心扩展法（推荐⭐⭐⭐）

### 优化思路
暴力法的问题是"先取子串，再检查回文"，做了大量无用功。

回文的本质是**中心对称**，我们可以反过来思考：
- 从每个可能的"中心"出发
- 向两边同时扩展，只要左右字符相同就继续
- 一旦不同就停止，记录这个回文的长度

> 💡 **关键想法**：回文的中心有两种情况：
> - **奇数长度回文**：中心是单个字符（如 `"aba"` 的中心是 `'b'`）
> - **偶数长度回文**：中心是两个字符之间（如 `"abba"` 的中心在两个 `'b'` 之间）

### 图解过程

```
示例：s = "babad"

中心扩展的所有可能中心点：
  b   a   b   a   d
  ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑
  0 0 1 1 2 2 3 3 4  (共9个中心：5个奇数中心 + 4个偶数中心)

【奇数长度回文】中心 = 单个字符
中心在 i=0 ('b'):
  b a b a d
  ↑         左右不同，停止 → 长度1

中心在 i=1 ('a'):
  b a b a d
    ↑ ↑ ↑   左右相同，继续扩展
  ← · · · → 左右不同，停止 → 长度3 ("bab")

中心在 i=2 ('b'):
  b a b a d
      ↑     左右相同
  · · ↑ · · 再扩展，左右不同，停止 → 长度3 ("aba")

【偶数长度回文】中心 = 两个字符之间
中心在 (0,1) 之间:
  b a b a d
  ↑ ↑       b != a，停止 → 长度0

中心在 (1,2) 之间:
  b a b a d
    ↑ ↑     a != b，停止 → 长度0

最大长度 = 3，对应 "bab" 或 "aba"
```

偶数回文示例：`s = "cbbd"`
```
中心在 (1,2) 之间 ('bb'):
  c b b d
    ↑ ↑   b == b，扩展成功
  ← · · → c != d，停止 → 长度2 ("bb")
```

### Python代码

```python
def longest_palindrome_expand(s: str) -> str:
    """
    解法二：中心扩展法
    思路：以每个字符/每对字符为中心，向两边扩展
    """
    def expand_around_center(left: int, right: int) -> int:
        """
        从中心(left, right)向两边扩展，返回回文长度
        奇数回文：left == right（单个字符中心）
        偶数回文：right == left + 1（两字符中间）
        """
        while left >= 0 and right < len(s) and s[left] == s[right]:
            left -= 1   # 向左扩展
            right += 1  # 向右扩展
        # 循环结束时，s[left] != s[right]，实际回文是 s[left+1...right-1]
        return right - left - 1  # 回文长度

    if not s:
        return ""

    start = 0  # 最长回文的起始位置
    max_len = 0  # 最长回文的长度

    for i in range(len(s)):
        # 情况1：奇数长度回文（中心是单个字符）
        len1 = expand_around_center(i, i)
        # 情况2：偶数长度回文（中心是两个字符之间）
        len2 = expand_around_center(i, i + 1)

        # 取两种情况的最大值
        current_len = max(len1, len2)

        # 如果找到更长的回文，更新答案
        if current_len > max_len:
            max_len = current_len
            # 计算起始位置：中心在i，长度current_len
            # 起始位置 = i - (长度-1)//2
            start = i - (current_len - 1) // 2

    return s[start:start + max_len]


# ✅ 测试
print(longest_palindrome_expand("babad"))  # 期望输出："bab" 或 "aba"
print(longest_palindrome_expand("cbbd"))   # 期望输出："bb"
print(longest_palindrome_expand("a"))      # 期望输出："a"
print(longest_palindrome_expand("ac"))     # 期望输出："a" 或 "c"
```

### 复杂度分析
- **时间复杂度**：O(n²) — 有 2n-1 个中心，每个中心最多扩展 O(n)，总共 O(n²)
  - 具体地说：如果输入规模 n=1000，大约需要 10^6 次操作，**能通过 ✅**
- **空间复杂度**：O(1) — 只用了几个变量，没有额外数组

---

## 🚀 解法三：动态规划（DP进阶）

### 优化思路
用二维DP表记录每个子串是否为回文，避免重复计算。

**状态定义**：`dp[i][j]` 表示子串 `s[i...j]` 是否为回文

**状态转移方程**：
```
dp[i][j] = True  当且仅当：
  1. s[i] == s[j]（首尾字符相同）
  2. 内部子串 s[i+1...j-1] 也是回文（即 dp[i+1][j-1] == True）
     特殊情况：如果 j - i < 2（长度<=2），只需满足 s[i] == s[j]
```

> 💡 **关键想法**：长回文依赖短回文的结果，从短到长逐步构建

### 图解过程

```
示例：s = "babad"

DP表构建过程（✅=回文，❌=非回文）：

    j → 0   1   2   3   4
i ↓     b   a   b   a   d
0 (b)   ✅  ❌  ✅  ❌  ❌
1 (a)       ✅  ❌  ✅  ❌
2 (b)           ✅  ❌  ❌
3 (a)               ✅  ❌
4 (d)                   ✅

填表顺序（按子串长度递增）：
长度1：dp[0][0]=✅, dp[1][1]=✅, ... (单字符必定回文)
长度2：dp[0][1]: s[0]!=s[1] → ❌
       dp[1][2]: s[1]!=s[2] → ❌
       dp[2][3]: s[2]!=s[3] → ❌
       dp[3][4]: s[3]!=s[4] → ❌
长度3：dp[0][2]: s[0]==s[2] && dp[1][1]=✅ → ✅ ("bab")
       dp[1][3]: s[1]==s[3] && dp[2][2]=✅ → ✅ ("aba")
       dp[2][4]: s[2]!=s[4] → ❌
长度4：dp[0][3]: s[0]!=s[3] → ❌
       dp[1][4]: s[1]!=s[4] → ❌
长度5：dp[0][4]: s[0]!=s[4] → ❌

最长回文：dp[0][2]=✅ 或 dp[1][3]=✅，长度3
```

### Python代码

```python
def longest_palindrome_dp(s: str) -> str:
    """
    解法三：动态规划
    思路：dp[i][j] 表示 s[i...j] 是否为回文
    """
    n = len(s)
    if n < 2:
        return s

    # 初始化DP表（False表示非回文）
    dp = [[False] * n for _ in range(n)]

    # 单字符都是回文
    for i in range(n):
        dp[i][i] = True

    start = 0  # 最长回文的起始位置
    max_len = 1  # 最长回文的长度

    # 按子串长度递增填表（从长度2开始）
    for length in range(2, n + 1):
        for i in range(n - length + 1):
            j = i + length - 1  # 终点位置

            if s[i] == s[j]:  # 首尾字符相同
                # 长度<=3 或 内部子串是回文
                if length <= 3 or dp[i + 1][j - 1]:
                    dp[i][j] = True
                    if length > max_len:
                        max_len = length
                        start = i

    return s[start:start + max_len]


# ✅ 测试
print(longest_palindrome_dp("babad"))  # 期望输出："bab" 或 "aba"
print(longest_palindrome_dp("cbbd"))   # 期望输出："bb"
print(longest_palindrome_dp("a"))      # 期望输出："a"
```

### 复杂度分析
- **时间复杂度**：O(n²) — 填充 n×n 的DP表
- **空间复杂度**：O(n²) — DP表占用 n×n 的空间

---

## 🐍 Pythonic 写法

利用 Python 的字符串切片和逆序特性，可以简化回文检查：

```python
# 方法一：利用字符串切片的逆序检查
def longest_palindrome_pythonic(s: str) -> str:
    """一行判断回文 + 内置max"""
    n = len(s)
    # 生成所有子串，过滤出回文，返回最长的
    palindromes = [s[i:j] for i in range(n) for j in range(i+1, n+1) if s[i:j] == s[i:j][::-1]]
    return max(palindromes, key=len) if palindromes else ""

# 方法二：配合中心扩展的简洁版
def longest_palindrome_compact(s: str) -> str:
    """紧凑版中心扩展"""
    def expand(l, r):
        while l >= 0 and r < len(s) and s[l] == s[r]:
            l -= 1
            r += 1
        return r - l - 1

    start, end = 0, 0
    for i in range(len(s)):
        len1, len2 = expand(i, i), expand(i, i + 1)
        max_len = max(len1, len2)
        if max_len > end - start:
            start = i - (max_len - 1) // 2
            end = start + max_len
    return s[start:end]

print(longest_palindrome_compact("babad"))  # "bab" 或 "aba"
```

> ⚠️ **面试建议**：先写清晰版本展示思路（解法二），再提 Pythonic 写法展示语言功底。
> 面试官更看重你的**思考过程**，而非代码行数。

---

## 📊 解法对比

| 维度 | 解法一：暴力枚举 | 解法二：中心扩展 | 解法三：动态规划 |
|------|--------------|--------------|--------------|
| 时间复杂度 | O(n³) | O(n²) | O(n²) |
| 空间复杂度 | O(1) | O(1) | O(n²) |
| 代码难度 | 简单 | 中等 | 较难 |
| 面试推荐 | ⭐ | ⭐⭐⭐ | ⭐⭐ |
| 适用场景 | 小数据演示思路 | **面试首选，时空平衡** | 展示DP能力 |

**面试建议**：
1. 先用1分钟讲暴力法思路（展示基本理解）
2. 立刻优化到**中心扩展法**（展示优化能力），这是最推荐的解法
3. 如果面试官追问"还有其他方法吗？"，再提DP（展示算法广度）
4. 如果时间充裕，提一句Manacher算法（O(n)线性时间，但面试中很少考）

---

## 🎤 面试现场

> 模拟面试中的完整对话流程，帮你练习"边想边说"。

**面试官**：请你解决一下这道题——找出字符串中最长的回文子串。

**你**：（审题30秒）好的，这道题要求找最长回文子串。让我先确认一下，回文是指正着读和倒着读一样的字符串，比如"aba"对吧？

**面试官**：对的。

**你**：我的第一个想法是暴力法：枚举所有子串，逐个检查是否回文。这需要两层循环枚举 O(n²) 个子串，每次检查回文需要 O(n)，总时间复杂度 O(n³)，对于n=1000会超时。

不过我们可以优化！回文的核心特征是**中心对称**，我们可以用**中心扩展法**：
- 对每个可能的中心（包括单字符中心和两字符之间的中心）
- 向两边同时扩展，只要左右字符相同就继续
- 时间复杂度降到 O(n²)，空间复杂度 O(1)

**面试官**：很好，请写一下代码。

**你**：（边写边说）我先写一个辅助函数 `expand_around_center`，接收左右边界，向两边扩展直到不匹配。然后在主函数中，对每个位置i，分别尝试奇数回文和偶数回文...（写出解法二的代码）

**面试官**：测试一下？

**你**：用示例 "babad" 走一遍：
- 中心在索引1的'a'，向两边扩展得到"bab"，长度3
- 中心在索引2的'b'，向两边扩展得到"aba"，长度3
- 最终返回长度3的回文

再测一个边界情况 "cbbd"：
- 中心在索引1和2之间（"bb"），扩展后得到"bb"，长度2
- 结果正确 ✅

**面试官**：如果字符串特别长（比如10^5），有没有更快的方法？

**你**：有！可以用**Manacher算法**，能做到O(n)线性时间，但实现比较复杂。它的核心思想是利用已知回文的信息来跳过部分计算，维护一个"回文右边界"来避免重复扩展。不过在实际面试中，中心扩展法的O(n²)通常已经足够了。

### 高频追问

| 追问 | 应答策略 |
|------|---------|
| "还有更优解吗？" | "有Manacher算法可以做到O(n)，但实现较复杂，工程中很少用。中心扩展法的O(n²)通常已经足够满足要求。" |
| "如果要返回所有最长回文怎么办？" | "用一个列表保存所有达到最大长度的回文，在更新max_len时判断是否相等。" |
| "能用DP做吗？" | "可以！用dp[i][j]表示s[i...j]是否回文，状态转移方程是dp[i][j] = (s[i]==s[j]) && dp[i+1][j-1]。时间O(n²)，空间O(n²)。" |
| "空间能不能O(1)？" | "中心扩展法已经是O(1)空间。DP法需要O(n²)空间存表，无法优化到O(1)。" |
| "实际工程中怎么用？" | "DNA序列分析中检测回文结构，文本处理中查找对称模式，数据压缩中利用回文特性。" |

---

## 🎓 知识点总结

### Python技巧卡片 🐍
```python
# 技巧1：字符串切片逆序 — 快速判断回文
s = "aba"
is_palindrome = (s == s[::-1])  # True

# 技巧2：利用max的key参数找最长字符串
strings = ["a", "bab", "bb"]
longest = max(strings, key=len)  # "bab"

# 技巧3：推导式生成所有子串
s = "abc"
substrings = [s[i:j] for i in range(len(s)) for j in range(i+1, len(s)+1)]
# ['a', 'ab', 'abc', 'b', 'bc', 'c']
```

### 💡 底层原理（选读）

> **为什么中心扩展比DP更省空间？**
>
> - **DP法**需要一个 n×n 的二维表存储所有子串的回文状态，空间 O(n²)
> - **中心扩展法**只需要几个变量（start, max_len, left, right），空间 O(1)
> - 核心区别：DP需要"记住"所有中间结果，而中心扩展法"即用即扔"，只保留最优答案
>
> **字符串切片 `s[::-1]` 的原理？**
>
> - Python 的切片语法 `s[start:end:step]`，当 step=-1 时表示逆序
> - 底层实现：创建一个新字符串，从后往前复制字符，时间 O(n)
> - 注意：每次 `s[::-1]` 都会创建新对象，频繁调用会影响性能

### 算法模式卡片 📐
- **模式名称**：中心扩展法（Expand Around Center）
- **适用条件**：寻找回文子串、回文子序列、对称结构
- **识别关键词**：题目中出现"回文"、"对称"、"镜像"
- **模板代码**：
```python
def expand_around_center(s: str, left: int, right: int) -> int:
    """从中心(left, right)向两边扩展，返回回文长度"""
    while left >= 0 and right < len(s) and s[left] == s[right]:
        left -= 1
        right += 1
    return right - left - 1  # 实际回文长度

# 主函数中枚举所有中心
for i in range(len(s)):
    len1 = expand_around_center(s, i, i)      # 奇数回文
    len2 = expand_around_center(s, i, i + 1)  # 偶数回文
    max_len = max(len1, len2)
```

### 易错点 ⚠️
1. **忘记处理偶数长度回文** — 回文的中心不仅是单个字符，还可能在两个字符之间！必须同时检查 `(i, i)` 和 `(i, i+1)` 两种情况。
   - ❌ 错误：只考虑 `expand(i, i)`，会漏掉 "abba" 这种偶数回文
   - ✅ 正确：同时考虑 `expand(i, i)` 和 `expand(i, i+1)`

2. **起始位置计算错误** — 从中心长度计算起始位置时容易出错
   - 公式：`start = center - (length - 1) // 2`
   - 示例：中心在索引2，长度5，起始位置 = 2 - (5-1)//2 = 0
   - ❌ 错误：写成 `start = center - length // 2`
   - ✅ 正确：`start = i - (current_len - 1) // 2`

3. **边界条件遗漏** — while 循环的边界判断顺序很重要
   - ❌ 错误：`while s[left] == s[right] and left >= 0 and right < len(s)` → 先访问可能越界
   - ✅ 正确：`while left >= 0 and right < len(s) and s[left] == s[right]` → 先检查边界

---

## 🏗️ 工程实战（选读）

> 这个算法思想在真实项目中的应用，让你知道"学了有什么用"。

- **场景1：DNA序列分析** — 生物信息学中检测DNA回文结构（限制性内切酶识别位点）。中心扩展法能高效找到对称序列，用于基因编辑和蛋白质折叠研究。

- **场景2：文本相似度检测** — 判断两段文本是否为"变体"（如逆序抄袭）。先提取最长回文，再比对相似度，用于学术论文查重。

- **场景3：数据压缩** — 利用回文结构的对称性，只存储一半数据，通过镜像恢复另一半，节省存储空间（应用于图像/视频编码）。

---

## 🏋️ 举一反三

完成本课后，试试这些同类题目来巩固知识：

| 题目 | 难度 | 相关知识点 | 提示 |
|------|------|-----------|------|
| LeetCode 647. 回文子串 | Medium | 中心扩展计数 | 不求最长，而是求回文子串的总数量 |
| LeetCode 516. 最长回文子序列 | Medium | DP（子序列非子串） | 注意"子序列"可以不连续，需要二维DP |
| LeetCode 9. 回文数 | Easy | 数字回文 | 不转字符串，用数学方法翻转数字 |
| LeetCode 131. 分割回文串 | Medium | 回溯+回文判断 | 把字符串分割成多个回文子串，用回溯枚举 |
| LeetCode 214. 最短回文串 | Hard | KMP + 回文 | 在字符串前添加最少字符使其成为回文 |

---

## 📝 课后小测

试试这道变体题，不要看答案，自己先想5分钟！

**题目**：给定字符串 `s`，你可以删除其中一个字符。判断删除后能否形成回文串。

例如：
- 输入：`s = "abca"`，输出：`True`（删除 'b' 或 'c' 后得到 "aca" 或 "aba"）
- 输入：`s = "abc"`，输出：`False`

<details>
<summary>💡 提示（实在想不出来再点开）</summary>

用双指针从两端向中间移动，当遇到不匹配时，尝试跳过左边或右边的字符，检查剩余部分是否回文。

</details>

<details>
<summary>✅ 参考答案</summary>

```python
def valid_palindrome_ii(s: str) -> bool:
    """
    验证回文串II：允许删除一个字符
    思路：双指针 + 分情况讨论
    """
    def is_palindrome(sub: str) -> bool:
        """辅助函数：检查子串是否回文"""
        return sub == sub[::-1]

    left, right = 0, len(s) - 1

    while left < right:
        if s[left] != s[right]:
            # 遇到不匹配，尝试删除左边或右边的字符
            # 情况1：删除左边的字符，检查s[left+1...right]
            # 情况2：删除右边的字符，检查s[left...right-1]
            return is_palindrome(s[left+1:right+1]) or is_palindrome(s[left:right])
        left += 1
        right -= 1

    return True  # 原本就是回文


# ✅ 测试
print(valid_palindrome_ii("abca"))  # True (删除'b'或'c')
print(valid_palindrome_ii("abc"))   # False
print(valid_palindrome_ii("aba"))   # True (本身回文)
```

**核心思路**：
1. 用双指针从两端向中间匹配
2. 第一次遇到不匹配时，尝试"跳过左边"或"跳过右边"
3. 检查跳过后的剩余部分是否回文
4. 时间复杂度 O(n)，空间复杂度 O(1)

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
