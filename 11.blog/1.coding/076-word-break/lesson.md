# 📖 第76课:单词拆分

> **模块**:动态规划 | **难度**:Medium ⭐⭐⭐
> **LeetCode 链接**:https://leetcode.cn/problems/word-break/
> **前置知识**:第71课(爬楼梯)、第75课(零钱兑换)
> **预计学习时间**:30分钟

---

## 🎯 题目描述

给你一个字符串 `s` 和一个字符串列表 `wordDict` 作为字典。请你判断是否可以利用字典中的单词拼接出 `s`。字典中的单词可以重复使用,但拼接时不能有字符剩余或重叠。

**示例:**
```
输入:s = "leetcode", wordDict = ["leet", "code"]
输出:true
解释:"leetcode" 可以由 "leet" 和 "code" 拼接而成
```

```
输入:s = "applepenapple", wordDict = ["apple", "pen"]
输出:true
解释:"applepenapple" 可以由 "apple", "pen", "apple" 拼接而成
```

```
输入:s = "catsandog", wordDict = ["cats", "dog", "sand", "and", "cat"]
输出:false
```

**约束条件:**
- 1 <= s.length <= 300
- 1 <= wordDict.length <= 1000
- 1 <= wordDict[i].length <= 20
- 字典中的单词可以重复使用
- 所有字符串仅由小写英文字母组成

---

### 🧪 边界用例(面试必考)

| 用例类型 | 输入 | 期望输出 | 考察点 |
|---------|------|---------|--------|
| 空字符串 | s="", wordDict=["a"] | true | 空串认为可拆分 |
| 单个单词 | s="apple", wordDict=["apple"] | true | 基本功能 |
| 重复使用 | s="aaaa", wordDict=["aa"] | true | 单词可重复 |
| 前缀陷阱 | s="aaab", wordDict=["aa","aaa"] | false | 贪心失败案例 |
| 无解情况 | s="catsandog", wordDict=["cats","dog","sand","and","cat"] | false | "sandog"无法拆分 |

---

## 💡 思路引导

### 生活化比喻
> 想象你在玩拼图游戏,有一串字母 "leetcode",手上有若干单词卡片["leet", "code"]可以无限次使用...
>
> 🐌 **笨办法**:从左到右尝试匹配。先试"leet"能匹配,剩下"code"继续试...如果遇到"applepenapple"就麻烦了,可能试"apple"、试"pen"、再试"apple",每次都要回溯重试,组合爆炸!
>
> 🚀 **聪明办法**:建一张"可拆分表",从左到右标记"前i个字符能否拆分"。比如标记"leet"(前4个字符)可拆分,那么看"code"能不能接上去。只要找到某个中间位置j,使得"前j个字符可拆分"且"j到i是个单词",那么"前i个字符"也可拆分!这样只需要扫一遍字符串。

### 关键洞察
**字符串DP的核心:用 `dp[i]` 表示"前i个字符能否拆分",枚举所有可能的分割点!**

---

## 🧠 解题思维链

> 这一节模拟你在面试中"从零开始思考"的过程。

### Step 1:理解题目 → 锁定输入输出
- **输入**:字符串s + 字典wordDict
- **输出**:布尔值,能否完全拆分
- **限制**:字典中单词可重复使用,必须恰好拼接成s(不能有剩余)

### Step 2:先想笨办法(回溯暴力)
从位置0开始,尝试所有可能的单词:
- 如果s[0:len(word)] == word,递归判断剩余部分s[len(word):]
- 如果所有单词都试完仍无解,返回False

```python
def word_break_backtrack(s, wordDict):
    def dfs(start):
        if start == len(s): return True  # 全部匹配完
        for word in wordDict:
            if s[start:start+len(word)] == word:
                if dfs(start + len(word)):  # 递归尝试剩余部分
                    return True
        return False
    return dfs(0)
```

- 时间复杂度:O(2^n) — 每个位置可能有多个单词匹配,递归树指数级
- 瓶颈在哪:**重复计算**,比如s="aaaa",可能多次计算"从位置2开始能否拆分"

### Step 3:瓶颈分析 → 优化方向
递归树中"从位置i开始能否拆分"被重复计算。
- 核心问题:每个起始位置的结果被重复计算
- 优化思路:用数组 `dp[i]` 记录"前i个字符能否拆分",从左到右填表

### Step 4:选择武器
- 选用:**字符串DP**
- 理由:将大问题(整个字符串)拆成子问题(前i个字符),每个位置只判断一次

> 🔑 **模式识别提示**:当题目出现"字符串"+"能否完全匹配"+"可重复使用资源",优先考虑"字符串DP"

---

## 🔑 解法一:动态规划(自底向上)

### 思路
定义 `dp[i]` 表示前i个字符能否拆分。对于每个位置i,枚举所有可能的分割点j,如果 `dp[j]` 为真且 `s[j:i]` 是字典中的单词,则 `dp[i]` 为真。

### 图解过程

```
示例:s = "leetcode", wordDict = ["leet", "code"]

初始化 dp 数组(dp[i]表示前i个字符能否拆分):
dp = [True, False, False, False, False, False, False, False, False]
      0     1      2      3      4      5      6      7      8
      ""    l      le     lee    leet   leetc  leetco leetcod leetcode

遍历每个位置 i=1 to 8:

i=1("l"):
  尝试分割点j=0:s[0:1]="l" 不在字典,dp[1]=False

i=2("le"):
  尝试分割点j=0:s[0:2]="le" 不在字典,dp[2]=False

i=3("lee"):
  尝试分割点j=0:s[0:3]="lee" 不在字典,dp[3]=False

i=4("leet"):
  尝试分割点j=0:s[0:4]="leet" 在字典!dp[0]=True
  ✅ dp[4] = True

i=5("leetc"):
  尝试分割点j=0,1,2,3,4:都不满足条件,dp[5]=False

i=6("leetco"):
  尝试分割点j=0,1,2,3,4,5:都不满足,dp[6]=False

i=7("leetcod"):
  尝试分割点j=0,1,2,3,4,5,6:都不满足,dp[7]=False

i=8("leetcode"):
  尝试分割点j=4:dp[4]=True 且 s[4:8]="code" 在字典!
  ✅ dp[8] = True

最终 dp[8] = True,返回 true
```

### Python代码

```python
from typing import List


def word_break(s: str, wordDict: List[str]) -> bool:
    """
    解法一:动态规划(自底向上)
    思路:dp[i]表示前i个字符能否拆分,枚举分割点j
    """
    n = len(s)
    # 转为集合加速查找
    word_set = set(wordDict)

    # 初始化:dp[i]表示前i个字符能否拆分
    dp = [False] * (n + 1)
    dp[0] = True  # 空字符串认为可拆分

    # 遍历每个位置
    for i in range(1, n + 1):
        # 枚举所有可能的分割点j
        for j in range(i):
            # 如果前j个字符可拆分 且 [j, i)是字典中的单词
            if dp[j] and s[j:i] in word_set:
                dp[i] = True
                break  # 找到一个即可,无需继续

    return dp[n]


# ✅ 测试
print(word_break("leetcode", ["leet", "code"]))           # 期望输出:True
print(word_break("applepenapple", ["apple", "pen"]))      # 期望输出:True
print(word_break("catsandog", ["cats","dog","sand","and","cat"]))  # 期望输出:False
print(word_break("aaaa", ["aa"]))                         # 期望输出:True (重复使用)
```

### 复杂度分析
- **时间复杂度**:O(n² × L) — n是字符串长度,L是单词平均长度。外层循环O(n),内层枚举分割点O(n),每次切片和查找O(L)
  - 具体地说:如果s长度n=300,单词平均长度L=10,大约需要 300 × 300 × 10 = 900000 次操作
- **空间复杂度**:O(n + m×L) — dp数组O(n),word_set存储m个单词共O(m×L)

### 优缺点
- ✅ 逻辑清晰,易于理解
- ✅ 时间复杂度已达最优(必须检查每个位置)
- ⚠️ 切片操作 `s[j:i]` 有额外开销,可优化

---

## 🏆 解法二:优化DP(按单词长度枚举)(最优解)

### 优化思路
解法一中枚举所有分割点j效率不高。观察到字典中单词长度有限(最大20),可以反过来:对于每个位置i,只枚举字典中的单词,检查s是否以该单词结尾!

> 💡 **关键想法**:与其枚举所有分割点,不如直接尝试匹配字典中的单词,效率更高!

### 图解过程

```
示例:s = "leetcode", wordDict = ["leet", "code"]

初始化:dp = [True, False, ..., False]

i=4时:
  尝试单词"leet"(长度4):
    s[0:4]="leet" 匹配!且 dp[0]=True
    ✅ dp[4] = True

i=8时:
  尝试单词"code"(长度4):
    s[4:8]="code" 匹配!且 dp[4]=True
    ✅ dp[8] = True

只需要尝试字典中的单词,避免枚举所有分割点!
```

### Python代码

```python
from typing import List


def word_break_optimized(s: str, wordDict: List[str]) -> bool:
    """
    解法二:优化DP(按单词长度枚举) 🏆
    思路:对每个位置i,直接尝试匹配字典中的单词,而非枚举所有分割点
    """
    n = len(s)
    word_set = set(wordDict)

    dp = [False] * (n + 1)
    dp[0] = True

    for i in range(1, n + 1):
        # 直接尝试字典中的每个单词
        for word in word_set:
            word_len = len(word)
            # 如果单词长度不超过i,且s以该单词结尾
            if i >= word_len and dp[i - word_len] and s[i - word_len:i] == word:
                dp[i] = True
                break  # 找到一个即可

    return dp[n]


# ✅ 测试
print(word_break_optimized("leetcode", ["leet", "code"]))           # 期望输出:True
print(word_break_optimized("applepenapple", ["apple", "pen"]))      # 期望输出:True
print(word_break_optimized("catsandog", ["cats","dog","sand","and","cat"]))  # 期望输出:False
```

### 复杂度分析
- **时间复杂度**:O(n × m × L) — n是字符串长度,m是字典单词数,L是单词平均长度。相比解法一的O(n²×L),当m << n时更快!
  - 具体地说:如果n=300,m=10,L=10,只需 300 × 10 × 10 = 30000 次操作(比解法一快30倍!)
- **空间复杂度**:O(n + m×L) — 相同

---

## ⚡ 解法三:Trie树优化(进阶)

### 优化思路
将字典构建成Trie树,对于每个位置i,从i向右匹配Trie,找到所有可能的单词结尾。避免了字符串切片和集合查找。

### Python代码

```python
from typing import List


class TrieNode:
    def __init__(self):
        self.children = {}
        self.is_end = False


def word_break_trie(s: str, wordDict: List[str]) -> bool:
    """
    解法三:Trie树优化
    思路:用Trie树存储字典,对每个位置向右匹配
    """
    # 构建Trie树
    root = TrieNode()
    for word in wordDict:
        node = root
        for ch in word:
            if ch not in node.children:
                node.children[ch] = TrieNode()
            node = node.children[ch]
        node.is_end = True

    n = len(s)
    dp = [False] * (n + 1)
    dp[0] = True

    for i in range(n):
        if not dp[i]:
            continue  # 如果前i个字符无法拆分,跳过
        # 从位置i开始向右匹配Trie
        node = root
        for j in range(i, n):
            ch = s[j]
            if ch not in node.children:
                break  # 无法继续匹配
            node = node.children[ch]
            if node.is_end:  # 找到一个单词
                dp[j + 1] = True

    return dp[n]


# ✅ 测试
print(word_break_trie("leetcode", ["leet", "code"]))           # 期望输出:True
print(word_break_trie("applepenapple", ["apple", "pen"]))      # 期望输出:True
```

### 复杂度分析
- **时间复杂度**:O(n² + m×L) — 构建Trie O(m×L),DP部分每个位置最多匹配n个字符
- **空间复杂度**:O(m×L×26) — Trie树空间

---

## 🐍 Pythonic 写法

利用 any() 和生成器表达式:

```python
def word_break_pythonic(s: str, wordDict: List[str]) -> bool:
    word_set = set(wordDict)
    dp = [True] + [False] * len(s)

    for i in range(1, len(s) + 1):
        dp[i] = any(dp[j] and s[j:i] in word_set for j in range(i))

    return dp[-1]
```

解释:
- `any(...)` 只要有一个条件为真就返回True,比手动循环更简洁
- 列表推导 `[True] + [False] * len(s)` 初始化dp数组

> ⚠️ **面试建议**:先写清晰的循环版本展示思路,再提Pythonic写法展示语言功底。

---

## 📊 解法对比

| 维度 | 解法一:枚举分割点 | 🏆 解法二:枚举单词(最优) | 解法三:Trie树 |
|------|-----------------|----------------------|-------------|
| 时间复杂度 | O(n²×L) | **O(n×m×L)** ← m<<n时更快 | O(n² + m×L) |
| 空间复杂度 | O(n + m×L) | **O(n + m×L)** | O(m×L×26) |
| 代码难度 | 简单 | **简单** ← 逻辑更清晰 | 较难 |
| 面试推荐 | ⭐⭐ | **⭐⭐⭐** ← 首选 | ⭐ |
| 适用场景 | 通用 | **字典单词数较少** | 字典巨大且查询频繁 |

**为什么解法二是最优解**:
- 当字典单词数m远小于字符串长度n时,时间复杂度从O(n²×L)优化到O(n×m×L),提升巨大
- 代码逻辑更清晰:直接尝试匹配单词,而非枚举抽象的分割点
- 实际性能更好:避免了大量无效的分割点检查

**面试建议**:
1. 先口述思路:"这是字符串DP问题,用dp[i]表示前i个字符能否拆分"
2. 提出优化:"与其枚举所有分割点,不如直接尝试字典中的单词,效率更高"
3. 写🏆解法二的代码
4. 强调关键点:"将wordDict转为set加速查找,dp[0]=True表示空串可拆分"
5. 手动测试边界用例(空串、单个单词、重复使用)

---

## 🎤 面试现场

> 模拟面试中的完整对话流程,帮你练习"边想边说"。

**面试官**:请你解决一下这道题。

**你**:(审题30秒)好的,这是一道经典的字符串DP问题。要判断字符串s能否由字典中的单词拼接而成,单词可以重复使用。

我的思路是用动态规划:定义 `dp[i]` 表示前i个字符能否拆分。对于每个位置i,尝试字典中的每个单词,如果s以该单词结尾且前面部分可拆分,则dp[i]为真。

这样时间复杂度是O(n × m × L),其中n是字符串长度,m是字典单词数,L是单词平均长度。

**面试官**:很好,请写一下代码。

**你**:(边写边说)
```python
def word_break(s, wordDict):
    n = len(s)
    word_set = set(wordDict)  # 转为集合加速查找
    dp = [False] * (n + 1)
    dp[0] = True  # 空字符串可拆分

    for i in range(1, n + 1):
        for word in word_set:
            word_len = len(word)
            # 如果s以word结尾,且前面部分可拆分
            if i >= word_len and dp[i - word_len] and s[i - word_len:i] == word:
                dp[i] = True
                break  # 找到一个即可

    return dp[n]
```

核心是 `dp[i]` 表示前i个字符能否拆分,内层循环直接尝试匹配字典中的单词,避免枚举所有分割点。

**面试官**:测试一下?

**你**:用示例 `s="leetcode", wordDict=["leet","code"]` 走一遍...
- i=4时,尝试"leet",s[0:4]="leet"匹配!dp[0]=True,所以dp[4]=True
- i=8时,尝试"code",s[4:8]="code"匹配!dp[4]=True,所以dp[8]=True
- 返回dp[8]=True,结果正确!

再测边界情况 `s=""` 返回True(空串可拆分),`s="catsandog"` 无解返回False,结果正确!

### 高频追问

| 追问 | 应答策略 |
|------|---------|
| "还有更优解吗?" | "时间已接近最优。可以用Trie树优化字符串匹配,但代码复杂度增加,实际提升不大。对于本题数据范围,当前解法足够高效" |
| "能输出所有可能的拆分方案吗?" | "可以!在dp基础上加回溯:从dp[n]往回找,记录每次使用的单词。时间复杂度可能达到O(2^n)因为方案数可能很多" |
| "如果字典非常大怎么办?" | "可以用Trie树存储字典,将查找从O(L)优化到O(L)(虽然渐进复杂度相同,但常数更小)。或者用字符串哈希避免切片操作" |
| "为什么不能用贪心?" | "反例:s='aaab',wordDict=['aa','aaa']。贪心选'aaa'会导致剩余'ab'无法拆分,但选'aa'+'aa'可以拆分成'aaab'...等等,这个反例本身也无解。真正的反例需要更复杂构造,但总之贪心无法保证全局最优" |

---

## 🎓 知识点总结

### Python技巧卡片 🐍
```python
# 技巧1:列表转集合加速查找
word_set = set(wordDict)  # O(1)查找 vs 列表的O(n)

# 技巧2:字符串切片 s[i:j]
s[0:4]  # 左闭右开,取索引0,1,2,3
s[i - word_len:i]  # 取以i结尾的word_len个字符

# 技巧3:any() 判断是否存在
any(condition for item in items)  # 只要有一个True就返回True
```

### 💡 底层原理(选读)

> **为什么是字符串DP而非完全背包?**
> - 完全背包:物品无序,可以任意排列组合(如零钱兑换:5+1和1+5相同)
> - 字符串DP:有序匹配,必须按照字符串顺序拼接
>
> 本题虽然单词可以重复使用,但必须按顺序拼接成s,所以不是完全背包,而是字符串DP。
>
> **为什么解法二比解法一快?**
> - 解法一:枚举所有分割点j(0到i-1),即使大部分分割点无效
> - 解法二:只尝试字典中的m个单词,当m << n时大幅减少计算
> - 极端情况:如果字典只有2个单词,解法二内层只循环2次,而解法一要循环n次

### 算法模式卡片 📐
- **模式名称**:字符串DP
- **适用条件**:判断字符串能否由某些子串组成/匹配
- **识别关键词**:"字符串拆分"、"子串匹配"、"能否组成"
- **模板代码**:
```python
def string_dp(s, patterns):
    n = len(s)
    dp = [False] * (n + 1)
    dp[0] = True

    for i in range(1, n + 1):
        for pattern in patterns:
            plen = len(pattern)
            if i >= plen and dp[i - plen] and s[i - plen:i] == pattern:
                dp[i] = True
                break

    return dp[n]
```

### 易错点 ⚠️
1. **初始化错误**:忘记设置 `dp[0] = True`,导致所有位置都无法转移
   - 错误:`dp = [False] * (n + 1)`
   - 正确:`dp[0] = True; dp = [False] * (n + 1); dp[0] = True`

2. **字符串切片越界**:未检查 `i >= word_len` 就切片
   - 错误:`if s[i - word_len:i] == word`
   - 正确:`if i >= word_len and s[i - word_len:i] == word`

3. **忘记转集合**:直接用列表查找 `word in wordDict`,时间复杂度O(m)
   - 错误:`if word in wordDict`
   - 正确:`word_set = set(wordDict); if word in word_set`

---

## 🏗️ 工程实战(选读)

> 这个算法思想在真实项目中的应用,让你知道"学了有什么用"。

- **场景1**:搜索引擎查询分词 — 用户输入"iphone手机壳",需要拆分成["iphone", "手机壳"]或["iphone", "手机", "壳"],判断是否为有效查询
- **场景2**:自然语言处理 — 中文分词系统判断一段文本能否由词典中的词语组成
- **场景3**:URL路由匹配 — 判断请求路径能否由预定义的路由规则拼接而成

---

## 🏋️ 举一反三

完成本课后,试试这些同类题目来巩固知识:

| 题目 | 难度 | 相关知识点 | 提示 |
|------|------|-----------|------|
| LeetCode 140. 单词拆分II | Hard | 字符串DP+回溯 | 在本题基础上加回溯输出所有方案 |
| LeetCode 472. 连接词 | Hard | 字符串DP | 判断单词能否由其他单词拼接而成 |
| LeetCode 44. 通配符匹配 | Hard | 二维字符串DP | 支持*和?的字符串匹配 |
| LeetCode 583. 两个字符串的删除操作 | Medium | 二维字符串DP | 最少删除次数使两个字符串相同 |

---

## 📝 课后小测

试试这道变体题,不要看答案,自己先想5分钟!

**题目**:现在不仅要判断能否拆分,还要输出所有可能的拆分方案。例如s="catsanddog",wordDict=["cat","cats","and","sand","dog"],输出[["cats","and","dog"], ["cat","sand","dog"]]

<details>
<summary>💡 提示(实在想不出来再点开)</summary>

在DP的基础上加回溯:从dp[n]往回找,每次找到一个有效分割点就记录单词,递归处理前半部分。

</details>

<details>
<summary>✅ 参考答案</summary>

```python
def word_break_all(s, wordDict):
    word_set = set(wordDict)
    n = len(s)

    # 先用DP判断每个位置能否拆分
    dp = [False] * (n + 1)
    dp[0] = True
    for i in range(1, n + 1):
        for word in word_set:
            wlen = len(word)
            if i >= wlen and dp[i - wlen] and s[i - wlen:i] == word:
                dp[i] = True
                break

    if not dp[n]:  # 无法拆分
        return []

    # 回溯找出所有方案
    result = []
    def backtrack(index, path):
        if index == 0:
            result.append(path[::-1])  # 倒序输出
            return
        for word in word_set:
            wlen = len(word)
            if index >= wlen and dp[index - wlen] and s[index - wlen:index] == word:
                path.append(word)
                backtrack(index - wlen, path)
                path.pop()

    backtrack(n, [])
    return result
```

核心思路:DP判断可行性,回溯枚举所有方案。时间复杂度O(2^n)因为方案数可能很多。

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
