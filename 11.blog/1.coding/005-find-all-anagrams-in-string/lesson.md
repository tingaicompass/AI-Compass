> 想系统提升编程能力、查看更完整的学习路线，欢迎访问 AI Compass：https://github.com/tingaicompass/AI-Compass
> 仓库持续更新刷题题解、Python 基础和 AI 实战内容，适合想高效进阶的你。

# 📖 第5课:找到字符串中所有字母异位词

> **模块**:哈希表 | **难度**:Medium ⭐⭐
> **LeetCode 链接**:https://leetcode.cn/problems/find-all-anagrams-in-a-string/
> **前置知识**:第2课(字母异位词分组)
> **预计学习时间**:25分钟

---

## 🎯 题目描述

给定两个字符串 `s` 和 `p`,找出 `s` 中所有 `p` 的**异位词**的起始索引。

**异位词**是指由相同字母重新排列形成的字符串。

**示例:**
```
输入:s = "cbaebabacd", p = "abc"
输出:[0, 6]
解释:
  起始索引 0 的子串是 "cba",是 "abc" 的异位词
  起始索引 6 的子串是 "bac",是 "abc" 的异位词
```

```
输入:s = "abab", p = "ab"
输出:[0, 1, 2]
解释:
  起始索引 0 的子串是 "ab",是 "ab" 的异位词
  起始索引 1 的子串是 "ba",是 "ab" 的异位词
  起始索引 2 的子串是 "ab",是 "ab" 的异位词
```

**约束条件:**
- 1 ≤ s.length, p.length ≤ 3 × 10⁴
- s 和 p 仅包含小写字母

---

## 🧪 边界用例(面试必考)

| 用例类型 | 输入 | 期望输出 | 考察点 |
|---------|------|---------|--------|
| 最小输入 | s="a", p="a" | [0] | 基本功能 |
| p比s长 | s="ab", p="abc" | [] | 边界处理 |
| 无匹配 | s="abc", p="def" | [] | 无结果情况 |
| 全部匹配 | s="aaa", p="a" | [0,1,2] | 重叠窗口 |
| 重复字符 | s="aaaa", p="aa" | [0,1,2] | 重复元素 |

---

## 💡 思路引导

### 生活化比喻

> 想象你是一位图书管理员,需要在一排书架(字符串s)上找出所有包含特定书籍组合(字符串p)的连续区域。
>
> 🐌 **笨办法**:每看到一个位置,你就从这个位置开始数出p长度的书,然后把这些书搬到桌上排序,再和p的排序结果对比。这样每次都要搬书、排序,太累了!
>
> 🚀 **聪明办法**:你拿一个固定大小的"窗口框"(长度等于p),在书架上滑动。窗口框右边进来一本新书,左边就移出一本旧书。你只需要维护一个"书籍计数表",每次滑动只更新进出的两本书的计数,就能立刻判断窗口内的书是否和p匹配!

### 关键洞察

**固定窗口滑动 + 字符频率比对 = O(n)时间内找出所有异位词!**

---

## 🧠 解题思维链

> 这一节模拟你在面试中"从零开始思考"的过程。

### Step 1:理解题目 → 锁定输入输出
- **输入**:两个字符串 s 和 p,都是小写字母
- **输出**:整数数组,表示所有异位词的起始索引
- **限制**:需要找出**所有**满足条件的位置,不能遗漏

### Step 2:先想笨办法(暴力法)

对于s中每个可能的起始位置i(从0到len(s)-len(p)),截取长度为len(p)的子串,判断是否是p的异位词。

判断异位词的方法:将两个字符串排序后比较,或者比较字符频率。

- 时间复杂度:O(n × m × log m) 或 O(n × m × 26)
  - n = len(s), m = len(p)
  - 对每个位置排序需要 O(m log m),或者计数比较需要 O(m)
- 瓶颈在哪:**每次都要重新统计窗口内的字符频率**,大量重复计算

### Step 3:瓶颈分析 → 优化方向

观察相邻两个窗口:
- 窗口 [i, i+m-1] 的字符频率已经统计好了
- 窗口 [i+1, i+m] 只是右边多了一个字符,左边少了一个字符

核心问题:**能不能利用前一个窗口的信息,O(1)时间更新到下一个窗口?**

优化思路:**滑动窗口**!维护一个大小固定为 len(p) 的窗口,窗口每次右移时:
- 加入右边新字符的频率
- 减去左边移出字符的频率
- O(1)时间完成频率表更新

### Step 4:选择武器
- 选用:**固定大小滑动窗口 + 哈希表(Counter/数组)**
- 理由:
  - 滑动窗口避免重复统计 → O(n)遍历
  - 哈希表 O(1) 更新字符频率
  - 比较两个频率表只需 O(26) = O(1)

> 🔑 **模式识别提示**:当题目出现"连续子串"+"满足某种字符条件",优先考虑"滑动窗口"

---

## 🔑 解法一:暴力统计法(朴素)

### 思路

对s中每个可能的起始位置,截取长度为len(p)的子串,使用Counter统计字符频率并与p的频率比较。

### 图解过程

```
示例:s = "cbaebabacd", p = "abc"
目标频率:Counter({'a':1, 'b':1, 'c':1})

Step 1:检查位置 0,子串 "cba"
  Counter("cba") = {'c':1, 'b':1, 'a':1} ✅ 匹配!

Step 2:检查位置 1,子串 "bae"
  Counter("bae") = {'b':1, 'a':1, 'e':1} ❌ 不匹配

Step 3:检查位置 2,子串 "aeb"
  Counter("aeb") = {'a':1, 'e':1, 'b':1} ❌ 不匹配

...继续到位置 6,子串 "bac"
  Counter("bac") = {'b':1, 'a':1, 'c':1} ✅ 匹配!

结果:[0, 6]
```

### Python代码

```python
from typing import List
from collections import Counter


def findAnagrams(s: str, p: str) -> List[int]:
    """
    解法一:暴力统计法
    思路:对每个位置截取子串并用Counter比较
    """
    result = []
    len_s, len_p = len(s), len(p)

    if len_s < len_p:  # 边界:s比p短,直接返回空
        return result

    # 统计p的字符频率
    p_count = Counter(p)

    # 遍历每个可能的起始位置
    for i in range(len_s - len_p + 1):
        # 截取长度为len_p的子串
        substring = s[i:i + len_p]
        # 统计子串的字符频率并比较
        if Counter(substring) == p_count:
            result.append(i)

    return result


# ✅ 测试
print(findAnagrams("cbaebabacd", "abc"))  # 期望输出:[0, 6]
print(findAnagrams("abab", "ab"))         # 期望输出:[0, 1, 2]
print(findAnagrams("a", "a"))             # 期望输出:[0]
```

### 复杂度分析
- **时间复杂度**:O(n × m) — n = len(s), m = len(p)
  - 具体地说:遍历 n-m+1 个位置,每个位置创建Counter需要 O(m)
  - 如果 s = 10000, p = 100,大约需要 10000 × 100 = 100万次操作
- **空间复杂度**:O(1) — Counter最多存储26个小写字母

### 优缺点
- ✅ 代码简洁,易于理解
- ❌ 每次都重新创建Counter,存在大量重复计算 → 引出滑动窗口优化

---

## ⚡ 解法二:滑动窗口 + Counter(优化)

### 优化思路

使用固定大小的滑动窗口,维护窗口内字符频率。窗口每次右移时:
- 加入右边新字符
- 移出左边旧字符
- O(1)更新频率,O(1)比较(因为只有26个字母)

> 💡 **关键想法**:窗口滑动时只需增量更新,而不是重新统计整个窗口!

### 图解过程

```
示例:s = "cbaebabacd", p = "abc"
p_count = {'a':1, 'b':1, 'c':1}

初始化窗口 [0, 2]:"cba"
  window = {'c':1, 'b':1, 'a':1} ✅ 匹配! → 记录索引 0

滑动窗口:右边+1,左边+1
[1, 3]:"bae"
  window 移出 'c':{'b':1, 'a':1}
  window 加入 'e':{'b':1, 'a':1, 'e':1} ❌

[2, 4]:"aeb"
  window 移出 'b':{'a':1, 'e':1}
  window 加入 'b':{'a':1, 'e':1, 'b':1} ❌

[3, 5]:"eba"
  window 移出 'a':{'e':1, 'b':1}
  window 加入 'a':{'e':1, 'b':1, 'a':1} ❌

[4, 6]:"bab"
  window 移出 'e':{'b':1, 'a':1}
  window 加入 'b':{'b':2, 'a':1} ❌

[5, 7]:"aba"
  window 移出 'b':{'b':1, 'a':1}
  window 加入 'a':{'b':1, 'a':2} ❌

[6, 8]:"bac"
  window 移出 'a':{'b':1, 'a':1}
  window 加入 'c':{'b':1, 'a':1, 'c':1} ✅ 匹配! → 记录索引 6

结果:[0, 6]
```

### Python代码

```python
from typing import List
from collections import Counter


def findAnagrams_v2(s: str, p: str) -> List[int]:
    """
    解法二:滑动窗口 + Counter
    思路:维护固定大小窗口,增量更新字符频率
    """
    result = []
    len_s, len_p = len(s), len(p)

    if len_s < len_p:
        return result

    # 统计p的字符频率
    p_count = Counter(p)
    # 初始化窗口:前len_p个字符
    window = Counter(s[:len_p])

    # 检查第一个窗口
    if window == p_count:
        result.append(0)

    # 滑动窗口:从位置1开始
    for i in range(1, len_s - len_p + 1):
        # 移出左边字符 s[i-1]
        left_char = s[i - 1]
        window[left_char] -= 1
        if window[left_char] == 0:
            del window[left_char]  # 频率为0时删除键,保持字典简洁

        # 加入右边字符 s[i+len_p-1]
        right_char = s[i + len_p - 1]
        window[right_char] = window.get(right_char, 0) + 1

        # 比较当前窗口和p_count
        if window == p_count:
            result.append(i)

    return result


# ✅ 测试
print(findAnagrams_v2("cbaebabacd", "abc"))  # 期望输出:[0, 6]
print(findAnagrams_v2("abab", "ab"))         # 期望输出:[0, 1, 2]
print(findAnagrams_v2("a", "a"))             # 期望输出:[0]
```

### 复杂度分析
- **时间复杂度**:O(n) — n = len(s)
  - 窗口滑动 n-m 次,每次更新 O(1),比较字典 O(26) = O(1)
  - 如果 s = 10000,只需约 10000 次操作
- **空间复杂度**:O(1) — 两个Counter最多各存26个字母

---

## 🚀 解法三:滑动窗口 + 数组计数(最优)

### 优化思路

由于只有26个小写字母,可以用长度为26的数组代替Counter,进一步提升性能:
- 数组索引直接映射字符(ord(c) - ord('a'))
- 数组比较可以用 Python 的列表比较,或者维护一个"匹配字符数"变量

> 💡 **关键想法**:字符集固定时,数组比字典更快!

### 图解过程

```
使用数组表示频率:
p = "abc" → p_count = [1,1,1,0,0,...,0]
             索引:     a b c d e ... z

窗口 "cba" → window = [1,1,1,0,0,...,0] ✅ 数组相等!
```

### Python代码

```python
from typing import List


def findAnagrams_v3(s: str, p: str) -> List[int]:
    """
    解法三:滑动窗口 + 数组计数
    思路:用长度26的数组代替Counter,性能更优
    """
    result = []
    len_s, len_p = len(s), len(p)

    if len_s < len_p:
        return result

    # 初始化频率数组(26个小写字母)
    p_count = [0] * 26
    window = [0] * 26

    # 统计p的字符频率
    for char in p:
        p_count[ord(char) - ord('a')] += 1

    # 初始化窗口:前len_p个字符
    for i in range(len_p):
        window[ord(s[i]) - ord('a')] += 1

    # 检查第一个窗口
    if window == p_count:
        result.append(0)

    # 滑动窗口
    for i in range(1, len_s - len_p + 1):
        # 移出左边字符
        window[ord(s[i - 1]) - ord('a')] -= 1
        # 加入右边字符
        window[ord(s[i + len_p - 1]) - ord('a')] += 1

        # 比较数组
        if window == p_count:
            result.append(i)

    return result


# ✅ 测试
print(findAnagrams_v3("cbaebabacd", "abc"))  # 期望输出:[0, 6]
print(findAnagrams_v3("abab", "ab"))         # 期望输出:[0, 1, 2]
print(findAnagrams_v3("a", "a"))             # 期望输出:[0]
```

### 复杂度分析
- **时间复杂度**:O(n) — 与解法二相同,但常数因子更小
- **空间复杂度**:O(1) — 两个固定长度26的数组

---

## 🐍 Pythonic 写法

利用 Python 的 `collections.Counter` 和列表推导式的简洁写法:

```python
from collections import Counter

def findAnagrams_pythonic(s: str, p: str) -> list[int]:
    """一行流写法:Counter + 列表推导"""
    p_count = Counter(p)
    len_p = len(p)
    return [
        i for i in range(len(s) - len_p + 1)
        if Counter(s[i:i + len_p]) == p_count
    ]

# 测试
print(findAnagrams_pythonic("cbaebabacd", "abc"))  # [0, 6]
```

这个写法本质上是解法一,虽然代码简洁,但性能不如滑动窗口。

> ⚠️ **面试建议**:先写清晰版本展示滑动窗口思路(解法二/三),再提 Pythonic 写法展示语言功底。
> 面试官更看重你的**优化思维**,而非代码行数。

---

## 📊 解法对比

| 维度 | 解法一:暴力统计 | 解法二:滑窗+Counter | 解法三:滑窗+数组 |
|------|--------------|------------------|----------------|
| 时间复杂度 | O(n×m) | O(n) | O(n) |
| 空间复杂度 | O(1) | O(1) | O(1) |
| 代码难度 | 简单 | 中等 | 中等 |
| 面试推荐 | ⭐ | ⭐⭐⭐ | ⭐⭐⭐ |
| 适用场景 | 小数据、快速验证 | 通用推荐 | 性能要求高 |

**面试建议**:先讲解法一建立理解,立刻指出瓶颈,然后优化到解法二展示滑动窗口思维。如果面试官追问性能,可以提解法三的数组优化。

---

## 🎤 面试现场

> 模拟面试中的完整对话流程,帮你练习"边想边说"。

**面试官**:请你解决一下这道题。

**你**:(审题30秒)好的,这道题要求找出字符串s中所有p的异位词的起始索引。异位词就是字符组成相同但顺序不同的字符串。

让我先想一下...我的第一个想法是对每个可能的起始位置截取子串,用Counter比较字符频率,时间复杂度是 O(n×m)。

不过我们可以用**滑动窗口**来优化。维护一个固定大小的窗口,每次滑动只需 O(1) 更新字符频率,整体优化到 O(n)。核心思路是增量更新窗口的字符频率表。

**面试官**:很好,请写一下代码。

**你**:(边写边说)我先用Counter统计p的字符频率作为目标。然后初始化一个窗口包含s的前len(p)个字符。接下来滑动窗口,每次移出左边字符、加入右边字符,比较当前窗口和目标频率是否相等。(写下解法二的代码)

**面试官**:测试一下?

**你**:用示例 "cbaebabacd" 和 "abc" 走一遍。初始窗口是 "cba",Counter是 {c:1,b:1,a:1},和p匹配,记录索引0。然后窗口右移到 "bae",不匹配...最后到 "bac" 再次匹配,记录索引6。结果是 [0,6],正确。

再测一个边界情况:s="a", p="a",只有一个窗口,匹配,返回 [0],也正确。

**面试官**:如果s非常长,p也很长,还能优化吗?

**你**:时间上已经是 O(n) 最优了,因为至少要遍历一遍s。空间上可以进一步优化:因为只有26个小写字母,可以用长度26的数组代替Counter,访问速度更快。或者可以维护一个"匹配字符数"变量,避免每次都比较整个频率表。

### 高频追问

| 追问 | 应答策略 |
|------|---------|
| "还有更优解吗?" | 时间已经是O(n)最优,可以讨论用数组代替Counter的常数优化 |
| "如果字符集很大怎么办?" | Counter更合适,因为数组会浪费空间;或者用哈希表 |
| "能不能只用一个哈希表?" | 可以!维护一个差值表,记录window和p_count的差异,当差值全为0时匹配 |
| "如果要找最长异位词?" | 改为可变窗口,维护最大窗口长度,类似"最长无重复子串" |

---

## 🎓 知识点总结

### Python技巧卡片 🐍

```python
# 技巧1:Counter 直接比较 — 两个Counter相等当且仅当所有键值对相同
from collections import Counter
Counter("abc") == Counter("bca")  # True

# 技巧2:字符转数组索引 — ord()获取ASCII码
index = ord('c') - ord('a')  # index = 2 (对应数组位置)

# 技巧3:删除频率为0的键 — 保持Counter简洁
counter = Counter({'a': 1, 'b': 0})
if counter['b'] == 0:
    del counter['b']  # counter = Counter({'a': 1})

# 技巧4:列表推导式 + 条件过滤
result = [i for i in range(10) if i % 2 == 0]  # [0, 2, 4, 6, 8]
```

### 💡 底层原理(选读)

> **为什么滑动窗口这么快?**
>
> 暴力法对每个位置都要重新统计频率,存在大量重复计算。滑动窗口利用了**增量计算**的思想:
> - 相邻窗口 [i, i+m-1] 和 [i+1, i+m] 只差两个字符
> - 只需更新这两个字符的频率,O(1) 完成
> - 避免了 O(m) 的重复统计
>
> **Counter vs 数组?**
> - Counter 本质是字典,哈希查找 O(1),但有哈希函数和冲突处理的开销
> - 数组直接索引访问,常数因子更小,但需要预知字符集大小
> - 小写字母(26个)用数组,Unicode字符用Counter

### 算法模式卡片 📐

- **模式名称**:固定窗口滑动 + 字符频率匹配
- **适用条件**:
  - 需要在字符串中找满足某种字符频率条件的连续子串
  - 子串长度固定
- **识别关键词**:"异位词"、"排列"、"包含所有字符"、"固定长度子串"
- **模板代码**:

```python
def sliding_window_fixed(s: str, p: str) -> list[int]:
    """固定窗口模板"""
    result = []
    window_size = len(p)
    target = Counter(p)
    window = Counter(s[:window_size])

    if window == target:
        result.append(0)

    for i in range(1, len(s) - window_size + 1):
        # 移出左边
        window[s[i - 1]] -= 1
        if window[s[i - 1]] == 0:
            del window[s[i - 1]]
        # 加入右边
        window[s[i + window_size - 1]] += 1
        # 检查
        if window == target:
            result.append(i)

    return result
```

### 易错点 ⚠️

1. **忘记删除频率为0的键**
   - 错误:`Counter({'a': 1, 'b': 0}) != Counter({'a': 1})`
   - 正确:当字符频率减为0时,`del window[char]`

2. **窗口右边界越界**
   - 错误:`for i in range(len(s) - len_p)` 会少检查最后一个窗口
   - 正确:`for i in range(len(s) - len_p + 1)`

3. **初始窗口遗漏检查**
   - 错误:只在循环中检查,遗漏了第一个窗口[0, len_p-1]
   - 正确:在循环前先检查初始窗口

---

## 🏗️ 工程实战(选读)

> 这个算法思想在真实项目中的应用,让你知道"学了有什么用"。

- **场景1:日志异常检测** — 在系统日志流中检测固定时间窗口内的异常模式。例如:某个错误码在5分钟内出现的频率超过阈值,用滑动窗口统计每个时间窗口的错误码分布。

- **场景2:DNA序列分析** — 生物信息学中查找DNA序列的重复片段或特定基因模式。DNA只有ACGT四种碱基,可以用数组优化的滑动窗口快速匹配。

- **场景3:文本相似度检测** — 检测文章中是否包含某段文字的同义改写(字符组成相同但顺序不同),用于查重或抄袭检测。

---

## 🏋️ 举一反三

完成本课后,试试这些同类题目来巩固知识:

| 题目 | 难度 | 相关知识点 | 提示 |
|------|------|-----------|------|
| LeetCode 567. 字符串的排列 | Medium | 固定窗口+频率匹配 | 和本题几乎相同,只需返回bool |
| LeetCode 76. 最小覆盖子串 | Hard | 可变窗口+频率匹配 | 窗口大小可变,找最小的满足条件的窗口 |
| LeetCode 3. 无重复字符的最长子串 | Medium | 可变窗口+去重 | 改为判断窗口内字符无重复,窗口大小可变 |
| LeetCode 239. 滑动窗口最大值 | Hard | 固定窗口+单调队列 | 用单调队列维护窗口最大值 |

---

## 📝 课后小测

试试这道变体题,不要看答案,自己先想5分钟!

**题目**:给定字符串 s 和 p,判断 s 是否包含 p 的排列(即任意一个异位词)。返回 True 或 False。

例如:s = "eidbaooo", p = "ab",返回 True(因为 s 包含 "ba")

<details>
<summary>💡 提示(实在想不出来再点开)</summary>

这就是找一个异位词,和本题的区别是只需返回布尔值,找到第一个匹配就可以立即返回 True。

</details>

<details>
<summary>✅ 参考答案</summary>

```python
from collections import Counter

def checkInclusion(s: str, p: str) -> bool:
    """判断s是否包含p的排列"""
    if len(s) < len(p):
        return False

    p_count = Counter(p)
    window = Counter(s[:len(p)])

    if window == p_count:
        return True

    for i in range(1, len(s) - len(p) + 1):
        # 滑动窗口
        window[s[i - 1]] -= 1
        if window[s[i - 1]] == 0:
            del window[s[i - 1]]
        window[s[i + len(p) - 1]] += 1

        # 找到匹配立即返回
        if window == p_count:
            return True

    return False
```

核心思路和本题完全相同,只是找到第一个匹配就返回 True,最后返回 False。这就是 LeetCode 567 题!

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
