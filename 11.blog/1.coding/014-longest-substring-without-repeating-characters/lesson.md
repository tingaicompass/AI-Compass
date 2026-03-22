# 📖 第14课：无重复字符的最长子串

> **模块**:滑动窗口 | **难度**:Medium ⭐⭐⭐
> **LeetCode 链接**:https://leetcode.cn/problems/longest-substring-without-repeating-characters/
> **前置知识**:[第1课:两数之和](../001-two-sum/lesson.md) (哈希表基础)
> **预计学习时间**:25分钟

---

## 🎯 题目描述

给定一个字符串 `s`,请你找出其中**不含有重复字符**的**最长子串**的长度。

**注意**:子串是**连续的**字符序列,不同于子序列。

**示例:**
```
输入:s = "abcabcbb"
输出:3
解释:最长无重复子串是 "abc",长度为 3
```

```
输入:s = "bbbbb"
输出:1
解释:最长无重复子串是 "b",长度为 1
```

```
输入:s = "pwwkew"
输出:3
解释:最长无重复子串是 "wke",长度为 3
     注意答案必须是子串,"pwke" 是子序列而不是子串
```

**约束条件:**
- `0 <= s.length <= 5 * 10^4`
- `s` 由英文字母、数字、符号和空格组成

---

### 🧪 边界用例(面试必考)

| 用例类型 | 输入 | 期望输出 | 考察点 |
|---------|------|---------|--------|
| 空字符串 | `""` | `0` | 边界处理 |
| 单字符 | `"a"` | `1` | 最小输入 |
| 全部相同 | `"aaaa"` | `1` | 极端重复 |
| 全部不同 | `"abcdef"` | `6` | 无重复 |
| 重复在末尾 | `"abca"` | `3` | 窗口收缩 |
| 空格和特殊字符 | `"a b!c"` | `4` | 字符集处理 |
| 大规模 | `n=50000` | — | 性能边界 O(n) |

---

## 💡 思路引导

### 生活化比喻
> 想象你在阅读一本书,用一个**透明的窗户**在书页上滑动,窗户里的文字不能有重复字。
>
> 🐌 **笨办法**:检查所有可能的子串——从第一个字符开始,尝试长度为1、2、3...的子串,检查每个是否有重复。这需要三重循环:起点 × 终点 × 检查重复,O(n³) 复杂度太高!
>
> 🤔 **进阶想法**:用哈希表优化重复检查,O(n²) 还是太慢。
>
> 🚀 **聪明办法**:用一个**会伸缩的窗户**(滑动窗口)!
> - 右手向右滑动窗户右边界,不断"扩大窗口",尽可能多地容纳字符
> - 一旦发现窗口里有重复字符,左手就收缩窗口左边界,直到重复消失
> - 记录过程中窗口的最大长度
> - **只需扫描一遍**,O(n) 搞定!

### 关键洞察
**用可伸缩的滑动窗口 + 哈希表维护窗口内字符 → 右指针扩展,左指针收缩,O(n)一次遍历!**

---

## 🧠 解题思维链

> 这一节模拟你在面试中"从零开始思考"的过程。

### Step 1:理解题目 → 锁定输入输出
- **输入**:字符串 `s`,可能包含各种字符
- **输出**:最长无重复子串的**长度**(不是子串本身)
- **限制**:必须是连续的子串,不能跳跃

### Step 2:先想笨办法(暴力枚举)
最直接的思路:枚举所有可能的子串,检查是否有重复。
```python
max_len = 0
for i in range(len(s)):           # O(n)
    for j in range(i, len(s)):    # O(n)
        substring = s[i:j+1]
        if len(substring) == len(set(substring)):  # O(n)
            max_len = max(max_len, len(substring))
```
- 时间复杂度:O(n³) — 三重循环
- 瓶颈在哪:**每次都重新检查子串**,大量重复计算

### Step 3:瓶颈分析 → 优化方向
- 核心问题:暴力法中,`s[i:j]` 无重复不代表 `s[i:j+1]` 需要重新检查整个子串
- 优化思路:能否**动态维护窗口内的字符**,增量式地检查?

**关键洞察**:
- 维护一个**窗口** `[left, right]`,用哈希表/集合记录窗口内的字符
- **右指针** `right` 不断扩展:
  - 如果 `s[right]` 不在窗口内,加入,更新最大长度
  - 如果 `s[right]` 在窗口内,说明重复了!
- **左指针** `left` 收缩:
  - 从左边移除字符,直到重复消失

### Step 4:选择武器
- 选用:**滑动窗口 + 哈希表(或集合)**
- 理由:
  - 滑动窗口避免重复枚举,O(n) 遍历
  - 哈希表/集合 O(1) 检查字符是否在窗口内
  - 左右指针配合,动态维护无重复窗口

> 🔑 **模式识别提示**:当题目要求"连续子串/子数组的最值"+"满足某条件",优先考虑"滑动窗口"

---

## 🔑 解法一:暴力枚举(三重循环)

### 思路
枚举所有可能的子串,检查每个是否有重复字符,记录最大长度。

### Python代码

```python
def lengthOfLongestSubstring_brute(s: str) -> int:
    """
    解法一:暴力枚举
    思路:枚举所有子串,检查是否有重复
    """
    n = len(s)
    max_len = 0

    for i in range(n):
        for j in range(i, n):
            # 检查 s[i:j+1] 是否有重复字符
            substring = s[i:j+1]
            if len(substring) == len(set(substring)):  # 用 set 去重检查
                max_len = max(max_len, len(substring))
            else:
                break  # 一旦重复,后续更长的子串也会重复,跳出

    return max_len


# ✅ 测试
print(lengthOfLongestSubstring_brute("abcabcbb"))  # 期望: 3
print(lengthOfLongestSubstring_brute("bbbbb"))     # 期望: 1
print(lengthOfLongestSubstring_brute("pwwkew"))    # 期望: 3
```

### 复杂度分析
- **时间复杂度**:O(n³) — 两重循环 O(n²) × 检查重复 O(n)
  - 具体地说:如果 n=1000,可能需要 10亿次操作,非常慢!
- **空间复杂度**:O(n) — set 去重需要存储子串的字符

### 优缺点
- ✅ 思路简单,易于理解
- ❌ **时间复杂度太高**,大规模数据会超时
- ❌ 有大量重复计算

---

## ⚡ 解法二:滑动窗口 + 集合(最优解)

### 优化思路
用**滑动窗口**避免重复枚举:
- 维护一个窗口 `[left, right]`,用集合 `window` 记录窗口内的字符
- **右指针**不断右移,尝试扩大窗口:
  - 如果 `s[right]` 不在 `window` 中,加入,更新最大长度
  - 如果 `s[right]` 在 `window` 中,进入收缩阶段
- **左指针**收缩窗口:
  - 从 `window` 中移除 `s[left]`,`left++`
  - 直到 `s[right]` 不再重复

> 💡 **关键想法**:窗口动态伸缩,右指针扩展探索,左指针收缩消除重复,一次遍历完成!

### 图解过程

```
输入:s = "abcabcbb"

初始:left = 0, right = 0, window = {}, max_len = 0

Step 1: right=0, s[0]='a' 不在 window
  window = {'a'}, max_len = 1
  [a]bcabcbb
   ↑
   L,R

Step 2: right=1, s[1]='b' 不在 window
  window = {'a','b'}, max_len = 2
  [ab]cabcbb
   ↑ ↑
   L R

Step 3: right=2, s[2]='c' 不在 window
  window = {'a','b','c'}, max_len = 3
  [abc]abcbb
   ↑  ↑
   L  R

Step 4: right=3, s[3]='a' 在 window 中!重复!
  收缩窗口:移除 s[left]='a', left++
  window = {'b','c'}
  a[bc]abcbb
     ↑ ↑
     L R

  再尝试加入 s[3]='a'
  window = {'b','c','a'}, max_len 保持 3
  a[bca]bcbb
     ↑  ↑
     L  R

Step 5: right=4, s[4]='b' 在 window 中!重复!
  收缩窗口:移除 s[left]='b', left++
  window = {'c','a'}
  ab[ca]bcbb
      ↑ ↑
      L R

  再尝试加入 s[4]='b'
  window = {'c','a','b'}, max_len 保持 3
  ab[cab]cbb
      ↑  ↑
      L  R

Step 6: right=5, s[5]='c' 在 window 中!重复!
  收缩窗口:移除 s[left]='c', left++, 移除 'a', left++
  window = {'b'}
  abca[b]cbb
        ↑↑
        LR

  再尝试加入 s[5]='c'
  window = {'b','c'}, max_len 保持 3
  abca[bc]bb
        ↑ ↑
        L R

Step 7-8: 类似处理...

最终:max_len = 3 (子串 "abc")
```

### Python代码

```python
def lengthOfLongestSubstring(s: str) -> int:
    """
    解法二:滑动窗口 + 集合
    思路:右指针扩展,左指针收缩,维护无重复窗口
    """
    # 边界:空字符串
    if not s:
        return 0

    left = 0  # 窗口左边界
    max_len = 0  # 记录最大长度
    window = set()  # 窗口内的字符集合

    # 右指针遍历整个字符串
    for right in range(len(s)):
        # 如果 s[right] 在窗口内,收缩左边界直到不重复
        while s[right] in window:
            window.remove(s[left])  # 移除左边界字符
            left += 1  # 左指针右移

        # 将当前字符加入窗口
        window.add(s[right])

        # 更新最大长度
        max_len = max(max_len, right - left + 1)

    return max_len


# ✅ 测试
print(lengthOfLongestSubstring("abcabcbb"))  # 期望: 3
print(lengthOfLongestSubstring("bbbbb"))     # 期望: 1
print(lengthOfLongestSubstring("pwwkew"))    # 期望: 3
print(lengthOfLongestSubstring(""))          # 期望: 0
print(lengthOfLongestSubstring("abcdef"))    # 期望: 6
```

### 复杂度分析
- **时间复杂度**:O(n) — 右指针遍历一次 O(n),左指针最多也遍历一次 O(n),总共 O(2n) = O(n)
  - 具体地说:如果 n=50000,最多需要 100000 次操作,非常快!
- **空间复杂度**:O(min(n, m)) — m 是字符集大小(如 ASCII 128),窗口内最多存储 m 个不同字符

---

## 🚀 解法三:滑动窗口 + 哈希表优化(跳跃式收缩)

### 优化思路
解法二中,左指针每次只移动一格,能否**一步跳到重复字符的下一个位置**?

用哈希表记录**每个字符最后出现的位置**:
- 当 `s[right]` 重复时,直接将 `left` 跳到 `last_pos[s[right]] + 1`
- 无需逐个移除字符,一步到位!

> 💡 **关键想法**:哈希表记录字符位置,遇到重复时直接跳跃,更高效!

### Python代码

```python
def lengthOfLongestSubstring_optimized(s: str) -> int:
    """
    解法三:滑动窗口 + 哈希表(跳跃式收缩)
    思路:记录字符最后位置,遇到重复直接跳跃
    """
    left = 0
    max_len = 0
    char_index = {}  # 记录字符最后出现的索引

    for right in range(len(s)):
        # 如果 s[right] 之前出现过,且在当前窗口内
        if s[right] in char_index and char_index[s[right]] >= left:
            # 直接跳到重复字符的下一个位置
            left = char_index[s[right]] + 1

        # 更新字符的最后位置
        char_index[s[right]] = right

        # 更新最大长度
        max_len = max(max_len, right - left + 1)

    return max_len


# ✅ 测试
print(lengthOfLongestSubstring_optimized("abcabcbb"))  # 期望: 3
print(lengthOfLongestSubstring_optimized("bbbbb"))     # 期望: 1
print(lengthOfLongestSubstring_optimized("pwwkew"))    # 期望: 3
print(lengthOfLongestSubstring_optimized("abba"))      # 期望: 2
```

### 复杂度分析
- **时间复杂度**:O(n) — 只需一次遍历,每个字符访问一次
- **空间复杂度**:O(min(n, m)) — 哈希表最多存储 m 个字符

---

## 🐍 Pythonic 写法

利用字典的 `get` 方法和 `max` 函数,让代码更简洁:

```python
def lengthOfLongestSubstring_pythonic(s: str) -> int:
    """
    Pythonic 写法:简洁版滑动窗口
    """
    char_index = {}
    left = max_len = 0

    for right, char in enumerate(s):
        # 如果字符重复且在当前窗口内,跳跃
        if char in char_index and char_index[char] >= left:
            left = char_index[char] + 1

        char_index[char] = right
        max_len = max(max_len, right - left + 1)

    return max_len


# 更简洁的写法(单行更新)
def lengthOfLongestSubstring_oneline(s: str) -> int:
    char_index, left, max_len = {}, 0, 0
    for right, char in enumerate(s):
        left = max(left, char_index.get(char, -1) + 1)
        char_index[char] = right
        max_len = max(max_len, right - left + 1)
    return max_len
```

> ⚠️ **面试建议**:优先使用解法二(集合版),代码最清晰,逻辑最直观。解法三虽然更优化,但可以作为进阶优化点在面试中提出。

---

## 📊 解法对比

| 维度 | 解法一:暴力枚举 | 解法二:滑动窗口+集合 | 解法三:滑动窗口+哈希表 |
|------|--------------|-------------------|---------------------|
| 时间复杂度 | O(n³) | O(n) | O(n) |
| 空间复杂度 | O(n) | O(min(n,m)) | O(min(n,m)) |
| 代码难度 | 简单 | 中等 | 中等 |
| 面试推荐 | ⭐ | ⭐⭐⭐ | ⭐⭐⭐ |
| 适用场景 | 小规模数据 | 通用最优解 | 更优化的版本 |

**面试建议**:
1. 先提出暴力解法,展示你能快速想出可行方案
2. 分析瓶颈,引出滑动窗口优化
3. 实现解法二(集合版),清晰讲解窗口伸缩逻辑
4. 如果时间充裕,提出解法三的优化思路

---

## 🎤 面试现场

> 模拟面试中的完整对话流程,帮你练习"边想边说"。

**面试官**:请你找出字符串中最长的无重复子串的长度。

**你**:(审题30秒)好的,这道题要求找**连续的**无重复字符子串的最大长度。让我先想一下...

最直观的思路是**暴力枚举**:枚举所有子串,检查是否有重复。时间复杂度 O(n³),太慢了。

我想到可以用**滑动窗口**优化:
- 维护一个动态窗口 `[left, right]`,用集合记录窗口内的字符
- **右指针**不断右移,扩大窗口,尝试容纳更多字符
- 当遇到重复字符时,**左指针**收缩窗口,移除字符直到不重复
- 记录过程中窗口的最大长度

这样只需一次遍历,时间复杂度 O(n),空间复杂度 O(字符集大小)。

**面试官**:很好,请写一下代码。

**你**:(边写边说)
```python
def lengthOfLongestSubstring(s):
    if not s:
        return 0

    left = 0
    max_len = 0
    window = set()  # 窗口内的字符

    for right in range(len(s)):
        # 如果 s[right] 重复,收缩左边界
        while s[right] in window:
            window.remove(s[left])
            left += 1

        # 加入当前字符
        window.add(s[right])

        # 更新最大长度
        max_len = max(max_len, right - left + 1)

    return max_len
```

关键点:
1. 用 `set` 快速检查字符是否在窗口内(O(1))
2. 右指针扩展,左指针收缩,动态维护无重复窗口
3. 每次更新最大长度

**面试官**:为什么用 while 而不是 if?

**你**:因为可能需要**连续移除多个字符**才能消除重复。

例如 `s = "abba"`,当 `right=3` 遇到第二个 'a' 时:
- 窗口是 `[abb]`,需要移除 'a' 和 'b' 两个字符
- 如果只用 `if`,只移除一次,还是会重复
- 用 `while` 确保左指针移动到重复字符的下一个位置

**面试官**:能否优化左指针的移动?

**你**:可以!用**哈希表记录字符的最后位置**,遇到重复时直接跳跃:
```python
def optimized(s):
    left, max_len = 0, 0
    char_index = {}

    for right, char in enumerate(s):
        if char in char_index and char_index[char] >= left:
            left = char_index[char] + 1  # 直接跳到重复字符的下一个位置

        char_index[char] = right
        max_len = max(max_len, right - left + 1)

    return max_len
```

这样避免了 while 循环,更高效。

### 高频追问

| 追问 | 应答策略 |
|------|---------|
| "如果要返回最长子串本身,而不是长度?" | 在更新 `max_len` 时,同时记录起始位置 `start = left`,最后返回 `s[start:start+max_len]`。 |
| "如果允许最多 k 个重复字符?" | 改用哈希表计数,当窗口内重复字符数 > k 时收缩左边界。这是 LeetCode 340 的原题。 |
| "时间能否优化到 O(log n)?" | 不能。必须遍历所有字符至少一次才能找到最长子串,O(n) 是最优的。 |
| "空间能否优化到 O(1)?" | 如果字符集有限(如只有小写字母),可以用固定大小的数组代替哈希表,但本质上还是 O(字符集大小)。 |

---

## 🎓 知识点总结

### Python技巧卡片 🐍
```python
# 技巧1:set 的快速增删查
window = set()
window.add('a')       # O(1) 添加
window.remove('a')    # O(1) 删除
'a' in window         # O(1) 检查

# 技巧2:dict.get 提供默认值
char_index.get(char, -1)  # 如果 char 不存在,返回 -1

# 技巧3:enumerate 优雅地获取索引和值
for index, char in enumerate(s):
    print(f"索引 {index}: 字符 {char}")

# 技巧4:max 的灵活使用
max_len = max(max_len, new_len)  # 更新最大值
left = max(left, new_left)       # 取较大值
```

### 💡 底层原理(选读)

> **为什么滑动窗口能优化到 O(n)?**
>
> 关键在于**避免重复计算**:
> - 暴力法:每次检查 `s[i:j]` 都要重新扫描整个子串
> - 滑动窗口:窗口从 `[i, j]` 扩展到 `[i, j+1]`,只需检查 `s[j+1]` 是否在窗口内,O(1) 完成
> - 左右指针**单调移动**,每个字符最多被访问两次(加入和移除),总共 O(2n) = O(n)
>
> **为什么用 set/dict 而不是数组?**
>
> - set/dict 提供 **O(1) 的查找、插入、删除**
> - 如果用数组记录,每次检查重复需要 O(n) 扫描
> - 字符集可能很大(Unicode),用数组会浪费空间
> - 但如果字符集固定且小(如只有小写字母 26 个),可以用长度 26 的数组优化常数
>
> **滑动窗口的本质**
>
> - 滑动窗口是**双指针的高级应用**
> - 核心思想:**用两个指针维护一个动态区间,避免重复枚举**
> - 通用模板:右指针扩展探索,左指针收缩调整,动态维护窗口内的性质

### 算法模式卡片 📐
- **模式名称**:滑动窗口
- **适用条件**:
  - 连续子串/子数组的最值问题
  - 需要满足某种条件(无重复、和为K、包含特定字符等)
  - 暴力枚举会导致 O(n²) 或更高复杂度
- **识别关键词**:"最长/最短连续子串"、"满足条件的子数组"、"窗口"、"连续"
- **模板代码**:
```python
def sliding_window_template(s: str) -> int:
    """滑动窗口通用模板"""
    left = 0
    result = 0  # 记录结果(最大/最小长度、计数等)
    window = {}  # 维护窗口状态(字符计数、集合等)

    for right in range(len(s)):
        # 1. 将 s[right] 加入窗口
        window[s[right]] = window.get(s[right], 0) + 1

        # 2. 判断窗口是否需要收缩
        while not is_valid(window):  # 自定义条件:窗口不满足要求
            # 移除 s[left]
            window[s[left]] -= 1
            if window[s[left]] == 0:
                del window[s[left]]
            left += 1

        # 3. 更新结果(在窗口合法时)
        result = max(result, right - left + 1)

    return result
```

### 易错点 ⚠️
1. **边界条件:空字符串**
   - ❌ 错误:直接进入循环,没有处理空字符串
   - ⚠️ 为什么错:空字符串时 `range(0)` 不会执行,但应该明确返回 0
   - ✅ 正确:开头加 `if not s: return 0`

2. **窗口收缩条件:用 if 而不是 while**
   - ❌ 错误:`if s[right] in window:`
   - ⚠️ 为什么错:可能需要连续移除多个字符才能消除重复
   - ✅ 正确:`while s[right] in window:`

3. **哈希表优化时忘记检查字符是否在当前窗口内**
   - ❌ 错误:`if char in char_index: left = char_index[char] + 1`
   - ⚠️ 为什么错:字符可能在窗口左边界之前出现过,不应该跳跃
   - ✅ 正确:`if char in char_index and char_index[char] >= left:`

4. **窗口长度计算错误**
   - ❌ 错误:`max_len = right - left`
   - ⚠️ 为什么错:长度应该是 索引差 + 1
   - ✅ 正确:`max_len = right - left + 1`

---

## 🏗️ 工程实战(选读)

> 这个算法思想在真实项目中的应用,让你知道"学了有什么用"。

- **场景1:网络数据包去重** — 网络监控系统实时接收数据包,需要找出最长的无重复 ID 的数据包序列,用于检测异常流量。滑动窗口可以高效实时处理。

- **场景2:基因序列分析** — 生物信息学中,需要找到 DNA 序列中最长的无重复碱基片段,用于基因突变检测。滑动窗口可以在 O(n) 时间内完成。

- **场景3:推荐系统去重** — 推荐算法生成的内容列表需要去重,找出最长的无重复内容序列展示给用户。滑动窗口可以在线实时处理推荐流。

- **场景4:日志分析** — 运维系统分析日志文件,找出最长的无重复错误类型的时间窗口,用于定位问题根源。

---

## 🏋️ 举一反三

完成本课后,试试这些同类题目来巩固知识:

| 题目 | 难度 | 相关知识点 | 提示 |
|------|------|-----------|------|
| LeetCode 159. 至多包含两个不同字符的最长子串 | Medium | 滑动窗口+计数 | 用哈希表计数,当不同字符 > 2 时收缩 |
| LeetCode 340. 至多包含 K 个不同字符的最长子串 | Hard | 滑动窗口+计数 | 159 的通用版本 |
| LeetCode 424. 替换后的最长重复字符 | Medium | 滑动窗口+贪心 | 窗口内非最多字符的数量 <= k |
| LeetCode 76. 最小覆盖子串 | Hard | 滑动窗口模板题 | 需要包含所有目标字符,求最短窗口 |

---

## 📝 课后小测

试试这道变体题,不要看答案,自己先想5分钟!

**题目**:给定字符串 `s` 和整数 `k`,找出**至多**包含 k 个不同字符的最长子串长度。

例如:`s = "eceba"`, `k = 2`,输出 3 (子串 "ece")

<details>
<summary>💡 提示(实在想不出来再点开)</summary>

用哈希表计数记录窗口内每个字符的数量,当不同字符种类 > k 时收缩左边界!

</details>

<details>
<summary>✅ 参考答案</summary>

```python
def lengthOfLongestSubstringKDistinct(s: str, k: int) -> int:
    """
    至多 k 个不同字符的最长子串
    核心:滑动窗口 + 哈希表计数
    """
    if k == 0:
        return 0

    left = 0
    max_len = 0
    char_count = {}  # 记录窗口内每个字符的数量

    for right in range(len(s)):
        # 将 s[right] 加入窗口
        char_count[s[right]] = char_count.get(s[right], 0) + 1

        # 当不同字符数量 > k 时,收缩左边界
        while len(char_count) > k:
            char_count[s[left]] -= 1
            if char_count[s[left]] == 0:
                del char_count[s[left]]  # 字符数量为0时删除
            left += 1

        # 更新最大长度
        max_len = max(max_len, right - left + 1)

    return max_len


# 测试
print(lengthOfLongestSubstringKDistinct("eceba", 2))  # 期望: 3 ("ece")
print(lengthOfLongestSubstringKDistinct("aa", 1))     # 期望: 2 ("aa")
```

**核心思路**:
- 用哈希表 `char_count` 记录窗口内**每个字符的数量**
- `len(char_count)` 就是不同字符的种类数
- 当种类数 > k 时,收缩左边界:
  - 减少 `char_count[s[left]]`
  - 如果数量变为 0,从哈希表中删除(减少种类数)
- 与本题相比,只是把"无重复(种类数 = 窗口长度)"改为"种类数 <= k"

这展示了滑动窗口的**通用性**——只需调整收缩条件,就能解决各种变体问题!

</details>

---

## 🎉 恭喜开启滑动窗口模块！

你已经学会了滑动窗口的**核心思想**:
- ✅ 动态窗口维护(右指针扩展,左指针收缩)
- ✅ 用哈希表/集合优化窗口状态检查
- ✅ 避免重复枚举,O(n) 一次遍历

**滑动窗口是字符串/数组题的必杀技**,接下来的3道题会让你更加熟练这个技巧,加油!💪

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
