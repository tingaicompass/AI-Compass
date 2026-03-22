# 📖 第94课:删除无效的括号

> **模块**:图论 | **难度**:Hard ⭐
> **LeetCode 链接**:https://leetcode.cn/problems/remove-invalid-parentheses/
> **前置知识**:第33课(有效的括号)、第89课(岛屿数量)
> **预计学习时间**:40分钟

---

## 🎯 题目描述

给定一个包含括号的字符串 s,删除**最少数量**的无效括号,使得结果字符串有效。返回所有可能的结果(顺序任意)。

一个有效的括号字符串满足:
- 左括号必须用相同类型的右括号闭合
- 左括号必须以正确的顺序闭合

**示例:**
```
输入:s = "()())()"
输出:["(())()","()()()"]
解释:删除 1 个 ')' 有两种方案

输入:s = "(a)())()"
输出:["(a())()","(a)()()"]
解释:可以包含其他字符

输入:s = ")("
输出:[""]
解释:需要删除所有括号
```

**约束条件:**
- 1 <= s.length <= 25
- s 由小写字母、'(' 和 ')' 组成

---

### 🧪 边界用例(面试必考)

| 用例类型 | 输入 | 期望输出 | 考察点 |
|---------|------|---------|--------|
| 已经有效 | "()" | ["()"] | 无需删除 |
| 全部无效 | ")))" | [""] | 删除所有 |
| 包含字母 | "a)b(c" | ["abc"] | 非括号字符保留 |
| 多种方案 | "()()" | ["()()"] | 去重逻辑 |
| 最大长度 | 25 个字符 | — | 性能边界 |

---

## 💡 思路引导

### 生活化比喻

> 想象你在整理一串彩色珠子,其中有些珠子坏掉了(无效括号)。你的目标是**扔掉最少的珠子**,让剩下的珠子能串成一条对称的项链。
>
> 🐌 **笨办法**:用回溯法穷举所有删除方案(删 0 个、删 1 个、删 2 个...),每种方案都检查是否有效。如果字符串长度为 n,总共有 2^n 种删除组合,太慢了!
>
> 🚀 **聪明办法**:用 BFS 逐层扩展!从原始字符串开始,每次只删除 1 个字符,生成所有可能的新字符串。第一次遇到有效字符串时,就是**删除次数最少**的方案。这就像"层层剥洋葱",第一层找到答案就立即停止,不会浪费时间继续搜索更深的层次。

### 关键洞察

**用 BFS 的"层序遍历"特性保证找到的第一批有效解就是最少删除次数的解,避免回溯法的深度优先盲目搜索。**

---

## 🧠 解题思维链

> 这一节模拟你在面试中"从零开始思考"的过程。

### Step 1:理解题目 → 锁定输入输出

- **输入**:包含括号和字母的字符串 s
- **输出**:所有删除最少括号后的有效字符串(可能有多个)
- **限制**:必须删除最少数量的字符,结果不能重复

### Step 2:先想笨办法(暴力法)

用回溯法枚举所有可能的删除组合:
- 删除 0 个字符 → 检查是否有效 → 如果是,返回
- 删除 1 个字符 → 生成 n 个新串,检查每个
- 删除 2 个字符 → 生成 C(n,2) 个新串,检查每个
- ...

- 时间复杂度:O(2^n × n) — 2^n 种删除组合,每种检查有效性 O(n)
- 瓶颈在哪:**无脑枚举所有删除方案**,即使已经找到删除 k 个字符的有效解,仍会继续搜索删除 k+1 个的方案

### Step 3:瓶颈分析 → 优化方向

- 核心问题:回溯法不知道"最少删除几个",只能从 0 开始逐个尝试,无法提前终止
- 优化思路:用 **BFS 层序遍历**!
  - 第 0 层:原始字符串(删除 0 个)
  - 第 1 层:删除 1 个字符的所有可能(共 n 种)
  - 第 2 层:删除 2 个字符的所有可能
  - ...
  - 第一次在某一层找到有效字符串时,立即返回该层的所有有效解

### Step 4:选择武器

- 选用:**BFS + 集合去重**
- 理由:BFS 的层序特性保证第一次找到的有效解就是删除次数最少的,可以立即终止;集合自动去重,避免重复结果。

> 🔑 **模式识别提示**:当题目要求"最少操作次数"且需要返回所有方案,优先考虑 **BFS 层序遍历**(类似"最短路径"思想)

---

## 🔑 解法一:BFS 层序遍历(暴力枚举)

### 思路

从原始字符串开始,每次删除一个字符生成新字符串,用 BFS 层序遍历。第一次遇到有效字符串时,返回该层的所有有效解。

### 图解过程

```
示例:s = "()())()"

第 0 层(删除 0 个):
  "()())()" → 检查:无效(多了一个 ')')

第 1 层(删除 1 个,共 7 个位置):
  删除位置0: ")())()"
  删除位置1: "()())()"
  删除位置2: "(())()"  ← 有效!
  删除位置3: "()()()"  ← 有效!
  删除位置4: "()())("
  删除位置5: "()())"
  删除位置6: "()())("

找到有效解!返回第 1 层的所有有效结果:["(())()","()()()"]
(注意会有重复,需要用 set 去重)
```

### Python代码

```python
from typing import List
from collections import deque


def removeInvalidParentheses(s: str) -> List[str]:
    """
    解法一:BFS 层序遍历
    思路:每次删除一个字符,第一次找到有效串时立即返回该层所有结果
    """

    def is_valid(string: str) -> bool:
        """检查字符串是否有效"""
        count = 0
        for ch in string:
            if ch == '(':
                count += 1
            elif ch == ')':
                count -= 1
                if count < 0:  # 右括号多了
                    return False
        return count == 0  # 左右括号数量相等

    # 如果原始字符串已经有效,直接返回
    if is_valid(s):
        return [s]

    # BFS 初始化
    queue = deque([s])
    visited = {s}  # 已访问集合,避免重复
    result = []
    found = False  # 标记是否找到有效解

    while queue and not found:
        # 处理当前层的所有节点
        level_size = len(queue)
        for _ in range(level_size):
            current = queue.popleft()

            # 检查当前字符串是否有效
            if is_valid(current):
                result.append(current)
                found = True  # 找到了,标记不再扩展下一层
                continue

            # 如果还未找到有效解,继续扩展下一层
            if not found:
                # 尝试删除每个位置的字符
                for i in range(len(current)):
                    # 只删除括号(字母不删)
                    if current[i] not in '()':
                        continue

                    # 生成删除第 i 个字符后的新字符串
                    next_str = current[:i] + current[i+1:]

                    # 去重:避免重复访问
                    if next_str not in visited:
                        visited.add(next_str)
                        queue.append(next_str)

    return result


# ✅ 测试
print(removeInvalidParentheses("()())()"))  # 期望输出:["(())()","()()()"]
print(removeInvalidParentheses("(a)())()"))  # 期望输出:["(a())()","(a)()()"]
print(removeInvalidParentheses(")("))  # 期望输出:[""]
```

### 复杂度分析

- **时间复杂度**:O(2^n × n) — 最坏情况遍历所有子串(2^n),每次检查有效性 O(n)
  - 具体地说:如果 n=25,理论上需要检查 2^25 ≈ 3300 万个子串,但实际会因为去重和提前终止而快很多
- **空间复杂度**:O(2^n) — visited 集合和队列最坏存储所有子串

### 优缺点

- ✅ 逻辑清晰,BFS 保证找到最少删除次数
- ✅ 用 set 自动去重,避免重复结果
- ❌ 时间复杂度高,会生成大量无用中间状态(如删除字母)

---

## ⚡ 解法二:优化 BFS(剪枝 + 预计算)

### 优化思路

解法一会尝试删除所有字符(包括字母),浪费时间。我们可以先计算出**至少需要删除多少个左括号和右括号**,然后只删除括号。

> 💡 **关键想法**:先扫描一遍字符串,计算出需要删除的左右括号数量,BFS 时只删除这些需要删除的括号,避免盲目删除。

### 图解过程

```
示例:s = "()())()"

预计算阶段:
  扫描字符串,统计需要删除的括号数:
  - 左括号多余:0 个
  - 右括号多余:1 个

BFS 阶段:
  只删除 1 个右括号,生成所有可能:
  - 删除位置4的 ')': "(())()"  ← 有效
  - 删除位置5的 ')': "()()()"  ← 有效

剪枝效果:不会尝试删除 '(' 或字母,大大减少搜索空间!
```

### Python代码

```python
from typing import List
from collections import deque


def removeInvalidParenthesesOptimized(s: str) -> List[str]:
    """
    解法二:优化 BFS(剪枝 + 预计算)
    思路:先计算需要删除的左右括号数量,BFS 时只删除必要的括号
    """

    def is_valid(string: str) -> bool:
        """检查字符串是否有效"""
        count = 0
        for ch in string:
            if ch == '(':
                count += 1
            elif ch == ')':
                count -= 1
                if count < 0:
                    return False
        return count == 0

    def calc_remove_count(string: str):
        """计算需要删除的左右括号数量"""
        left_remove = right_remove = 0

        # 从左到右扫描,统计多余的右括号
        for ch in string:
            if ch == '(':
                left_remove += 1
            elif ch == ')':
                if left_remove > 0:
                    left_remove -= 1
                else:
                    right_remove += 1

        return left_remove, right_remove

    # 预计算需要删除的括号数量
    left_remove, right_remove = calc_remove_count(s)

    # BFS 初始化
    queue = deque([(s, left_remove, right_remove)])
    visited = {s}
    result = []

    while queue:
        current, l_rem, r_rem = queue.popleft()

        # 如果已经删除了足够的括号,检查是否有效
        if l_rem == 0 and r_rem == 0:
            if is_valid(current):
                result.append(current)
            continue

        # 尝试删除每个位置的括号
        for i in range(len(current)):
            # 只删除需要删除的括号类型
            if current[i] == '(' and l_rem > 0:
                next_str = current[:i] + current[i+1:]
                if next_str not in visited:
                    visited.add(next_str)
                    queue.append((next_str, l_rem - 1, r_rem))
            elif current[i] == ')' and r_rem > 0:
                next_str = current[:i] + current[i+1:]
                if next_str not in visited:
                    visited.add(next_str)
                    queue.append((next_str, l_rem, r_rem - 1))

    return result if result else [""]


# ✅ 测试
print(removeInvalidParenthesesOptimized("()())()"))  # 期望输出:["(())()","()()()"]
print(removeInvalidParenthesesOptimized("(a)())()"))  # 期望输出:["(a())()","(a)()()"]
```

### 复杂度分析

- **时间复杂度**:O(C(n, k) × n) — k 是需要删除的括号数,C(n,k) 是组合数
  - 相比解法一,剪枝大幅减少了搜索空间
- **空间复杂度**:O(C(n, k)) — 队列和集合大小

---

## 🏆 解法三:回溯 DFS(最优解)

### 优化思路

BFS 需要存储所有中间状态,空间开销大。用 **DFS 回溯** 可以在递归栈上完成,空间更优。

> 💡 **关键想法**:先计算需要删除的左右括号数量,然后用 DFS 回溯,每次决定"删除当前字符"或"保留当前字符",同时维护有效性约束。

### Python代码

```python
from typing import List


def removeInvalidParenthesesDFS(s: str) -> List[str]:
    """
    解法三:回溯 DFS(最优解)
    思路:先预计算需要删除的括号数,用 DFS 回溯生成所有有效方案
    """

    def calc_remove_count(string: str):
        """计算需要删除的左右括号数量"""
        left_remove = right_remove = 0
        for ch in string:
            if ch == '(':
                left_remove += 1
            elif ch == ')':
                if left_remove > 0:
                    left_remove -= 1
                else:
                    right_remove += 1
        return left_remove, right_remove

    def is_valid(string: str) -> bool:
        """检查字符串是否有效"""
        count = 0
        for ch in string:
            if ch == '(':
                count += 1
            elif ch == ')':
                count -= 1
                if count < 0:
                    return False
        return count == 0

    def dfs(index: int, path: str, left_rem: int, right_rem: int,
            open_count: int):
        """
        DFS 回溯生成所有有效方案
        index: 当前处理到的位置
        path: 当前构建的字符串
        left_rem: 还需删除多少个 '('
        right_rem: 还需删除多少个 ')'
        open_count: 当前未匹配的 '(' 数量
        """
        # 递归终止
        if index == len(s):
            if left_rem == 0 and right_rem == 0 and is_valid(path):
                result.add(path)
            return

        ch = s[index]

        # 如果是字母,直接保留
        if ch not in '()':
            dfs(index + 1, path + ch, left_rem, right_rem, open_count)
            return

        # 决策1:删除当前括号(如果还需要删除)
        if ch == '(' and left_rem > 0:
            dfs(index + 1, path, left_rem - 1, right_rem, open_count)
        if ch == ')' and right_rem > 0:
            dfs(index + 1, path, left_rem, right_rem - 1, open_count)

        # 决策2:保留当前括号
        if ch == '(':
            dfs(index + 1, path + ch, left_rem, right_rem, open_count + 1)
        elif ch == ')' and open_count > 0:  # 只有在有未匹配的 '(' 时才能保留 ')'
            dfs(index + 1, path + ch, left_rem, right_rem, open_count - 1)

    # 预计算需要删除的括号数量
    left_remove, right_remove = calc_remove_count(s)

    # DFS 回溯
    result = set()
    dfs(0, "", left_remove, right_remove, 0)

    return list(result) if result else [""]


# ✅ 测试
print(removeInvalidParenthesesDFS("()())()"))  # 期望输出:["(())()","()()()"]
print(removeInvalidParenthesesDFS("(a)())()"))  # 期望输出:["(a())()","(a)()()"]
print(removeInvalidParenthesesDFS(")("))  # 期望输出:[""]
```

### 复杂度分析

- **时间复杂度**:O(2^n) — 每个字符有删/留两种选择,剪枝后实际远小于 2^n
- **空间复杂度**:O(n) — 递归栈深度 O(n) + 结果集合

---

## 🐍 Pythonic 写法

利用 Python 的递归简化代码:

```python
# 简洁版:用列表推导式 + 递归
def removeInvalidParenthesesPythonic(s: str) -> List[str]:
    def remove_min_invalid(s, last_i, last_j, pair):
        count = 0
        for i in range(last_i, len(s)):
            if s[i] == pair[0]:
                count += 1
            if s[i] == pair[1]:
                count -= 1
            if count < 0:
                for j in range(last_j, i + 1):
                    if s[j] == pair[1] and (j == last_j or s[j-1] != pair[1]):
                        remove_min_invalid(s[:j] + s[j+1:], i, j, pair)
                return

        reversed_s = s[::-1]
        if pair[0] == '(':
            remove_min_invalid(reversed_s, 0, 0, (')', '('))
        else:
            result.append(reversed_s)

    result = []
    remove_min_invalid(s, 0, 0, ('(', ')'))
    return result
```

> ⚠️ **面试建议**:先写清晰版本展示思路,再提 Pythonic 写法展示语言功底。
> 面试官更看重你的**思考过程**,而非代码行数。

---

## 📊 解法对比

| 维度 | 解法一:BFS 层序 | 解法二:优化 BFS | 🏆 解法三:DFS 回溯(最优) |
|------|--------------|--------------|---------------------|
| 时间复杂度 | O(2^n × n) | O(C(n,k) × n) | **O(2^n)** ← 剪枝后最快 |
| 空间复杂度 | O(2^n) | O(C(n,k)) | **O(n)** ← 只用递归栈 |
| 代码难度 | 简单 | 中等 | 较难 |
| 面试推荐 | ⭐⭐ | ⭐⭐⭐ | **⭐⭐⭐** ← 首选 |
| 适用场景 | 理解 BFS 思想 | 剪枝优化 | **空间受限、生产环境** |

**为什么 DFS 回溯是最优解**:
- 空间复杂度从 O(2^n) 降到 O(n),在长字符串(n=25)上优势明显
- 通过预计算和剪枝,实际运行时间比 BFS 更快
- 代码更符合"决策树"直觉,易于理解和维护

**面试建议**:
1. 先说明 BFS 思路(按层删除,第一层找到就是最少删除)
2. 立即优化到 🏆 DFS 回溯,强调空间优势和剪枝策略
3. **重点讲解预计算**:先算出需要删除多少左右括号,避免盲目搜索
4. 强调为什么这是最优:空间 O(n) 已达最优,时间上剪枝后接近实际最优

---

## 🎤 面试现场

> 模拟面试中的完整对话流程,帮你练习"边想边说"。

**面试官**:请你解决一下这道题。

**你**:(审题 30 秒)好的,这道题要求删除最少的括号使字符串有效,并返回所有可能的结果。我的第一个想法是用 BFS 层序遍历,每次删除一个字符,第一次遇到有效串时就是删除次数最少的方案。

不过这样会生成很多无用状态(比如删除字母)。我们可以先预计算需要删除多少个左括号和右括号,然后用 **DFS 回溯**,每次决定删除还是保留当前括号,同时维护有效性约束。这样空间复杂度从 O(2^n) 降到 O(n),效率更高。

**面试官**:很好,请写一下代码。

**你**:(边写边说)先扫描一遍字符串,统计出需要删除的左右括号数量。然后 DFS 回溯:
- 如果是字母,直接保留
- 如果是括号,可以选择删除(如果还需要删除)或保留(如果满足有效性约束)
- 用 set 收集所有有效结果,自动去重

**面试官**:测试一下?

**你**:用示例 "()())()" 走一遍。预计算阶段:左括号不需要删除,右括号需要删除 1 个。DFS 回溯时,会在删除位置 4 或位置 5 的 ')' 时生成两个有效解 "(())()" 和 "()()()"。再测一个边界 ")(",需要删除所有括号,返回 [""]。

### 高频追问

| 追问 | 应答策略 |
|------|---------|
| "还有更优解吗?" | 空间已经是 O(n) 最优(递归栈无法避免),时间上可以继续优化剪枝策略,但提升有限。 |
| "如果要求只返回一个结果?" | 可以在找到第一个有效解时立即返回,不继续搜索,时间会快很多。 |
| "如果字符串很长(n>1000)?" | 当前算法不适用,需要考虑启发式搜索或近似算法(如只删除前/后若干个括号)。 |
| "如何处理其他类型的括号?" | 可以扩展为多种括号类型(如 []、{}),用栈维护多种配对关系,核心思路不变。 |

---

## 🎓 知识点总结

### Python技巧卡片 🐍

```python
# 技巧1:字符串切片拼接 — 删除第 i 个字符
s = "hello"
new_s = s[:2] + s[3:]  # "helo" (删除索引2的 'l')

# 技巧2:set 自动去重 — 收集结果
result = set()
result.add("abc")
result.add("abc")  # 重复添加无效
list(result)  # ['abc']

# 技巧3:递归中维护多个状态 — DFS 回溯
def dfs(index, path, left_rem, right_rem, open_count):
    # index: 当前位置
    # path: 当前构建的字符串
    # left_rem: 还需删除多少 '('
    # right_rem: 还需删除多少 ')'
    # open_count: 当前未匹配的 '(' 数量
    pass
```

### 💡 底层原理(选读)

> **为什么 BFS 能保证找到最少删除次数?**
>
> BFS 按层遍历,第 k 层的所有节点都是删除了 k 个字符的结果。第一次在某一层找到有效解时,说明删除 k 个字符就能有效,不可能有更少的方案(因为前面 k-1 层都没找到)。
>
> 这类似于**无权图的最短路径**:
> - 每条边权重为 1(删除一个字符)
> - 起点是原字符串,终点是有效字符串
> - BFS 第一次到达终点时,路径长度就是最短路径

### 算法模式卡片 📐

- **模式名称**:BFS 最短路径(层序遍历找最优解)
- **适用条件**:
  - 需要找"最少操作次数"的所有方案
  - 每步操作代价相同(如删除 1 个字符)
  - 解空间是一个隐式图(每个状态有多个后继状态)
- **识别关键词**:"最少"、"所有方案"、"删除/添加操作"
- **模板代码**:
```python
# BFS 找最少操作次数的所有方案
def bfs_min_operations(start):
    queue = deque([start])
    visited = {start}
    found = False
    result = []

    while queue and not found:
        level_size = len(queue)
        for _ in range(level_size):
            current = queue.popleft()

            if is_target(current):
                result.append(current)
                found = True
                continue

            if not found:
                for next_state in get_neighbors(current):
                    if next_state not in visited:
                        visited.add(next_state)
                        queue.append(next_state)

    return result
```

### 易错点 ⚠️

1. **忘记去重导致结果重复**
   - 错误:用列表直接收集结果 `result.append(path)`
   - 正确:用 set 收集 `result.add(path)`,最后转列表

2. **DFS 剪枝条件不完整**
   - 错误:保留 ')' 时不检查是否有未匹配的 '('
   - 正确:只有 `open_count > 0` 时才能保留 ')'

3. **预计算括号数量逻辑错误**
   - 错误:只统计数量差 `count('(') - count(')')`
   - 正确:从左到右扫描,动态维护左右括号的多余数量

---

## 🏗️ 工程实战(选读)

> 这个算法思想在真实项目中的应用,让你知道"学了有什么用"。

- **场景1:代码格式化工具**
  - IDE 自动修复不匹配的括号:检测出多余的括号并建议删除位置

- **场景2:表达式解析器**
  - 数学表达式引擎容错处理:当用户输入不匹配的括号时,自动修复为最接近的有效表达式

---

## 🏋️ 举一反三

完成本课后,试试这些同类题目来巩固知识:

| 题目 | 难度 | 相关知识点 | 提示 |
|------|------|-----------|------|
| LeetCode 20. 有效的括号 | Easy | 栈匹配 | 本题的基础:用栈检查括号是否有效 |
| LeetCode 921. 使括号有效的最少添加 | Medium | 贪心 | 不删除,只添加括号使其有效 |
| LeetCode 1249. 移除无效的括号(简化版) | Medium | 栈 + 贪心 | 只需删除最少数量,不需返回所有方案 |

---

## 📝 课后小测

试试这道变体题,不要看答案,自己先想 5 分钟!

**题目**:给定一个只包含 '(' 和 ')' 的字符串,每次操作可以**翻转**任意一个字符(将 '(' 变为 ')' 或反之)。求最少翻转次数使字符串有效。

<details>
<summary>💡 提示(实在想不出来再点开)</summary>

从左到右扫描,维护未匹配的 '(' 数量。遇到 ')' 时,如果没有未匹配的 '(',就需要翻转这个 ')' 为 '('。

</details>

<details>
<summary>✅ 参考答案</summary>

```python
def minFlipsToValid(s: str) -> int:
    """最少翻转次数使括号串有效"""
    open_count = 0  # 未匹配的 '(' 数量
    flips = 0  # 翻转次数

    for ch in s:
        if ch == '(':
            open_count += 1
        else:  # ch == ')'
            if open_count > 0:
                open_count -= 1  # 匹配掉一个 '('
            else:
                flips += 1  # 没有 '(' 可匹配,翻转 ')' 为 '('
                open_count += 1

    # 扫描结束后,如果还有未匹配的 '(',需要翻转一半
    flips += open_count // 2

    return flips

# 测试
print(minFlipsToValid("(()))"))  # 输出:1 (翻转最后一个 ')' 为 '(')
print(minFlipsToValid(")))"))  # 输出:2 (翻转前两个 ')' 为 '(')
```

核心思路:贪心策略——从左到右扫描,遇到无法匹配的 ')' 立即翻转为 '(',扫描结束后处理多余的 '('。时间 O(n),空间 O(1)。

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
